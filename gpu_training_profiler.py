#!/usr/bin/env python3
"""
GPU Training Profiler - eBPF-based profiler for ML training workloads
Monitors: GPU driver activity, disk I/O, network (NCCL), CPU usage
"""

from bcc import BPF
import time
import ctypes as ct
from collections import defaultdict
import argparse
import sys

# BPF program
bpf_program = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <net/sock.h>

// Event types
#define EVENT_GPU_IOCTL 1
#define EVENT_DISK_READ 2
#define EVENT_NETWORK_SEND 3
#define EVENT_NETWORK_RECV 4

struct event_t {
    u64 ts;           // timestamp (ns)
    u32 pid;          // process ID
    u32 tid;          // thread ID
    u32 type;         // event type
    u64 latency_ns;   // operation latency
    u64 size;         // bytes (for I/O)
    char comm[16];    // process name
};

BPF_PERF_OUTPUT(events);
BPF_HASH(start_times, u64, u64);  // Track operation start times
BPF_HASH(gpu_ioctl_count, u32, u64);
BPF_HASH(disk_bytes, u32, u64);
BPF_HASH(net_bytes_sent, u32, u64);
BPF_HASH(net_bytes_recv, u32, u64);

// Hook: ioctl entry (GPU driver communication)
int trace_ioctl_entry(struct pt_regs *ctx, int fd, unsigned long cmd) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    
    // Store start time for latency calculation
    start_times.update(&pid_tgid, &ts);
    
    return 0;
}

// Hook: ioctl return (measure latency)
int trace_ioctl_return(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 latency = end_ts - *start_ts;
    
    // Count GPU ioctls (proxy for GPU activity)
    u32 pid = pid_tgid >> 32;
    u64 *count = gpu_ioctl_count.lookup(&pid);
    u64 new_count = count ? *count + 1 : 1;
    gpu_ioctl_count.update(&pid, &new_count);
    
    // Send event for high-latency ioctls (> 1ms, likely synchronization)
    if (latency > 1000000) {
        struct event_t event = {};
        event.ts = end_ts;
        event.pid = pid;
        event.tid = pid_tgid & 0xFFFFFFFF;
        event.type = EVENT_GPU_IOCTL;
        event.latency_ns = latency;
        bpf_get_current_comm(&event.comm, sizeof(event.comm));
        events.perf_submit(ctx, &event, sizeof(event));
    }
    
    start_times.delete(&pid_tgid);
    return 0;
}

// Hook: read syscall entry
int trace_read_entry(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_times.update(&pid_tgid, &ts);
    return 0;
}

// Hook: read syscall return
int trace_read_return(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 latency = end_ts - *start_ts;
    ssize_t bytes_read = PT_REGS_RC(ctx);
    
    if (bytes_read > 0) {
        u32 pid = pid_tgid >> 32;
        u64 *total = disk_bytes.lookup(&pid);
        u64 new_total = total ? *total + bytes_read : bytes_read;
        disk_bytes.update(&pid, &new_total);
        
        // Report slow reads (> 10ms)
        if (latency > 10000000) {
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.tid = pid_tgid & 0xFFFFFFFF;
            event.type = EVENT_DISK_READ;
            event.latency_ns = latency;
            event.size = bytes_read;
            bpf_get_current_comm(&event.comm, sizeof(event.comm));
            events.perf_submit(ctx, &event, sizeof(event));
        }
    }
    
    start_times.delete(&pid_tgid);
    return 0;
}

// Hook: sendmsg (network send, catches NCCL)
int trace_sendmsg_entry(struct pt_regs *ctx, int fd, struct msghdr *msg, int flags) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_times.update(&pid_tgid, &ts);
    return 0;
}

int trace_sendmsg_return(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 latency = end_ts - *start_ts;
    ssize_t bytes_sent = PT_REGS_RC(ctx);
    
    if (bytes_sent > 0) {
        u32 pid = pid_tgid >> 32;
        u64 *total = net_bytes_sent.lookup(&pid);
        u64 new_total = total ? *total + bytes_sent : bytes_sent;
        net_bytes_sent.update(&pid, &new_total);
        
        // Report large sends (likely NCCL all-reduce)
        if (bytes_sent > 1024 * 1024) {  // > 1MB
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.tid = pid_tgid & 0xFFFFFFFF;
            event.type = EVENT_NETWORK_SEND;
            event.latency_ns = latency;
            event.size = bytes_sent;
            bpf_get_current_comm(&event.comm, sizeof(event.comm));
            events.perf_submit(ctx, &event, sizeof(event));
        }
    }
    
    start_times.delete(&pid_tgid);
    return 0;
}

// Hook: recvmsg (network receive)
int trace_recvmsg_entry(struct pt_regs *ctx, int fd, struct msghdr *msg, int flags) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_times.update(&pid_tgid, &ts);
    return 0;
}

int trace_recvmsg_return(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 latency = end_ts - *start_ts;
    ssize_t bytes_recv = PT_REGS_RC(ctx);
    
    if (bytes_recv > 0) {
        u32 pid = pid_tgid >> 32;
        u64 *total = net_bytes_recv.lookup(&pid);
        u64 new_total = total ? *total + bytes_recv : bytes_recv;
        net_bytes_recv.update(&pid, &new_total);
        
        // Report large receives
        if (bytes_recv > 1024 * 1024) {  // > 1MB
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.tid = pid_tgid & 0xFFFFFFFF;
            event.type = EVENT_NETWORK_RECV;
            event.latency_ns = latency;
            event.size = bytes_recv;
            bpf_get_current_comm(&event.comm, sizeof(event.comm));
            events.perf_submit(ctx, &event, sizeof(event));
        }
    }
    
    start_times.delete(&pid_tgid);
    return 0;
}
"""

class Event(ct.Structure):
    _fields_ = [
        ("ts", ct.c_ulonglong),
        ("pid", ct.c_uint),
        ("tid", ct.c_uint),
        ("type", ct.c_uint),
        ("latency_ns", ct.c_ulonglong),
        ("size", ct.c_ulonglong),
        ("comm", ct.c_char * 16),
    ]

class GPUTrainingProfiler:
    EVENT_NAMES = {
        1: "GPU_IOCTL",
        2: "DISK_READ",
        3: "NETWORK_SEND",
        4: "NETWORK_RECV",
    }
    
    def __init__(self, target_pid=None, duration=10):
        self.target_pid = target_pid
        self.duration = duration
        self.events = []
        self.start_time = None
        
        # Statistics
        self.stats = {
            'gpu_ioctls': defaultdict(int),
            'disk_bytes': defaultdict(int),
            'net_sent': defaultdict(int),
            'net_recv': defaultdict(int),
            'gpu_sync_latency': [],
            'disk_latency': [],
            'net_send_latency': [],
            'net_recv_latency': [],
        }
        
        print("Loading eBPF program...")
        self.b = BPF(text=bpf_program)
        
        # Attach to syscalls
        self.b.attach_kprobe(event=self.b.get_syscall_fnname("ioctl"), fn_name="trace_ioctl_entry")
        self.b.attach_kretprobe(event=self.b.get_syscall_fnname("ioctl"), fn_name="trace_ioctl_return")
        
        self.b.attach_kprobe(event=self.b.get_syscall_fnname("read"), fn_name="trace_read_entry")
        self.b.attach_kretprobe(event=self.b.get_syscall_fnname("read"), fn_name="trace_read_return")
        
        self.b.attach_kprobe(event=self.b.get_syscall_fnname("sendmsg"), fn_name="trace_sendmsg_entry")
        self.b.attach_kretprobe(event=self.b.get_syscall_fnname("sendmsg"), fn_name="trace_sendmsg_return")
        
        self.b.attach_kprobe(event=self.b.get_syscall_fnname("recvmsg"), fn_name="trace_recvmsg_entry")
        self.b.attach_kretprobe(event=self.b.get_syscall_fnname("recvmsg"), fn_name="trace_recvmsg_return")
        
        print("✅ eBPF program loaded successfully")
        print(f"Monitoring for {duration} seconds...")
        if target_pid:
            print(f"Filtering for PID: {target_pid}")
        print()
    
    def event_callback(self, cpu, data, size):
        event = ct.cast(data, ct.POINTER(Event)).contents
        
        # Filter by PID if specified
        if self.target_pid and event.pid != self.target_pid:
            return
        
        event_type = self.EVENT_NAMES.get(event.type, "UNKNOWN")
        latency_ms = event.latency_ns / 1_000_000
        
        # Store for analysis
        if event.type == 1:  # GPU_IOCTL
            self.stats['gpu_sync_latency'].append(latency_ms)
        elif event.type == 2:  # DISK_READ
            self.stats['disk_latency'].append(latency_ms)
        elif event.type == 3:  # NETWORK_SEND
            self.stats['net_send_latency'].append(latency_ms)
        elif event.type == 4:  # NETWORK_RECV
            self.stats['net_recv_latency'].append(latency_ms)
        
        # Print significant events
        if event.type in [3, 4]:  # Network events
            size_mb = event.size / (1024 * 1024)
            print(f"[{event_type}] PID {event.pid} ({event.comm.decode()}) "
                  f"- {size_mb:.2f} MB in {latency_ms:.2f}ms "
                  f"({size_mb / (latency_ms / 1000):.2f} MB/s)")
    
    def collect_stats(self):
        """Collect aggregated statistics from BPF maps"""
        # GPU ioctl counts
        for k, v in self.b["gpu_ioctl_count"].items():
            self.stats['gpu_ioctls'][k.value] = v.value
        
        # Disk bytes
        for k, v in self.b["disk_bytes"].items():
            self.stats['disk_bytes'][k.value] = v.value
        
        # Network bytes
        for k, v in self.b["net_bytes_sent"].items():
            self.stats['net_sent'][k.value] = v.value
        
        for k, v in self.b["net_bytes_recv"].items():
            self.stats['net_recv'][k.value] = v.value
    
    def run(self):
        """Run the profiler"""
        self.start_time = time.time()
        self.b["events"].open_perf_buffer(self.event_callback)
        
        try:
            end_time = self.start_time + self.duration
            while time.time() < end_time:
                self.b.perf_buffer_poll(timeout=1000)
        except KeyboardInterrupt:
            print("\n\nProfiling interrupted by user")
        
        # Collect final stats
        self.collect_stats()
        self.print_report()
    
    def print_report(self):
        """Print comprehensive profiling report"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("GPU TRAINING PROFILER REPORT")
        print("="*70)
        print(f"Duration: {elapsed:.2f}s")
        print()
        
        # GPU Activity
        print("🖥️  GPU ACTIVITY")
        print("-" * 70)
        total_ioctls = sum(self.stats['gpu_ioctls'].values())
        if total_ioctls > 0:
            print(f"Total GPU ioctl calls: {total_ioctls:,}")
            print(f"GPU calls/sec: {total_ioctls / elapsed:.1f}")
            
            if self.stats['gpu_sync_latency']:
                avg_sync = sum(self.stats['gpu_sync_latency']) / len(self.stats['gpu_sync_latency'])
                max_sync = max(self.stats['gpu_sync_latency'])
                print(f"GPU sync operations: {len(self.stats['gpu_sync_latency'])}")
                print(f"Avg sync latency: {avg_sync:.2f}ms")
                print(f"Max sync latency: {max_sync:.2f}ms")
                
                if avg_sync > 10:
                    print("⚠️  WARNING: High GPU sync latency detected!")
                    print("   → GPU may be waiting for CPU or network")
        else:
            print("No GPU activity detected")
            print("   → Check if NVIDIA drivers are loaded")
            print("   → Run with a GPU workload (PyTorch, TensorFlow)")
        print()
        
        # Disk I/O
        print("💾 DISK I/O")
        print("-" * 70)
        total_disk = sum(self.stats['disk_bytes'].values())
        if total_disk > 0:
            disk_mb = total_disk / (1024 * 1024)
            disk_throughput = disk_mb / elapsed
            print(f"Total data read: {disk_mb:.2f} MB")
            print(f"Disk throughput: {disk_throughput:.2f} MB/s")
            
            if self.stats['disk_latency']:
                avg_disk = sum(self.stats['disk_latency']) / len(self.stats['disk_latency'])
                max_disk = max(self.stats['disk_latency'])
                print(f"Slow read operations (>10ms): {len(self.stats['disk_latency'])}")
                print(f"Avg read latency: {avg_disk:.2f}ms")
                print(f"Max read latency: {max_disk:.2f}ms")
                
                if avg_disk > 50:
                    print("⚠️  WARNING: High disk latency detected!")
                    print("   → Data loading may be bottleneck")
                    print("   → Consider: SSD, caching, more DataLoader workers")
        else:
            print("No significant disk I/O detected")
        print()
        
        # Network Activity (NCCL)
        print("🌐 NETWORK ACTIVITY (NCCL)")
        print("-" * 70)
        total_sent = sum(self.stats['net_sent'].values())
        total_recv = sum(self.stats['net_recv'].values())
        
        if total_sent > 0 or total_recv > 0:
            sent_mb = total_sent / (1024 * 1024)
            recv_mb = total_recv / (1024 * 1024)
            sent_throughput = sent_mb / elapsed
            recv_throughput = recv_mb / elapsed
            
            print(f"Data sent: {sent_mb:.2f} MB ({sent_throughput:.2f} MB/s)")
            print(f"Data received: {recv_mb:.2f} MB ({recv_throughput:.2f} MB/s)")
            
            if self.stats['net_send_latency']:
                avg_send = sum(self.stats['net_send_latency']) / len(self.stats['net_send_latency'])
                print(f"Large network sends (>1MB): {len(self.stats['net_send_latency'])}")
                print(f"Avg send latency: {avg_send:.2f}ms")
                
                if avg_send > 100:
                    print("⚠️  WARNING: High network latency!")
                    print("   → NCCL all-reduce may be bottleneck")
                    print("   → Check: network bandwidth, NCCL config, TCP vs RDMA")
        else:
            print("No large network transfers detected")
            print("   → Single GPU training or no NCCL activity")
        print()
        
        # Overall Assessment
        print("📊 BOTTLENECK ANALYSIS")
        print("-" * 70)
        
        # Heuristic bottleneck detection
        bottlenecks = []
        
        if total_disk > 0 and self.stats['disk_latency']:
            avg_disk = sum(self.stats['disk_latency']) / len(self.stats['disk_latency'])
            if avg_disk > 50:
                bottlenecks.append(("DISK I/O", avg_disk))
        
        if self.stats['net_send_latency']:
            avg_net = sum(self.stats['net_send_latency']) / len(self.stats['net_send_latency'])
            if avg_net > 50:
                bottlenecks.append(("NETWORK", avg_net))
        
        if self.stats['gpu_sync_latency']:
            avg_sync = sum(self.stats['gpu_sync_latency']) / len(self.stats['gpu_sync_latency'])
            if avg_sync > 10:
                bottlenecks.append(("GPU_SYNC", avg_sync))
        
        if bottlenecks:
            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            print("🚨 Potential bottlenecks detected:")
            for i, (name, latency) in enumerate(bottlenecks, 1):
                print(f"   {i}. {name}: {latency:.2f}ms avg latency")
        else:
            print("✅ No major bottlenecks detected")
            if total_ioctls == 0:
                print("   (Note: No GPU activity seen - run with actual workload)")
        
        print()
        print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description="GPU Training Profiler - eBPF-based ML training profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile all processes for 30 seconds
  sudo python3 gpu_training_profiler.py -d 30
  
  # Profile specific process
  sudo python3 gpu_training_profiler.py -p 12345 -d 60
  
  # Quick 10-second profile
  sudo python3 gpu_training_profiler.py
        """
    )
    parser.add_argument('-p', '--pid', type=int, help='Target process ID (optional)')
    parser.add_argument('-d', '--duration', type=int, default=10, 
                       help='Monitoring duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Check root
    import os
    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges")
        print("Please run with: sudo python3 gpu_training_profiler.py")
        sys.exit(1)
    
    profiler = GPUTrainingProfiler(target_pid=args.pid, duration=args.duration)
    profiler.run()

if __name__ == "__main__":
    main()
