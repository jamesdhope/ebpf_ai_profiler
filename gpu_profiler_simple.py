#!/usr/bin/env python3
"""
Simplified GPU Training Profiler - Compatible with more kernel versions
"""

from bcc import BPF
import time
import ctypes as ct
from collections import defaultdict
import argparse
import sys

# Simplified BPF program without complex includes
bpf_program = r"""
#include <uapi/linux/ptrace.h>

struct event_t {
    u64 ts;
    u32 pid;
    u32 type;
    u64 duration_ns;
    u64 size;
};

BPF_PERF_OUTPUT(events);
BPF_HASH(start_times, u64, u64);
BPF_HASH(read_bytes, u32, u64);
BPF_HASH(send_bytes, u32, u64);

// Trace read syscall
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_times.update(&pid_tgid, &ts);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_read) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 duration = end_ts - *start_ts;
    long bytes_read = args->ret;
    
    if (bytes_read > 0) {
        u32 pid = pid_tgid >> 32;
        u64 *total = read_bytes.lookup(&pid);
        u64 new_total = total ? *total + bytes_read : bytes_read;
        read_bytes.update(&pid, &new_total);
        
        // Report slow reads
        if (duration > 10000000 && bytes_read > 1024) {
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.type = 1;
            event.duration_ns = duration;
            event.size = bytes_read;
            events.perf_submit(args, &event, sizeof(event));
        }
    }
    
    start_times.delete(&pid_tgid);
    return 0;
}

// Trace sendmsg (network)
TRACEPOINT_PROBE(syscalls, sys_enter_sendmsg) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_times.update(&pid_tgid, &ts);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_sendmsg) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 duration = end_ts - *start_ts;
    long bytes_sent = args->ret;
    
    if (bytes_sent > 0) {
        u32 pid = pid_tgid >> 32;
        u64 *total = send_bytes.lookup(&pid);
        u64 new_total = total ? *total + bytes_sent : bytes_sent;
        send_bytes.update(&pid, &new_total);
        
        // Report large sends
        if (bytes_sent > 1048576) {  // > 1MB
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.type = 2;
            event.duration_ns = duration;
            event.size = bytes_sent;
            events.perf_submit(args, &event, sizeof(event));
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
        ("type", ct.c_uint),
        ("duration_ns", ct.c_ulonglong),
        ("size", ct.c_ulonglong),
    ]

class SimpleProfiler:
    def __init__(self, target_pid=None, duration=10):
        self.target_pid = target_pid
        self.duration = duration
        self.start_time = None
        self.stats = {
            'read_bytes': defaultdict(int),
            'send_bytes': defaultdict(int),
            'slow_reads': [],
            'large_sends': [],
        }
        
        print("Loading eBPF program...")
        try:
            self.b = BPF(text=bpf_program)
            print("✅ eBPF program loaded successfully")
            print(f"Monitoring for {duration} seconds...")
            if target_pid:
                print(f"Filtering for PID: {target_pid}")
            print()
        except Exception as e:
            print(f"❌ Failed to load eBPF program: {e}")
            sys.exit(1)
    
    def event_callback(self, cpu, data, size):
        event = ct.cast(data, ct.POINTER(Event)).contents
        
        if self.target_pid and event.pid != self.target_pid:
            return
        
        duration_ms = event.duration_ns / 1_000_000
        size_mb = event.size / (1024 * 1024)
        
        if event.type == 1:  # Disk read
            self.stats['slow_reads'].append((event.pid, duration_ms, size_mb))
            print(f"[DISK] PID {event.pid} - {size_mb:.2f} MB in {duration_ms:.2f}ms")
        elif event.type == 2:  # Network send
            self.stats['large_sends'].append((event.pid, duration_ms, size_mb))
            throughput = size_mb / (duration_ms / 1000) if duration_ms > 0 else 0
            print(f"[NETWORK] PID {event.pid} - {size_mb:.2f} MB in {duration_ms:.2f}ms ({throughput:.2f} MB/s)")
    
    def collect_stats(self):
        for k, v in self.b["read_bytes"].items():
            self.stats['read_bytes'][k.value] = v.value
        
        for k, v in self.b["send_bytes"].items():
            self.stats['send_bytes'][k.value] = v.value
    
    def run(self):
        self.start_time = time.time()
        self.b["events"].open_perf_buffer(self.event_callback)
        
        try:
            end_time = self.start_time + self.duration
            while time.time() < end_time:
                self.b.perf_buffer_poll(timeout=1000)
        except KeyboardInterrupt:
            print("\n\nProfiling interrupted")
        
        self.collect_stats()
        self.print_report()
    
    def print_report(self):
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("PROFILER REPORT")
        print("="*70)
        print(f"Duration: {elapsed:.2f}s\n")
        
        # Disk I/O
        print("💾 DISK I/O")
        print("-" * 70)
        total_read = sum(self.stats['read_bytes'].values())
        if total_read > 0:
            read_mb = total_read / (1024 * 1024)
            throughput = read_mb / elapsed
            print(f"Total data read: {read_mb:.2f} MB")
            print(f"Throughput: {throughput:.2f} MB/s")
            print(f"Slow reads (>10ms): {len(self.stats['slow_reads'])}")
            
            if self.stats['slow_reads']:
                durations = [d for _, d, _ in self.stats['slow_reads']]
                avg_duration = sum(durations) / len(durations)
                print(f"Avg slow read latency: {avg_duration:.2f}ms")
        else:
            print("No significant disk I/O detected")
        print()
        
        # Network
        print("🌐 NETWORK ACTIVITY")
        print("-" * 70)
        total_sent = sum(self.stats['send_bytes'].values())
        if total_sent > 0:
            sent_mb = total_sent / (1024 * 1024)
            throughput = sent_mb / elapsed
            print(f"Data sent: {sent_mb:.2f} MB ({throughput:.2f} MB/s)")
            print(f"Large transfers (>1MB): {len(self.stats['large_sends'])}")
            
            if self.stats['large_sends']:
                durations = [d for _, d, _ in self.stats['large_sends']]
                avg_duration = sum(durations) / len(durations)
                print(f"Avg transfer latency: {avg_duration:.2f}ms")
        else:
            print("No large network transfers detected")
        print()
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Simplified Training Profiler")
    parser.add_argument('-p', '--pid', type=int, help='Target process ID')
    parser.add_argument('-d', '--duration', type=int, default=10, help='Duration in seconds')
    
    args = parser.parse_args()
    
    import os
    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges")
        print("Please run with: sudo python3 gpu_profiler_simple.py")
        sys.exit(1)
    
    profiler = SimpleProfiler(target_pid=args.pid, duration=args.duration)
    profiler.run()

if __name__ == "__main__":
    main()
