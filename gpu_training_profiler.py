#!/usr/bin/env python3
"""
GPU Training Profiler - eBPF-based profiler for ML training workloads

This profiler uses eBPF (extended Berkeley Packet Filter) to monitor system-level
operations during ML training, including:
- Disk I/O (data loading from storage)
- Network activity (distributed training communication)

The profiler is kernel-version compatible, using stable tracepoint APIs instead
of kprobes, making it work across Ubuntu 20.04+, RHEL 8+, and other modern distros.

Architecture:
- BPF programs run in the kernel, collecting data with minimal overhead (<1% CPU)
- Userspace Python program loads BPF code, reads events, and generates reports
- Uses tracepoints (stable syscall hooks) for maximum compatibility
"""

from bcc import BPF
import time
import ctypes as ct
from collections import defaultdict
import argparse
import sys

# ============================================================================
# BPF PROGRAM (runs in kernel space)
# ============================================================================
# This C code is compiled by BCC and injected into the Linux kernel.
# It hooks into syscall tracepoints to monitor I/O operations.

bpf_program = r"""
#include <uapi/linux/ptrace.h>  // Only header needed - minimal dependencies

// Event structure - sent from kernel to userspace when interesting events occur
struct event_t {
    u64 ts;           // Timestamp in nanoseconds (from bpf_ktime_get_ns)
    u32 pid;          // Process ID that triggered the event
    u32 type;         // Event type: 1=disk read, 2=network send
    u64 duration_ns;  // How long the operation took (in nanoseconds)
    u64 size;         // Size of data transferred (in bytes)
};

// BPF Maps - data structures shared between kernel and userspace
// These persist across multiple syscalls and allow aggregation
BPF_PERF_OUTPUT(events);                    // Ring buffer for sending events to userspace
BPF_HASH(start_times, u64, u64);            // Track syscall entry time per thread
BPF_HASH(read_bytes, u32, u64);             // Total bytes read per process
BPF_HASH(send_bytes, u32, u64);             // Total bytes sent per process

// ============================================================================
// DISK I/O MONITORING - Hooks the read() syscall
// ============================================================================
// When a process calls read() to load data from disk, we measure latency
// and track total bytes read. This catches ML data loading bottlenecks.

// Hook: read() syscall entry point
// Triggered BEFORE the kernel performs the read operation
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    u64 pid_tgid = bpf_get_current_pid_tgid();  // Get process/thread ID (combined)
    u64 ts = bpf_ktime_get_ns();                 // Current time in nanoseconds
    start_times.update(&pid_tgid, &ts);          // Store start time for latency calc
    return 0;
}

// Hook: read() syscall exit point
// Triggered AFTER the kernel completes the read operation
// We calculate latency and accumulate total bytes read
TRACEPOINT_PROBE(syscalls, sys_exit_read) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);  // Get the start time we saved
    
    // If no start time found, this read wasn't tracked (race condition)
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();           // Current time
    u64 duration = end_ts - *start_ts;         // Calculate latency
    long bytes_read = args->ret;               // Read syscall return value = bytes read
    
    if (bytes_read > 0) {
        u32 pid = pid_tgid >> 32;              // Extract PID from combined pid_tgid
        
        // Accumulate total bytes read for this process
        u64 *total = read_bytes.lookup(&pid);
        u64 new_total = total ? *total + bytes_read : bytes_read;
        read_bytes.update(&pid, &new_total);
        
        // Report slow reads (>10ms and >1KB) to userspace for analysis
        // These indicate I/O bottlenecks in data loading
        if (duration > 10000000 && bytes_read > 1024) {  // 10ms = 10,000,000 ns
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.type = 1;                    // Type 1 = disk read event
            event.duration_ns = duration;
            event.size = bytes_read;
            events.perf_submit(args, &event, sizeof(event));  // Send to userspace
        }
    }
    
    start_times.delete(&pid_tgid);  // Clean up - prevent memory leak
    return 0;
}

// ============================================================================
// NETWORK MONITORING - Hooks the sendmsg() syscall
// ============================================================================
// Tracks network data transfer during distributed training (NCCL, MPI, etc.)
// sendmsg() is used for TCP/UDP communication between training nodes

// Hook: sendmsg() syscall entry
TRACEPOINT_PROBE(syscalls, sys_enter_sendmsg) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_times.update(&pid_tgid, &ts);  // Save timestamp for latency measurement
    return 0;
}

// Hook: sendmsg() syscall exit
// Triggered after network send completes
TRACEPOINT_PROBE(syscalls, sys_exit_sendmsg) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = start_times.lookup(&pid_tgid);
    
    if (start_ts == 0) {
        return 0;
    }
    
    u64 end_ts = bpf_ktime_get_ns();
    u64 duration = end_ts - *start_ts;
    long bytes_sent = args->ret;  // sendmsg return value = bytes sent
    
    if (bytes_sent > 0) {
        u32 pid = pid_tgid >> 32;
        
        // Accumulate total bytes sent for this process
        u64 *total = send_bytes.lookup(&pid);
        u64 new_total = total ? *total + bytes_sent : bytes_sent;
        send_bytes.update(&pid, &new_total);
        
        // Report large transfers (>1MB) - these indicate NCCL all-reduce,
        // gradient synchronization, or other distributed training communication
        if (bytes_sent > 1048576) {  // 1MB = 1,048,576 bytes
            struct event_t event = {};
            event.ts = end_ts;
            event.pid = pid;
            event.type = 2;                    // Type 2 = network send event
            event.duration_ns = duration;
            event.size = bytes_sent;
            events.perf_submit(args, &event, sizeof(event));
        }
    }
    
    start_times.delete(&pid_tgid);
    return 0;
}
"""

# ============================================================================
# PYTHON USERSPACE CODE
# ============================================================================

# Python structure matching the kernel's event_t struct
# ctypes allows us to parse binary data from the kernel
class Event(ct.Structure):
    _fields_ = [
        ("ts", ct.c_ulonglong),          # Timestamp
        ("pid", ct.c_uint),              # Process ID
        ("type", ct.c_uint),             # Event type (1=disk, 2=network)
        ("duration_ns", ct.c_ulonglong), # Operation latency
        ("size", ct.c_ulonglong),        # Bytes transferred
    ]

class SimpleProfiler:
    """Main profiler class that manages BPF program lifecycle and data collection."""
    
    def __init__(self, target_pid=None, duration=10):
        """Initialize the profiler with optional PID filter and duration."""
        self.target_pid = target_pid
        self.duration = duration
        self.start_time = None
        
        # Statistics collected during monitoring
        self.stats = {
            'read_bytes': defaultdict(int),   # Total bytes read per PID
            'send_bytes': defaultdict(int),   # Total bytes sent per PID
            'slow_reads': [],                  # List of (pid, latency_ms, size_mb)
            'large_sends': [],                 # List of (pid, latency_ms, size_mb)
        }
        
        print("Loading eBPF program...")
        try:
            # BCC compiles the C code, loads it into kernel, attaches hooks
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
        """Process events from kernel perf buffer for slow reads and large network transfers."""
        # Parse binary data into Python Event object
        event = ct.cast(data, ct.POINTER(Event)).contents
        
        # Filter by PID if specified
        if self.target_pid and event.pid != self.target_pid:
            return
        
        # Convert from nanoseconds/bytes to human-readable units
        duration_ms = event.duration_ns / 1_000_000
        size_mb = event.size / (1024 * 1024)
        
        if event.type == 1:  # Disk read event
            self.stats['slow_reads'].append((event.pid, duration_ms, size_mb))
            print(f"[DISK] PID {event.pid} - {size_mb:.2f} MB in {duration_ms:.2f}ms")
        elif event.type == 2:  # Network send event
            self.stats['large_sends'].append((event.pid, duration_ms, size_mb))
            throughput_mbps = size_mb / (duration_ms / 1000) if duration_ms > 0 else 0
            print(f"[NETWORK] PID {event.pid} - {size_mb:.2f} MB in {duration_ms:.2f}ms ({throughput_mbps:.2f} MB/s)")
    
    def collect_stats(self):
        """Read aggregated statistics from BPF hash maps in the kernel."""
        # Read total bytes read per process from kernel hash map
        for k, v in self.b["read_bytes"].items():
            self.stats['read_bytes'][k.value] = v.value
        
        # Read total bytes sent per process from kernel hash map
        for k, v in self.b["send_bytes"].items():
            self.stats['send_bytes'][k.value] = v.value
    
    def run(self):
        """Main monitoring loop that polls for events and generates the final report."""
        self.start_time = time.time()
        
        # Register callback to handle events from kernel
        self.b["events"].open_perf_buffer(self.event_callback)
        
        try:
            end_time = self.start_time + self.duration
            while time.time() < end_time:
                # Poll kernel for events (1 second timeout)
                # This receives events sent via perf_submit() in BPF code
                self.b.perf_buffer_poll(timeout=1000)
        except KeyboardInterrupt:
            print("\n\nProfiling interrupted")
        
        # Read final statistics from BPF maps
        self.collect_stats()
        
        # Generate and display report
        self.print_report()
    
    def print_report(self):
        """Generate and print human-readable profiling report with disk and network statistics."""
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
            print(f"Slow reads (over 10ms): {len(self.stats['slow_reads'])}")
            
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
            print(f"Large transfers (over 1MB): {len(self.stats['large_sends'])}")
            
            if self.stats['large_sends']:
                durations = [d for _, d, _ in self.stats['large_sends']]
                avg_duration = sum(durations) / len(durations)
                print(f"Avg transfer latency: {avg_duration:.2f}ms")
        else:
            print("No large network transfers detected")
        print()
        
        print("="*70)

def main():
    """Command-line entry point for the profiler."""
    parser = argparse.ArgumentParser(
        description="eBPF-based profiler for ML training workloads",
        epilog="Example: sudo python3 gpu_training_profiler.py -d 60"
    )
    parser.add_argument('-p', '--pid', type=int, 
                       help='Target process ID to monitor (optional, default: all processes)')
    parser.add_argument('-d', '--duration', type=int, default=10, 
                       help='Monitoring duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # eBPF requires root privileges to load programs into kernel
    import os
    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges to load eBPF programs")
        print("Please run with: sudo python3 gpu_training_profiler.py")
        sys.exit(1)
    
    # Create profiler instance and start monitoring
    profiler = SimpleProfiler(target_pid=args.pid, duration=args.duration)
    profiler.run()

if __name__ == "__main__":
    main()
