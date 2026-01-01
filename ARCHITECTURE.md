# GPU Training Profiler - Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER SPACE                                │
│                                                                   │
│  ┌──────────────────────┐         ┌─────────────────────────┐  │
│  │  Training Script     │         │  Profiler Script        │  │
│  │  (train.py or       │         │  (gpu_training_profiler)│  │
│  │   demo_training.py) │         │                         │  │
│  │                      │         │  - Loads BPF programs   │  │
│  │  - PyTorch/TF       │         │  - Reads BPF maps       │  │
│  │  - CUDA calls       │         │  - Generates reports    │  │
│  │  - Data loading     │         │                         │  │
│  │  - NCCL comms       │         └────────┬────────────────┘  │
│  └───────┬──────────────┘                  │                   │
│          │                                 │                   │
│          │ Syscalls                        │ Reads statistics  │
│          ↓                                 ↓                   │
└──────────┼─────────────────────────────────┼───────────────────┘
           │                                 │
═══════════╪═════════════════════════════════╪═══════════════════
           │          KERNEL SPACE            │
           │                                 │
┌──────────┼─────────────────────────────────┼───────────────────┐
│          ↓                                 │                   │
│  ┌────────────────────────────────────┐   │                   │
│  │      eBPF Programs (in kernel)      │   │                   │
│  │                                     │   │                   │
│  │  ┌──────────────────────────────┐  │   │                   │
│  │  │ GPU Driver Hook (ioctl)      │  │   │                   │
│  │  │ - Tracks GPU operations      │  │   │                   │
│  │  │ - Measures sync latency      │  │   │                   │
│  │  └──────────────────────────────┘  │   │                   │
│  │                                     │   │                   │
│  │  ┌──────────────────────────────┐  │   │                   │
│  │  │ Disk I/O Hook (read/write)   │  │   │                   │
│  │  │ - Tracks data loading        │  │   │                   │
│  │  │ - Measures latency           │  │   │                   │
│  │  └──────────────────────────────┘  │   │                   │
│  │                                     │   │                   │
│  │  ┌──────────────────────────────┐  │   │                   │
│  │  │ Network Hook (sendmsg/recv)  │  │   │                   │
│  │  │ - Tracks NCCL transfers      │  │   │                   │
│  │  │ - Measures bandwidth         │  │   │                   │
│  │  └──────────────────────────────┘  │   │                   │
│  │                                     │   │                   │
│  │  ┌──────────────────────────────┐  │   │                   │
│  │  │ BPF Maps (statistics)        │◄──┼───┘                   │
│  │  │ - Counters                   │  │                       │
│  │  │ - Histograms                 │  │                       │
│  │  │ - Per-process stats          │  │                       │
│  │  └──────────────────────────────┘  │                       │
│  └─────────────────────────────────────┘                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Kernel Subsystems                           │  │
│  │                                                           │  │
│  │  /dev/nvidia*  │  Filesystem  │  Network Stack           │  │
│  │  (GPU driver)  │   (VFS)      │  (TCP/IP, RDMA)          │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Script Execution
```
Training Script
    │
    ├─→ PyTorch forward()  ──→  CUDA kernels  ──→  ioctl(/dev/nvidia0)
    │                                                      ↓
    │                                              [BPF Hook catches]
    │                                                      ↓
    │                                              [Increment counter]
    │                                                      ↓
    │                                              [Store in BPF map]
    │
    ├─→ DataLoader read()  ──→  VFS read()  ──→  sys_read()
    │                                                      ↓
    │                                              [BPF Hook catches]
    │                                                      ↓
    │                                              [Measure latency]
    │                                                      ↓
    │                                              [Store in BPF map]
    │
    └─→ NCCL all_reduce()  ──→  sendmsg()  ──→  sys_sendmsg()
                                                            ↓
                                                    [BPF Hook catches]
                                                            ↓
                                                    [Track bytes sent]
                                                            ↓
                                                    [Store in BPF map]
```

### 2. Profiler Collection
```
Profiler Script
    │
    ├─→ Poll perf buffer  ──→  Receive events from BPF
    │                              ↓
    │                          [Print real-time events]
    │
    ├─→ Read BPF maps  ──→  Collect aggregated stats
    │                              ↓
    │                          [Counters, histograms]
    │
    └─→ Generate report  ──→  Analyze bottlenecks
                                   ↓
                           [Print recommendations]
```

## eBPF Program Structure

### 1. Hooks (Attach Points)
```c
// Entry hook: Start timing
int trace_ioctl_entry(struct pt_regs *ctx, int fd, unsigned long cmd) {
    u64 ts = bpf_ktime_get_ns();
    // Store start time in map
    start_times.update(&pid_tgid, &ts);
    return 0;
}

// Exit hook: Calculate latency
int trace_ioctl_return(struct pt_regs *ctx) {
    u64 end_ts = bpf_ktime_get_ns();
    // Retrieve start time
    u64 *start_ts = start_times.lookup(&pid_tgid);
    u64 latency = end_ts - *start_ts;
    
    // Update statistics
    gpu_ioctl_count.increment();
    
    // Send event to userspace
    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}
```

### 2. Data Structures

#### BPF Maps (Kernel-side storage)
```c
BPF_HASH(gpu_ioctl_count, u32, u64);      // PID → count
BPF_HASH(disk_bytes, u32, u64);            // PID → bytes read
BPF_HASH(net_bytes_sent, u32, u64);        // PID → bytes sent
BPF_PERF_OUTPUT(events);                   // Event stream
```

#### Events (Sent to userspace)
```c
struct event_t {
    u64 ts;           // Timestamp
    u32 pid;          // Process ID
    u32 type;         // Event type (GPU/disk/network)
    u64 latency_ns;   // Operation latency
    u64 size;         // Bytes transferred
    char comm[16];    // Process name
};
```

## Why This Design?

### ✅ Advantages

1. **Low Overhead**
   - BPF runs in kernel → no context switches
   - Aggregation in BPF maps → minimal data transfer
   - ~1-2% CPU overhead

2. **Safety**
   - BPF verifier ensures no kernel crashes
   - Sandboxed execution
   - Can't hang or deadlock kernel

3. **System-Wide Visibility**
   - Sees all processes (or filtered by PID)
   - Catches syscalls from any library (CUDA, NCCL, etc.)
   - No code modification needed

4. **Real-Time**
   - Events streamed as they happen
   - Immediate feedback on bottlenecks

### ⚠️ Trade-offs

1. **Limited GPU Visibility**
   - Cannot see inside GPU hardware
   - Only sees driver interactions
   - Need nvidia-smi/nsys for GPU internals

2. **Kernel Bypass Limitations**
   - RDMA data plane invisible (control plane visible)
   - NVLink transfers not directly observable
   - Must infer from frequency/patterns

3. **Complexity Limit**
   - BPF verifier restricts program complexity
   - No unbounded loops
   - Limited stack (512 bytes)

## Comparison with Other Approaches

| Approach | Visibility | Overhead | Safety | Flexibility |
|----------|-----------|----------|--------|-------------|
| **eBPF (This tool)** | System-level syscalls | 1-2% | ✅ Kernel-verified | Medium |
| **strace** | Syscalls | 50-100% | ✅ Safe | High |
| **ltrace** | Library calls | 30-50% | ✅ Safe | High |
| **PyTorch Profiler** | Python/CUDA API | 5-10% | ✅ Safe | Medium |
| **nsys/nvprof** | GPU internals | 5-15% | ✅ Safe | High |
| **Kernel module** | Everything | 1-2% | ❌ Can crash kernel | Very High |

## Extension Points

Want to add more monitoring? Easy to extend:

### Monitor CPU Scheduling
```c
TRACEPOINT_PROBE(sched, sched_switch) {
    // Track context switches
    // Detect CPU contention
}
```

### Monitor Memory Allocation
```c
int trace_malloc(struct pt_regs *ctx, size_t size) {
    // Track memory allocations
    // Detect memory pressure
}
```

### Monitor Specific NCCL Operations
```c
// Use uprobe on libnccl.so
uprobe:/path/to/libnccl.so:ncclAllReduce {
    // Track specific NCCL operations
}
```

## Further Reading

- [eBPF Documentation](https://ebpf.io/what-is-ebpf/)
- [BCC Reference Guide](https://github.com/iovisor/bcc/blob/master/docs/reference_guide.md)
- [Linux Tracing Systems](https://www.brendangregg.com/blog/2015-07-08/choosing-a-linux-tracer.html)
- [GPU Profiling Best Practices](https://docs.nvidia.com/nsight-systems/UserGuide/)

---

**Key Insight**: This profiler fills the gap between high-level tools (PyTorch Profiler) and low-level tools (nvidia-smi, nsys) by providing **system-level correlation** of GPU, I/O, and network activity.
