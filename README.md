# GPU Training Profiler

An eBPF-based profiler for ML/AI training workloads that monitors GPU driver activity, disk I/O, network transfers (NCCL), and identifies performance bottlenecks.

## 🎯 What Does It Do?

This profiler helps answer critical questions about GPU training performance:
- **"Why is my GPU only at 60% utilization?"**
- **"Is my training I/O bound, GPU bound, or network bound?"**
- **"How much time is spent waiting for data vs computing?"**
- **"Is NCCL all-reduce a bottleneck in multi-GPU training?"**

Unlike traditional profilers (nvidia-smi, PyTorch Profiler), this tool provides a **system-level view** by monitoring kernel activity, giving you insights into:
- GPU driver interactions (ioctl calls, synchronization overhead)
- Disk I/O patterns (data loading bottlenecks)
- Network activity (NCCL communication for distributed training)
- Correlation between GPU, CPU, disk, and network

## 🚀 Features

✅ **GPU Activity Monitoring**: Track GPU driver calls and synchronization overhead  
✅ **Disk I/O Analysis**: Detect slow data loading and measure throughput  
✅ **Network Profiling**: Monitor NCCL all-reduce operations and bandwidth  
✅ **Bottleneck Detection**: Automatic identification of performance issues  
✅ **Low Overhead**: eBPF runs in kernel space with minimal performance impact  
✅ **Framework Agnostic**: Works with PyTorch, TensorFlow, JAX, or any CUDA application  

## 📋 Requirements

### System Requirements
- **OS**: Linux (kernel 4.14+ recommended)
- **Privileges**: Root access (required for eBPF)
- **Hardware**: NVIDIA GPU (optional - can run without GPU for I/O/network profiling)

### Software Dependencies
- **BCC (BPF Compiler Collection)**
- **Python 3.6+**
- **Linux kernel headers**

### Installation

#### Ubuntu/Debian
```bash
# Install kernel headers
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)

# Install BCC
sudo apt-get install -y bpfcc-tools python3-bpfcc

# Verify installation
python3 -c "from bcc import BPF; print('BCC installed successfully')"
```

#### Fedora/CentOS/RHEL
```bash
# Install kernel headers
sudo dnf install -y kernel-devel-$(uname -r)

# Install BCC
sudo dnf install -y bcc-tools python3-bcc

# Verify
python3 -c "from bcc import BPF; print('BCC installed successfully')"
```

#### From Source (if packages not available)
```bash
# See: https://github.com/iovisor/bcc/blob/master/INSTALL.md
```

## 🎮 Usage

### Basic Usage

1. **Start the profiler** (in one terminal):
```bash
# Profile all processes for 30 seconds
sudo python3 gpu_training_profiler.py -d 30
```

2. **Run your training** (in another terminal):
```bash
# Your actual training script
python3 train.py

# OR use the demo script
python3 demo_training.py
```

### Advanced Usage

#### Profile Specific Process
```bash
# Get PID of your training process
pgrep -f python3

# Profile that specific PID
sudo python3 gpu_training_profiler.py -p 12345 -d 60
```

#### Different Profiling Durations
```bash
# Quick 10-second check
sudo python3 gpu_training_profiler.py

# Long-running profile (5 minutes)
sudo python3 gpu_training_profiler.py -d 300
```

### Demo Mode

Test the profiler with the included demo script:

**Terminal 1** (start profiler):
```bash
sudo python3 gpu_training_profiler.py -d 60
```

**Terminal 2** (run demo):
```bash
# Full training simulation
python3 demo_training.py --epochs 3 --batches 10

# Test specific components
python3 demo_training.py --mode io       # Disk I/O only
python3 demo_training.py --mode gpu      # GPU compute only
python3 demo_training.py --mode network  # Network only
```

## 📊 Understanding the Output

### Sample Report

```
======================================================================
GPU TRAINING PROFILER REPORT
======================================================================
Duration: 30.45s

🖥️  GPU ACTIVITY
----------------------------------------------------------------------
Total GPU ioctl calls: 12,450
GPU calls/sec: 408.9
GPU sync operations: 45
Avg sync latency: 8.32ms
Max sync latency: 25.10ms

💾 DISK I/O
----------------------------------------------------------------------
Total data read: 1,234.56 MB
Disk throughput: 40.52 MB/s
Slow read operations (>10ms): 234
Avg read latency: 15.23ms
Max read latency: 89.45ms
⚠️  WARNING: High disk latency detected!
   → Data loading may be bottleneck
   → Consider: SSD, caching, more DataLoader workers

🌐 NETWORK ACTIVITY (NCCL)
----------------------------------------------------------------------
Data sent: 512.34 MB (16.82 MB/s)
Data received: 512.28 MB (16.82 MB/s)
Large network sends (>1MB): 128
Avg send latency: 45.67ms

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
🚨 Potential bottlenecks detected:
   1. NETWORK: 45.67ms avg latency
   2. DISK_IO: 15.23ms avg latency
   3. GPU_SYNC: 8.32ms avg latency

======================================================================
```

### Interpreting Results

#### GPU Activity
- **High ioctl calls/sec** (>100): Good GPU utilization
- **Low ioctl calls/sec** (<50): GPU underutilized - check other bottlenecks
- **High sync latency** (>10ms): GPU waiting for CPU, network, or synchronization

#### Disk I/O
- **Good throughput**: >500 MB/s for NVMe SSD, >100 MB/s for SATA SSD
- **Warning signs**: Latency >50ms indicates slow storage or contention
- **Solutions**:
  - Increase `num_workers` in PyTorch DataLoader
  - Use faster storage (NVMe SSD)
  - Cache dataset in RAM (`/dev/shm`)
  - Enable `pin_memory=True` in DataLoader

#### Network (NCCL)
- **Expected bandwidth**: 
  - InfiniBand: 10-12 GB/s per link
  - 100GbE: 10-12 GB/s
  - 10GbE: 1-1.2 GB/s
- **Warning signs**: Latency >100ms or bandwidth <50% of capacity
- **Solutions**:
  - Check NCCL is using RDMA (not TCP fallback)
  - Verify `NCCL_IB_DISABLE` is not set
  - Check network topology with `nvidia-smi topo -m`
  - Tune NCCL settings: `NCCL_DEBUG=INFO`

#### Bottleneck Priority
The profiler ranks bottlenecks by severity. Focus on the top issue first:
1. **DISK_IO**: Increase DataLoader workers, use faster storage
2. **NETWORK**: Check NCCL config, verify RDMA is enabled
3. **GPU_SYNC**: Reduce unnecessary synchronization in code

## 🔬 How It Works

### eBPF Hooks

The profiler uses eBPF (extended Berkeley Packet Filter) to monitor kernel-level activity:

1. **GPU Driver (`ioctl`)**:
   - Hooks `sys_ioctl` calls to `/dev/nvidia*`
   - Tracks kernel launches, memory transfers, synchronization
   - Measures latency of GPU operations

2. **Disk I/O (`read`)**:
   - Hooks `sys_read` and `sys_pread64`
   - Measures read latency and throughput
   - Identifies slow file operations

3. **Network (`sendmsg`/`recvmsg`)**:
   - Hooks network syscalls
   - Detects large transfers (likely NCCL all-reduce)
   - Measures network bandwidth and latency

4. **Aggregation**:
   - BPF maps collect statistics (counters, histograms)
   - Userspace Python reads maps and generates report
   - Minimal overhead (~1-2% CPU)

### Why eBPF?

- **Safe**: Verified by kernel (won't crash your system)
- **Fast**: Runs in kernel space, minimal overhead
- **Dynamic**: Load/unload without rebooting
- **System-wide**: See activity from all processes
- **No code changes**: Profile without modifying your training script

## 🎓 Use Cases

### Use Case 1: Diagnosing Low GPU Utilization
**Symptom**: `nvidia-smi` shows GPU at 40%  
**Profiler Output**: High disk I/O latency (80ms avg)  
**Diagnosis**: Data loading is the bottleneck  
**Fix**: Increase `num_workers` from 4 to 8, GPU util → 95%

### Use Case 2: Slow Multi-GPU Training
**Symptom**: 8 GPUs slower than expected  
**Profiler Output**: Network latency 200ms, bandwidth 2 GB/s (should be 12 GB/s)  
**Diagnosis**: NCCL using TCP instead of RDMA  
**Fix**: Unset `NCCL_IB_DISABLE`, training speedup 3x

### Use Case 3: Training Crashes (OOM)
**Symptom**: CUDA out of memory  
**Profiler Output**: High frequency of ioctl calls (memory allocation patterns)  
**Diagnosis**: Memory thrashing  
**Fix**: Reduce batch size or enable gradient checkpointing

## ⚠️ Limitations

### What eBPF Can See
✅ GPU driver interactions (ioctl calls)  
✅ Disk I/O (reads, writes, latency)  
✅ Network activity (TCP/IP transfers)  
✅ Process-level activity  

### What eBPF Cannot See
❌ GPU internals (CUDA kernel execution details)  
❌ Tensor core utilization  
❌ RDMA data plane (kernel bypass)  
❌ NVLink transfers (hardware-level)  

For these, use complementary tools:
- **nvidia-smi**: GPU utilization, memory, power
- **nsys/nvprof**: CUDA kernel profiling
- **PyTorch Profiler**: Python-level timing

### Best Used With
This profiler is best used **alongside** other tools:
- eBPF Profiler → System bottlenecks (I/O, network, driver)
- nvidia-smi → GPU hardware stats
- PyTorch Profiler → Code-level hotspots

## 🐛 Troubleshooting

### "ERROR: This script requires root privileges"
**Solution**: Run with `sudo`

### "Failed to load BPF program"
**Possible causes**:
- Kernel too old (need 4.14+)
- Missing kernel headers: `sudo apt-get install linux-headers-$(uname -r)`
- BCC not installed correctly

### "No GPU activity detected"
**Possible causes**:
- No GPU workload running (normal if testing without CUDA)
- NVIDIA drivers not loaded: `nvidia-smi`
- Profiler PID filter doesn't match training process

### "Permission denied: /dev/nvidia0"
**Solution**: Ensure user has access to GPU device or run as root

## 🔧 Advanced Configuration

### Customizing Thresholds

Edit `gpu_training_profiler.py` to adjust detection thresholds:

```python
# Line ~140: GPU sync threshold (default: 1ms)
if latency > 1000000:  # Change to 5000000 for 5ms

# Line ~183: Disk latency threshold (default: 10ms)
if latency > 10000000:  # Change threshold

# Line ~225: Network transfer size threshold (default: 1MB)
if bytes_sent > 1024 * 1024:  # Change to detect smaller/larger transfers
```

### Adding Custom Hooks

You can extend the BPF program to hook additional syscalls:

```python
# Add to bpf_program variable
"""
int my_custom_hook(struct pt_regs *ctx) {
    // Your custom logic
    return 0;
}
"""

# Attach in __init__
self.b.attach_kprobe(event="my_function", fn_name="my_custom_hook")
```

## 📚 Further Reading

- **eBPF Introduction**: https://ebpf.io/
- **BCC Tutorial**: https://github.com/iovisor/bcc/blob/master/docs/tutorial.md
- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/
- **PyTorch Distributed**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

## 🤝 Contributing

Suggestions for improvements:
- [ ] Add support for AMD GPUs (ROCm driver)
- [ ] Detect specific NCCL algorithms (Ring, Tree, etc.)
- [ ] Integration with prometheus/grafana for real-time dashboard
- [ ] Auto-tuning recommendations based on detected patterns
- [ ] Support for other ML frameworks (TensorFlow, JAX)

## 📄 License

MIT License - feel free to use and modify for your needs.

## 🙏 Acknowledgments

Built with:
- [BCC (BPF Compiler Collection)](https://github.com/iovisor/bcc)
- eBPF technology from the Linux kernel
- Inspired by the need for better ML training observability

---

**Questions or Issues?** This is a research/demo tool. For production use, consider enterprise observability solutions like:
- Datadog APM
- New Relic
- Grafana + Prometheus
- Weights & Biases (for ML-specific metrics)
