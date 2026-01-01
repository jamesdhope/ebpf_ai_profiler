# Example: What the Profiler Can Detect

## Scenario 1: I/O Bottleneck

### Before Optimization
```bash
$ sudo python3 gpu_training_profiler.py -p 12345 -d 60
```

**Output:**
```
💾 DISK I/O
----------------------------------------------------------------------
Total data read: 2,345.67 MB
Disk throughput: 39.09 MB/s
Slow read operations (>10ms): 1,234
Avg read latency: 78.45ms
Max read latency: 234.12ms
⚠️  WARNING: High disk latency detected!
   → Data loading may be bottleneck

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
🚨 Potential bottlenecks detected:
   1. DISK_IO: 78.45ms avg latency  ← PRIMARY BOTTLENECK
   2. GPU_SYNC: 5.23ms avg latency
```

**Training Code (Before):**
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,        # ← TOO FEW!
    pin_memory=False,
)
```

### After Optimization
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,        # ← INCREASED!
    pin_memory=True,      # ← ENABLED!
    prefetch_factor=2,
)
```

**Result:**
```
💾 DISK I/O
----------------------------------------------------------------------
Total data read: 2,345.67 MB
Disk throughput: 156.38 MB/s  ← 4x improvement!
Slow read operations (>10ms): 23
Avg read latency: 12.34ms      ← Much better!

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
✅ No major bottlenecks detected
```

**Training Speed:** 3.2x faster!

---

## Scenario 2: NCCL Network Bottleneck

### Before Optimization
```
🌐 NETWORK ACTIVITY (NCCL)
----------------------------------------------------------------------
Data sent: 1,024.00 MB (17.07 MB/s)
Data received: 1,024.00 MB (17.07 MB/s)
Large network sends (>1MB): 256
Avg send latency: 185.67ms
⚠️  WARNING: High network latency!
   → NCCL all-reduce may be bottleneck
   → Check: network bandwidth, NCCL config, TCP vs RDMA

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
🚨 Potential bottlenecks detected:
   1. NETWORK: 185.67ms avg latency  ← PRIMARY BOTTLENECK
```

**Investigation:**
```bash
# Check NCCL is using TCP fallback instead of RDMA
$ echo $NCCL_IB_DISABLE
1  ← Oops! InfiniBand disabled!

# Check network capability
$ ibstat
CA 'mlx5_0'
    Port 1:
        State: Active
        Link speed: 100 Gb/sec  ← Network is fine!
```

**Root Cause:** NCCL configured to use TCP instead of RDMA!

### After Optimization
```bash
# Fix NCCL configuration
unset NCCL_IB_DISABLE
export NCCL_NET=IB
export NCCL_DEBUG=INFO
```

**Result:**
```
🌐 NETWORK ACTIVITY (NCCL)
----------------------------------------------------------------------
Data sent: 1,024.00 MB (512.00 MB/s)  ← 30x improvement!
Data received: 1,024.00 MB (512.00 MB/s)
Large network sends (>1MB): 256
Avg send latency: 6.23ms               ← Much faster!

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
✅ No major bottlenecks detected
```

**Training Speed:** 5x faster on 8 GPUs!

---

## Scenario 3: GPU Underutilization

### Observation
```bash
$ nvidia-smi
GPU Utilization: 45%  ← Low!

$ sudo python3 gpu_training_profiler.py -p 12345 -d 60
```

**Output:**
```
🖥️  GPU ACTIVITY
----------------------------------------------------------------------
Total GPU ioctl calls: 1,234
GPU calls/sec: 20.6  ← Very low!

💾 DISK I/O
----------------------------------------------------------------------
Disk throughput: 35.12 MB/s
Avg read latency: 145.23ms  ← Slow!

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
🚨 Potential bottlenecks detected:
   1. DISK_IO: 145.23ms avg latency
```

**Diagnosis:** GPU is starving for data!

**Fixes Applied:**
1. Increased DataLoader workers: 4 → 12
2. Moved dataset to NVMe SSD
3. Enabled `pin_memory=True`
4. Cached frequently accessed data in `/dev/shm`

**Result:**
```
🖥️  GPU ACTIVITY
----------------------------------------------------------------------
Total GPU ioctl calls: 14,567
GPU calls/sec: 242.8  ← Much better!

💾 DISK I/O
----------------------------------------------------------------------
Disk throughput: 487.34 MB/s  ← 14x improvement!
Avg read latency: 8.12ms

📊 BOTTLENECK ANALYSIS
----------------------------------------------------------------------
✅ No major bottlenecks detected
```

**GPU Utilization:** 45% → 96%  
**Training Speed:** 2.8x faster!

---

## Scenario 4: Straggler Detection (Multi-Node)

### What the Profiler Shows
```bash
# Run on each node
node1$ sudo python3 gpu_training_profiler.py -d 60 | tee node1.log
node2$ sudo python3 gpu_training_profiler.py -d 60 | tee node2.log

# Compare results
$ diff node1.log node2.log
```

**Node 1 (Fast):**
```
🌐 NETWORK ACTIVITY (NCCL)
----------------------------------------------------------------------
Avg send latency: 12.34ms
```

**Node 2 (Slow):**
```
🌐 NETWORK ACTIVITY (NCCL)
----------------------------------------------------------------------
Avg send latency: 234.56ms  ← Much slower!
```

**Investigation:** Node 2 has network issues (bad cable, switch problem)

**Fix:** Replace network cable

**Result:** All nodes now have ~12ms latency, training balanced!

---

## Real-World Impact Summary

| Scenario | Issue | Fix | Speedup |
|----------|-------|-----|---------|
| I/O Bottleneck | Low disk throughput | More workers, pin_memory | 3.2x |
| NCCL TCP Fallback | RDMA disabled | Enable InfiniBand | 5.0x |
| GPU Underutilized | Data starvation | Faster storage, caching | 2.8x |
| Network Straggler | Faulty cable | Hardware replacement | 1.8x |

## Key Takeaways

1. **The profiler shows what nvidia-smi cannot**: system-level bottlenecks
2. **Most "GPU problems" are actually I/O or network problems**
3. **Small config changes can have massive impact** (e.g., NCCL_IB_DISABLE)
4. **Always profile before and after** optimization to measure impact

## Try It Yourself!

```bash
# Run the demo to see these patterns
python3 demo_training.py --mode io       # See I/O bottleneck
python3 demo_training.py --mode network  # See network activity
python3 demo_training.py --mode full     # See everything

# Profile with:
sudo python3 gpu_training_profiler.py -d 30
```
