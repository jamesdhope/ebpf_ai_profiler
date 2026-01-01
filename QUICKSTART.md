# Quick Start Guide

Get started profiling GPU training in 5 minutes!

## 1️⃣ Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) bpfcc-tools python3-bpfcc

# Fedora/CentOS
sudo dnf install -y kernel-devel-$(uname -r) bcc-tools python3-bcc
```

## 2️⃣ Make Scripts Executable

```bash
chmod +x gpu_training_profiler.py
chmod +x demo_training.py
```

## 3️⃣ Test with Demo (Two Terminals)

**Terminal 1** - Start Profiler:
```bash
sudo ./gpu_training_profiler.py -d 60
```

**Terminal 2** - Run Demo:
```bash
./demo_training.py --epochs 2 --batches 5
```

## 4️⃣ Profile Your Own Training

```bash
# Option A: Profile by PID (recommended)
# Terminal 1: Get your training PID
python3 train.py &
echo $!  # Remember this PID

# Terminal 2: Profile it
sudo ./gpu_training_profiler.py -p <PID> -d 300

# Option B: Profile everything (noisy)
sudo ./gpu_training_profiler.py -d 60
```

## 5️⃣ Interpret Results

Look for these warnings in the output:

🚨 **"High disk latency detected"**
→ Fix: Increase DataLoader `num_workers`, use SSD, cache data

🚨 **"High network latency"**  
→ Fix: Check NCCL config (`NCCL_DEBUG=INFO`), verify RDMA enabled

🚨 **"High GPU sync latency"**  
→ Fix: Reduce synchronization points, check for blocking operations

## 💡 Tips

- Run profiler for at least 30-60 seconds to capture full training loop
- Filter by PID to reduce noise from other processes
- Compare results before/after optimization changes
- Use demo script to verify profiler is working correctly

## 🆘 Common Issues

**"This script requires root privileges"**
```bash
sudo python3 gpu_training_profiler.py  # Don't forget sudo!
```

**"No GPU activity detected"**
- Normal if no CUDA workload is running
- Check: `nvidia-smi` (drivers loaded?)
- Try demo script to verify profiler works

**"Failed to load BPF program"**
```bash
# Install kernel headers
sudo apt-get install linux-headers-$(uname -r)
```

## 📖 Full Documentation

See [README.md](README.md) for complete documentation.

## 🎯 Next Steps

1. Baseline your training: `sudo ./gpu_training_profiler.py -p <PID> -d 300`
2. Note the bottlenecks reported
3. Apply recommended fixes
4. Re-profile and compare improvements!

Happy profiling! 🚀
