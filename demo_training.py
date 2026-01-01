#!/usr/bin/env python3
"""
Demo Training Script - Simulates GPU training with I/O and network activity
Use this to test the GPU Training Profiler
"""

import time
import os
import sys
import random
import socket
import struct

def simulate_disk_io(size_mb=100, chunk_size_mb=10):
    """Simulate data loading from disk"""
    print(f"📂 Simulating disk I/O: reading {size_mb}MB...")
    
    # Create temporary file
    temp_file = "/tmp/training_data.bin"
    
    # Write data
    chunk_size = chunk_size_mb * 1024 * 1024
    with open(temp_file, "wb") as f:
        for i in range(size_mb // chunk_size_mb):
            data = os.urandom(chunk_size)
            f.write(data)
    
    # Read data (simulating DataLoader)
    total_read = 0
    with open(temp_file, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_read += len(chunk)
            # Simulate processing delay
            time.sleep(0.05)
    
    # Cleanup
    os.remove(temp_file)
    print(f"   ✓ Read {total_read / (1024*1024):.2f} MB from disk")

def simulate_gpu_compute(iterations=10, compute_time_ms=100):
    """Simulate GPU computation with ioctl calls"""
    print(f"🖥️  Simulating GPU compute: {iterations} iterations...")
    
    try:
        # Try to open NVIDIA device (if available)
        has_gpu = os.path.exists("/dev/nvidia0")
        if has_gpu:
            # This will trigger ioctl calls
            fd = os.open("/dev/nvidia0", os.O_RDWR)
            print("   ✓ Using real GPU device")
        else:
            print("   ⚠️  No GPU detected - using CPU simulation")
            fd = None
    except:
        print("   ⚠️  Cannot access GPU - using CPU simulation")
        fd = None
    
    for i in range(iterations):
        # Simulate computation
        if fd:
            try:
                # Trigger ioctl (will fail but generates syscall)
                os.ioctl(fd, 0x12345678, 0)
            except:
                pass
        
        # CPU work to simulate processing
        _ = sum(range(1000000))
        
        # Simulate synchronization
        time.sleep(compute_time_ms / 1000.0)
        
        if (i + 1) % 5 == 0:
            print(f"   Progress: {i+1}/{iterations} iterations")
    
    if fd:
        os.close(fd)
    
    print(f"   ✓ Completed {iterations} GPU iterations")

def simulate_network_transfer(size_mb=50, num_transfers=5):
    """Simulate NCCL all-reduce (network transfer)"""
    print(f"🌐 Simulating network transfer (NCCL): {num_transfers} transfers of {size_mb}MB...")
    
    try:
        # Create a local socket pair to simulate network I/O
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('127.0.0.1', 0))
        server_sock.listen(1)
        
        port = server_sock.getsockname()[1]
        
        # Fork process to simulate multi-GPU communication
        pid = os.fork()
        
        if pid == 0:  # Child process (receiver)
            try:
                conn, addr = server_sock.accept()
                for _ in range(num_transfers):
                    # Receive data
                    received = 0
                    target = size_mb * 1024 * 1024
                    while received < target:
                        chunk = conn.recv(min(1024*1024, target - received))
                        if not chunk:
                            break
                        received += len(chunk)
                conn.close()
            finally:
                sys.exit(0)
        
        else:  # Parent process (sender)
            time.sleep(0.1)  # Let server start
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.connect(('127.0.0.1', port))
            
            chunk_size = 2 * 1024 * 1024  # 2MB chunks
            data = b'X' * chunk_size
            
            for i in range(num_transfers):
                sent = 0
                target = size_mb * 1024 * 1024
                while sent < target:
                    to_send = min(chunk_size, target - sent)
                    client_sock.send(data[:to_send])
                    sent += to_send
                
                # Simulate processing time
                time.sleep(0.1)
                
                if (i + 1) % 2 == 0:
                    print(f"   Progress: {i+1}/{num_transfers} transfers")
            
            client_sock.close()
            
            # Wait for child
            os.waitpid(pid, 0)
            server_sock.close()
            
            print(f"   ✓ Completed {num_transfers} network transfers")
    
    except Exception as e:
        print(f"   ⚠️  Network simulation failed: {e}")

def simulate_training_loop(num_epochs=3, batches_per_epoch=10):
    """Simulate a complete training loop"""
    print("\n" + "="*70)
    print("🚀 STARTING SIMULATED TRAINING")
    print("="*70)
    print(f"Configuration: {num_epochs} epochs, {batches_per_epoch} batches/epoch")
    print(f"Process PID: {os.getpid()}")
    print("\nTip: Run profiler in another terminal:")
    print(f"     sudo python3 gpu_training_profiler.py -p {os.getpid()} -d 60")
    print("="*70 + "\n")
    
    time.sleep(2)  # Give time to start profiler
    
    for epoch in range(num_epochs):
        print(f"\n📊 EPOCH {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        for batch in range(batches_per_epoch):
            print(f"\nBatch {batch + 1}/{batches_per_epoch}:")
            
            # 1. Data loading phase
            simulate_disk_io(size_mb=50, chunk_size_mb=10)
            
            # 2. GPU forward + backward pass
            simulate_gpu_compute(iterations=5, compute_time_ms=50)
            
            # 3. Gradient synchronization (NCCL) - only in multi-GPU
            if batch % 3 == 0:  # Not every batch to make it interesting
                simulate_network_transfer(size_mb=30, num_transfers=2)
            
            time.sleep(0.5)
        
        print(f"\n✅ Epoch {epoch + 1} completed")
        time.sleep(1)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED")
    print("="*70)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo Training Script")
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batches', type=int, default=10, help='Batches per epoch')
    parser.add_argument('--mode', choices=['full', 'io', 'gpu', 'network'], 
                       default='full', help='What to simulate')
    
    args = parser.parse_args()
    
    print(f"\nDemo Training Script - PID: {os.getpid()}")
    
    if args.mode == 'full':
        simulate_training_loop(args.epochs, args.batches)
    elif args.mode == 'io':
        print("\n🔍 Testing Disk I/O only...")
        for i in range(5):
            simulate_disk_io(size_mb=100, chunk_size_mb=20)
            time.sleep(1)
    elif args.mode == 'gpu':
        print("\n🔍 Testing GPU compute only...")
        simulate_gpu_compute(iterations=20, compute_time_ms=100)
    elif args.mode == 'network':
        print("\n🔍 Testing Network transfer only...")
        for i in range(5):
            simulate_network_transfer(size_mb=100, num_transfers=3)
            time.sleep(1)

if __name__ == "__main__":
    main()
