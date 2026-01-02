#!/usr/bin/env python3
"""Test if the profiler has valid Python syntax."""

import py_compile
import sys

try:
    py_compile.compile('gpu_training_profiler.py', doraise=True)
    print("✅ gpu_training_profiler.py has VALID Python syntax")
    sys.exit(0)
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error found:")
    print(e)
    sys.exit(1)
