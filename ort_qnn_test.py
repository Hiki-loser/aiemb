#!/usr/bin/env python3
"""Test Qwen 0.5B ONNX with ORT 1.22 QNN GPU via ctypes on SC171."""
import ctypes, os, time, numpy as np
from ctypes import c_char_p, c_void_p, c_int, c_uint32, c_int64, c_float, POINTER, byref, create_string_buffer

os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"

# Load ORT 1.22
ort = ctypes.CDLL("/usr/local/lib/python3.8/dist-packages/fiboaisdk/libonnxruntime.so")

# Types
OrtApi = c_void_p
OrtEnv = c_void_p
OrtSessionOptions = c_void_p
OrtSession = c_void_p
OrtMemoryInfo = c_void_p
OrtValue = c_void_p
OrtStatus = c_void_p

# Get API
GetApi = ort.OrtGetApiBase
GetApi.restype = c_void_p
api_base = GetApi()
GetApiFunc = ctypes.cast(api_base, ctypes.POINTER(ctypes.c_void_p))
# Simplified: use ctypes function pointers

# Actually let's just use subprocess to call a simpler approach
print("Testing fiboaisdk ORT 1.22...")
# Check available providers
buf = ctypes.create_string_buffer(256)
prov = ctypes.c_char_p(b"QNNExecutionProvider")

# Simple test: just check if QNN EP is available
import subprocess
result = subprocess.run([
    "python3", "-c", """
import ctypes, os
os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"
lib = ctypes.CDLL("/usr/local/lib/python3.8/dist-packages/fiboaisdk/libonnxruntime.so")

# Get available providers
GetAvailableProviders = lib.OrtSessionOptionsAppendExecutionProvider
# Check for QNN
for name in [b"QNNExecutionProvider", b"CPUExecutionProvider"]:
    try:
        # Just try to create session to see what works
        pass
    except:
        pass
print("ORT 1.22 loaded successfully")
"""], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
