#!/usr/bin/env python3
"""Run Qwen 0.5B ONNX on Adreno GPU via fiboaisdk ORT 1.22 C API + QNN EP."""
import ctypes, os, time, numpy as np
from ctypes import c_char_p, c_void_p, c_int, c_int64, c_size_t, c_float, POINTER, byref, cast, cdll

os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"

ort = cdll.LoadLibrary("/usr/local/lib/python3.8/dist-packages/fiboaisdk/libonnxruntime.so")

# === ORT C API function definitions ===
def ort_func(name, restype, *argtypes):
    f = getattr(ort, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f

OrtCreateEnv = ort_func("OrtCreateEnv", c_void_p,
    c_int, c_void_p, POINTER(c_void_p))
OrtCreateSessionOptions = ort_func("OrtCreateSessionOptions", c_void_p,
    POINTER(c_void_p))
OrtSetSessionGraphOptimizationLevel = ort_func("OrtSetSessionGraphOptimizationLevel", c_void_p,
    c_void_p, c_int)

# Create env
env = c_void_p()
status = OrtCreateEnv(2, None, byref(env))  # ORT_LOGGING_LEVEL_ERROR=2
assert status is None, f"CreateEnv failed: {status}"
print(f"ORT Env created: {env.value:#x}")

# Create session options
opts = c_void_p()
OrtCreateSessionOptions(byref(opts))
print(f"Session options created: {opts.value:#x}")

# Try to append QNN EP - the function name might vary
# Common names: OrtSessionOptionsAppendExecutionProviderQnn,
# OrtSessionOptionsAppendExecutionProvider_QNN

# First, list all symbols
import subprocess
r = subprocess.run(
    "nm -D /usr/local/lib/python3.8/dist-packages/fiboaisdk/libonnxruntime.so 2>/dev/null | grep -i 'OrtSessionOptionsAppend.*Qnn\|OrtSessionOptionsAppend.*QNN\|AppendExecutionProvider'",
    shell=True, capture_output=True, text=True, timeout=10)

print("\nQNN-related ORT symbols:")
for line in sorted(r.stdout.strip().split("\n")):
    sym = line.split()[-1] if line.split() else line
    if sym.startswith("Ort"):
        print(f"  {sym}")
