#!/usr/bin/env python3
"""Run Qwen 0.5B ONNX on Adreno GPU via fiboaisdk InferAPI + QNN GPU backend."""
import os, sys, time, numpy as np
os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"
from fiboaisdk import api_aisdk_py

MODEL_PATH = "/home/fibo/model_fixed128.onnx"

# Init with qualcomm + GPU backend
api = api_aisdk_py.api_infer_py.InferAPI()
params = api_aisdk_py.api_infer_py.InferParams(MODEL_PATH, "qualcomm", "ONNX", "GPU", "high", 0)
print(f"Config: fw={params.get_framework()}, dev={params.get_device_unit()}")

t0 = time.time()
ret = api.Init(params)
print(f"Init: {ret} ({time.time()-t0:.1f}s)")

if ret != 0:
    print("Init failed, trying CPU fallback...")
    params2 = api_aisdk_py.api_infer_py.InferParams(MODEL_PATH, "qualcomm", "ONNX", "CPU", "high", 0)
    ret = api.Init(params2)
    print(f"CPU Init: {ret}")

# Run inference with fixed seq_len=128 (matching model export)
seq_len = 128
input_ids = np.zeros((1, seq_len), dtype=np.int64)
attention_mask = np.ones((1, seq_len), dtype=np.int64)
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

print(f"Running inference (seq_len={seq_len})...")
# Convert numpy arrays to lists for Execute_int64
inputs = {
    "input_ids": input_ids.flatten().tolist(),
    "attention_mask": attention_mask.flatten().tolist(),
    "position_ids": position_ids.flatten().tolist(),
}

t0 = time.time()
ret = api.Execute_int64(inputs)
print(f"Execute: {ret} ({time.time()-t0:.1f}s)")

outputs = api.FetchOutputs_float(["logits"])
print(f"FetchOutputs: {type(outputs)}")
if isinstance(outputs, dict):
    for name, arr in outputs.items():
        print(f"  {name}: len={len(arr)}")
elif isinstance(outputs, tuple):
    print(f"  ret={outputs[0]}, data_keys={list(outputs[1].keys()) if isinstance(outputs[1], dict) else 'N/A'}")

# Check GPU usage
try:
    with open("/sys/class/kgsl/kgsl-3d0/gpubusy") as f:
        print(f"\ngpubusy: {f.read().strip()}")
except:
    pass

api.Release()
print("Done!")
