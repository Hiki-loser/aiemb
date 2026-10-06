#!/usr/bin/env python3
"""Test Qwen 0.5B ONNX model with ONNX Runtime + QNN GPU on SC171."""
import os
import sys
import time
import numpy as np

# Set fiboaisdk library path for dependencies
os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"

MODEL_DIR = "/home/fibo/qwen_onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "model_raw.onnx")

# Add ORT to path
sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages/fiboaisdk")

import onnxruntime as ort

print("Available providers:", ort.get_available_providers())
print()

# Try with different EPs
for ep in ["QNNExecutionProvider", "CPUExecutionProvider"]:
    print(f"=== Testing {ep} ===")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=sess_options,
            providers=[ep],
        )
        print(f"  Session created with: {session.get_providers()}")

        # Test forward pass
        seq_len = 8
        input_ids = np.zeros((1, seq_len), dtype=np.int64)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        t0 = time.time()
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        elapsed = time.time() - t0

        logits = outputs[0]
        print(f"  Output shape: {logits.shape}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {e}")
    print()
