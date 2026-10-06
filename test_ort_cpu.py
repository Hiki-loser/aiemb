#!/usr/bin/env python3
"""Benchmark Qwen 0.5B ONNX on ORT 1.26 CPU on SC171.
Run with: /home/fibo/miniconda3/bin/python test_ort_cpu.py
"""
import numpy as np, time
import onnxruntime as ort

MODEL = "/home/fibo/qwen_onnx/model_raw.onnx"

print(f"ORT: {ort.__version__}, EPs: {ort.get_available_providers()}")

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 6

sess = ort.InferenceSession(MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
print(f"Inputs: {[i.name for i in sess.get_inputs()]}")

for seq_len in [8, 32, 64, 128]:
    i = np.zeros((1, seq_len), dtype=np.int64)
    a = np.ones((1, seq_len), dtype=np.int64)
    p = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

    # Warmup
    sess.run(None, {"input_ids": i, "attention_mask": a, "position_ids": p})

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sess.run(None, {"input_ids": i, "attention_mask": a, "position_ids": p})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"seq={seq_len:3d}: {avg_ms:6.0f}ms | shape={sess.run(None, {'input_ids': i, 'attention_mask': a, 'position_ids': p})[0].shape}")

print("Done!")
