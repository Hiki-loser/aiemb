#!/usr/bin/env python3
"""INT8 quantize Qwen 0.5B ONNX model for GPU deployment."""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import os

MODEL_IN = "/home/hiki/Code/school/aiemb/qwen0.5b_onnx/model_fixed128.onnx"
MODEL_OUT = "/home/hiki/Code/school/aiemb/qwen0.5b_onnx/model_fixed128_int8.onnx"

print(f"Loading: {MODEL_IN}")
model = onnx.load(MODEL_IN)
print(f"IR version: {model.ir_version}, opsets: {[o.domain + ':' + str(o.version) for o in model.opset_import]}")

print("\nRunning dynamic INT8 quantization...")
print("  This converts MatMul/Gemm weights to INT8 (~4x smaller)")

quantize_dynamic(
    model_input=MODEL_IN,
    model_output=MODEL_OUT,
    weight_type=QuantType.QInt8,
)

size_in = os.path.getsize(MODEL_IN) + os.path.getsize(MODEL_IN + ".data")
size_out = os.path.getsize(MODEL_OUT)
size_data = os.path.getsize(MODEL_OUT + ".data") if os.path.exists(MODEL_OUT + ".data") else 0
print(f"\nOriginal: {size_in/1024**3:.2f} GB")
print(f"INT8:     {size_out/1024**2:.1f} MB (graph) + {size_data/1024**2:.1f} MB (data)")
print(f"Total:    {(size_out+size_data)/1024**2:.1f} MB ({(size_out+size_data)/size_in*100:.0f}%)")

# Verify model structure
model_q = onnx.load(MODEL_OUT)
print(f"\nINT8 model: IR v{model_q.ir_version}, {len(model_q.graph.initializer)} initializers, {len(model_q.graph.node)} nodes")

# Count quantized ops
q_nodes = [n for n in model_q.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear")]
print(f"Quantize/Dequantize nodes: {len(q_nodes)}")
print("Done!")
