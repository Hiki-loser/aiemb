#!/usr/bin/env python3
"""Convert ONNX model to MNN format via MNN.expr."""
import MNN
from pathlib import Path

ONNX_PATH = "/home/hiki/Code/school/aiemb/qwen0.5b_onnx/model.onnx"
MNN_OUTPUT = "/home/hiki/Code/school/aiemb/qwen0.5b_onnx/qwen0.5b.mnn"

print(f"Loading ONNX: {ONNX_PATH}")

# Load ONNX into MNN expression graph
var_map = MNN.expr.load_as_dict(ONNX_PATH)
print(f"Variables loaded: {len(var_map)}")

# Check input/output vars
for name, var in var_map.items():
    print(f"  {name}: shape={var.shape}, dtype={var.data_type}")

# Save as MNN model
print(f"\nSaving MNN model to: {MNN_OUTPUT}")
# Find the logits output variable
output_vars = []
input_vars = []
for name, var in var_map.items():
    if name in ("input_ids", "attention_mask", "position_ids"):
        input_vars.append(var)
    if name in ("logits",):
        output_vars.append(var)

if not output_vars:
    # Try to find output var by checking last entry
    last_key = list(var_map.keys())[-1]
    output_vars = [var_map[last_key]]
    print(f"Using last var as output: {last_key}")

if not input_vars:
    first_keys = list(var_map.keys())[:3]
    input_vars = [var_map[k] for k in first_keys]
    print(f"Using first vars as inputs: {first_keys}")

MNN.expr.save(input_vars, output_vars, MNN_OUTPUT)

size_mb = Path(MNN_OUTPUT).stat().st_size / 1024**2
print(f"\nMNN model size: {size_mb:.1f} MB")
print("Done!")
