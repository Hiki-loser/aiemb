#!/usr/bin/env python3
"""Export Qwen2.5-0.5B to FP32 ONNX (opset 14) for QNN GPU."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_PATH = "/home/hiki/Code/school/aiemb/models/Qwen/Qwen2___5-0___5B-Instruct"
OUTPUT_DIR = Path("/home/hiki/Code/school/aiemb/qwen0.5b_onnx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float32,
    trust_remote_code=True, device_map="cpu", low_cpu_mem_usage=True)
model.eval().cpu()
for p in model.parameters():
    p.requires_grad_(False)

class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, i, a, p):
        o = self.m(input_ids=i, attention_mask=a, position_ids=p,
                   use_cache=False, return_dict=True)
        return o.logits

w = Wrapper(model).cpu().eval()
B, S = 1, 32
# Pass 4D float32 attention_mask directly to skip model's internal Cast+Reshape
dummy = (torch.zeros((B,S),dtype=torch.long),
         torch.ones((B,1,1,S),dtype=torch.float32),
         torch.arange(S,dtype=torch.float32).unsqueeze(0))

out = str(OUTPUT_DIR / "model_fp32_fixed.onnx")
torch.onnx.export(w, dummy, out,
    input_names=["input_ids","attention_mask","position_ids"],
    output_names=["logits"],
    opset_version=14, export_params=True, dynamo=False,
    do_constant_folding=True)

print(f"ONNX saved: {out}")
print(f"Size: {Path(out).stat().st_size/1024**2:.0f}MB")
print("Done!")
