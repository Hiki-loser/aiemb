#!/usr/bin/env python3
"""Export Qwen2.5-0.5B to ONNX with FIXED shapes for QNN GPU compatibility."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_PATH = "/home/hiki/Code/school/aiemb/models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("/home/hiki/Code/school/aiemb/qwen0.5b_onnx")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32,
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

# FIXED shapes (no dynamic axes) for QNN compatibility
B, S = 1, 128  # batch=1, seq=128
dummy = (torch.zeros((B,S),dtype=torch.long),
         torch.ones((B,S),dtype=torch.long),
         torch.arange(S,dtype=torch.long).unsqueeze(0))

out_fp32 = str(OUTPUT_DIR / "model_fixed128.onnx")
torch.onnx.export(w, dummy, out_fp32,
    input_names=["input_ids","attention_mask","position_ids"],
    output_names=["logits"],
    opset_version=16, export_params=True)
print(f"FP32 saved: {Path(out_fp32).stat().st_size/1024**2:.0f}MB graph + {Path(out_fp32+'.data').stat().st_size/1024**2:.0f}MB data")

# Also export seq=1 version for token generation
S2 = 1
dummy2 = (torch.zeros((B,S2),dtype=torch.long),
          torch.ones((B,S2),dtype=torch.long),
          torch.tensor([[S-1]],dtype=torch.long))

out_seq1 = str(OUTPUT_DIR / "model_seq1.onnx")
torch.onnx.export(w, dummy2, out_seq1,
    input_names=["input_ids","attention_mask","position_ids"],
    output_names=["logits"],
    opset_version=16, export_params=True)
print(f"Seq1 saved: {Path(out_seq1).stat().st_size/1024**2:.0f}MB graph + {Path(out_seq1+'.data').stat().st_size/1024**2:.0f}MB data")
print("Done!")
