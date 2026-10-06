#!/usr/bin/env python3
"""Full Qwen3 0.6B QNN DSP benchmark with proper chat template."""
import os, sys, time
os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"
from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api
from fiboaisdk.api_aisdk_py import license_py as license_api

def read_file(path):
    with open(path, 'rb') as f:
        return f.read()

# Init license
lk1 = read_file("/home/fibo/qcom_6490_license/key1.pem")
lk2 = read_file("/home/fibo/qcom_6490_license/key2.pem")
lk3 = read_file("/home/fibo/qcom_6490_license/key3.pem")
ldata = read_file("/home/fibo/qcom_6490_license/license.bin")
license_api.Init(lk1, lk2, lk3, ldata)

# Init model
print("Loading Qwen3 0.6B QNN DSP...")
api = nlp_api.NLPAPI()
api.Init("/home/fibo/AI_model/llm_models/Qwen3-0.6B", "")
print("Loaded!")

# Test prompts
prompts = [
    "你好，请用一句话介绍北京。",
    "1+1等于几？",
]

for prompt in prompts:
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"{'='*50}")

    t0 = time.time()
    result = nlp_api.ResultNlpText()
    api.GenerateSync(prompt, result)
    elapsed = time.time() - t0

    output = result.text.strip()
    # Rough token count (Chinese: ~2 chars per token)
    out_chars = len(output)
    tok_per_s = out_chars / elapsed / 1.5  # rough estimate

    print(f"Output: {output}")
    print(f"Time: {elapsed:.1f}s | Output chars: {out_chars} | ~{tok_per_s:.1f} tok/s")

# Check GPU and CPU usage
print(f"\n--- System ---")
try:
    with open("/sys/class/kgsl/kgsl-3d0/gpubusy") as f:
        print(f"GPU busy: {f.read().strip()}")
except: pass

import subprocess
r = subprocess.run(["top","-b","-n","1"], capture_output=True, text=True)
for line in r.stdout.split("\n"):
    if "CPU" in line or "Mem" in line:
        print(line)

api.Release()
print("Done!")
