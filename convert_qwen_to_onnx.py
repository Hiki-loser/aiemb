#!/usr/bin/env python3
"""
Convert Qwen2.5-0.5B-Instruct to ONNX format.
Uses optimum for streamlined export with dynamic axes.
"""
import os
import shutil
from pathlib import Path

OUTPUT_DIR = Path("/home/hiki/Code/school/aiemb/qwen0.5b_onnx")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    # Set proxy for HuggingFace downloads
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:20171"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:20171"

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch

    print(f"Downloading {MODEL_ID} ...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load model in float32 for ONNX export
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save tokenizer and config alongside ONNX
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    config.save_pretrained(str(OUTPUT_DIR))

    # Export to ONNX using torch.onnx with dynamic axes
    print("Exporting to ONNX...")

    # Create dummy inputs
    batch_size = 1
    seq_len = 8
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    onnx_path = OUTPUT_DIR / "model.onnx"

    # Use optimum for cleaner export
    from optimum.onnxruntime import ORTModelForCausalLM
    from optimum.exporters import TasksManager
    from optimum.exporters.onnx import export

    # Export the decoder model (without past_key_values for initial export)
    export(
        model=model,
        config=config,
        output=onnx_path,
        opset=17,
        device="cpu",
    )

    print(f"ONNX model saved to: {onnx_path}")
    print(f"Model size: {onnx_path.stat().st_size / 1024**2:.1f} MB")
    print(f"Files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

if __name__ == "__main__":
    main()
