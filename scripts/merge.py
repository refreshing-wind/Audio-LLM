import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.model import AudioQwenModel
from src.config import QWEN3_0_6B_MODEL_PATH
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Merge Audio-LLM Weights")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (Stage 2)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    return parser.parse_args()

def merge_model():
    args = parse_args()
    
    print(f"Loading base model from {QWEN3_0_6B_MODEL_PATH}...")
    # Load base model
    base_llm = AutoModelForCausalLM.from_pretrained(
        QWEN3_0_6B_MODEL_PATH, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B_MODEL_PATH, trust_remote_code=True)

    print(f"Loading LoRA adapters from {args.checkpoint}...")
    # Load LoRA model
    model_to_merge = PeftModel.from_pretrained(base_llm, args.checkpoint)
    
    print("Merging LoRA weights into base model...")
    # Merge LoRA weights
    merged_llm = model_to_merge.merge_and_unload()
    
    print(f"Saving merged LLM to {args.output_dir}...")
    merged_llm.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Handle Projector
    print("Handling Projector weights...")
    
    # 1. Try projector.pt (New format)
    pt_path = os.path.join(args.checkpoint, "projector.pt")
    if os.path.exists(pt_path):
        print(f"Found projector.pt at {pt_path}")
        shutil.copy(pt_path, os.path.join(args.output_dir, "projector.pt"))
        print("Projector weights copied.")
    else:
        # 2. Try pytorch_model.bin / safetensors (Old format / Full dump)
        ckpt_path = os.path.join(args.checkpoint, "pytorch_model.bin")
        if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.checkpoint, "model.safetensors")
        
        if os.path.exists(ckpt_path):
            print(f"Extracting projector from {ckpt_path}...")
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                
            # Extract projector weights
            projector_dict = {k.replace('projector.', ''): v for k, v in state_dict.items() if k.startswith('projector.')}
            
            if projector_dict:
                projector_path = os.path.join(args.output_dir, "projector.pt")
                torch.save(projector_dict, projector_path)
                print(f"Projector weights saved to {projector_path}")
            else:
                print("WARNING: No projector weights found in checkpoint.")
        else:
            print("WARNING: Checkpoint file not found.")

    print("Merge complete!")

if __name__ == "__main__":
    merge_model()
