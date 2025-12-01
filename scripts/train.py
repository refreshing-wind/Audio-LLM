import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from src.model import AudioQwenModel
from src.dataset import AudioTextDataset
from src.config import QWEN3_0_6B_MODEL_PATH
import argparse

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model to {output_dir}...")
        
        # Save LoRA (if applicable)
        if hasattr(self.model.llm, "save_pretrained"):
            self.model.llm.save_pretrained(output_dir)
        
        # Save Projector
        torch.save(self.model.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
        
        # Save Tokenizer
        # self.tokenizer.save_pretrained(output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio-LLM Model")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Training stage: 1 for Pre-alignment, 2 for Instruction Tuning")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (required for Stage 2)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "checkpoints"), help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Load Tokenizer
    print(f"Loading tokenizer from {QWEN3_0_6B_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    print("Loading base model...")
    model = AudioQwenModel(QWEN3_0_6B_MODEL_PATH)
    
    if args.stage == 1:
        print("\n=== Stage 1: Pre-alignment ===")
        print("Goal: Train Projector to align audio features with frozen LLM.")
        
        # Freeze LLM completely
        for param in model.llm.parameters():
            param.requires_grad = False
            
        # Enable Projector
        for param in model.projector.parameters():
            param.requires_grad = True
            
        print("LLM frozen. Projector trainable.")
        
    elif args.stage == 2:
        print("\n=== Stage 2: Instruction Tuning ===")
        print("Goal: Fine-tune LLM (LoRA) and Projector for the task.")
        
        # Load Projector weights from Stage 1
        if args.checkpoint:
            print(f"Loading projector weights from {args.checkpoint}")
            
            # 1. Try projector.pt (New format)
            pt_path = os.path.join(args.checkpoint, "projector.pt")
            if os.path.exists(pt_path):
                model.projector.load_state_dict(torch.load(pt_path, map_location="cpu"))
                print("Projector weights loaded from projector.pt")
            else:
                # 2. Try pytorch_model.bin / safetensors (Old format / Full dump)
                ckpt_path = os.path.join(args.checkpoint, "pytorch_model.bin")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(args.checkpoint, "model.safetensors")
                
                if os.path.exists(ckpt_path):
                    if ckpt_path.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(ckpt_path)
                    else:
                        state_dict = torch.load(ckpt_path, map_location="cpu")
                    
                    projector_dict = {k.replace('projector.', ''): v for k, v in state_dict.items() if k.startswith('projector.')}
                    if projector_dict:
                        model.projector.load_state_dict(projector_dict)
                        print("Projector weights loaded from checkpoint dump.")
                    else:
                        print("WARNING: No projector weights found in checkpoint dump.")
                else:
                    print("WARNING: No projector weights found.")
        else:
            print("WARNING: No checkpoint provided for Stage 2. Projector will be random initialized!")

        # Apply LoRA to LLM
        print("Applying LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] 
        )
        model.llm = get_peft_model(model.llm, peft_config)
        model.llm.print_trainable_parameters()
        
        # Ensure Projector is trainable (Fine-tune)
        for param in model.projector.parameters():
            param.requires_grad = True
            
    # Dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_path = os.path.join(base_dir, "data", "processed", "train.json")
    
    if not os.path.exists(train_data_path):
        # Fallback or error
        pass
        
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Train data not found at {train_data_path}. Please run scripts/process_chiser5.py first.")
        
    print(f"Loading training data from {train_data_path}")
    dataset = AudioTextDataset(train_data_path, tokenizer)
    
    # Training Arguments
    output_dir = f"{args.output_dir}/stage{args.stage}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        learning_rate=1e-3 if args.stage == 1 else 1e-4,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False 
    )
    
    # Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print(f"Starting training for Stage {args.stage}...")
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    print(f"Stage {args.stage} training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    train()
