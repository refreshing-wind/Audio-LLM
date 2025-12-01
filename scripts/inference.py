import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from src.model import AudioQwenModel
from src.config import QWEN3_0_6B_MODEL_PATH, EMOTION2VEC_MODEL_PATH
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import argparse
import soundfile as sf

def parse_args():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(base_dir, "output", "audio_qwen_merged")
    if not os.path.exists(default_model_path):
        default_model_path = QWEN3_0_6B_MODEL_PATH

    parser = argparse.ArgumentParser(description="Inference Audio-LLM Model")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the model (base or merged)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (optional, if not merged)")
    parser.add_argument("--audio_path", type=str, default="dummy_audio.wav", help="Path to audio file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = AudioQwenModel(args.model_path)
    
    # Check for projector weights in model_path (Merged Model case)
    projector_pt_path = os.path.join(args.model_path, "projector.pt")
    if os.path.exists(projector_pt_path):
        print(f"Loading projector weights from {projector_pt_path}...")
        model.projector.load_state_dict(torch.load(projector_pt_path, map_location="cpu"))
    
    # Load Adapter if provided (Unmerged case)
    if args.adapter_path:
        print(f"Loading LoRA adapters from {args.adapter_path}...")
        # Load Projector from adapter checkpoint if not already loaded
        if not os.path.exists(projector_pt_path):
            ckpt_path = os.path.join(args.adapter_path, "pytorch_model.bin")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.adapter_path, "model.safetensors")
            
            if os.path.exists(ckpt_path):
                print(f"Loading projector weights from {ckpt_path}...")
                if ckpt_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(ckpt_path)
                else:
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                
                projector_dict = {k.replace('projector.', ''): v for k, v in state_dict.items() if k.startswith('projector.')}
                if projector_dict:
                    model.projector.load_state_dict(projector_dict)
        
        # Load LoRA
        try:
            model.llm = PeftModel.from_pretrained(model.llm, args.adapter_path)
            print("LoRA weights loaded.")
        except Exception as e:
            print(f"Could not load LoRA weights: {e}")

    model.eval()
    
    # Load Audio Pipeline
    print("Loading audio pipeline...")
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model=EMOTION2VEC_MODEL_PATH)
    
    # Audio Input
    if not os.path.exists(args.audio_path):
        print(f"Generating dummy audio at {args.audio_path}...")
        sr = 16000
        data = np.random.uniform(-1, 1, sr)
        sf.write(args.audio_path, data, sr)
        
    # Extract features
    print(f"Extracting features from {args.audio_path}...")
    rec_result = inference_pipeline(args.audio_path, granularity="utterance", extract_embedding=True)
    feats = rec_result[0]['feats']
    audio_features = torch.tensor(feats, dtype=torch.float32)
    
    # Average if sequence (emotion2vec usually returns [1, 768] for utterance level, but let's be safe)
    if len(audio_features.shape) > 1 and audio_features.shape[0] > 1:
         # If it returns multiple frames, average them or take the first one depending on config. 
         # For 'utterance' granularity it should be a single vector or [1, dim].
         pass
         
    if len(audio_features.shape) == 1:
        audio_features = audio_features.unsqueeze(0) # [1, Dim]
        
    audio_features = audio_features.unsqueeze(0) # Batch dimension [1, 1, Dim]
    
    # Text Input
    # Must match the prompt used in training (process_dataset.py)
    prompt = "判断音频的情感分类，需要从['angry', 'happy', 'neutral', 'sad', 'surprise']中选择"
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            audio_features=audio_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
