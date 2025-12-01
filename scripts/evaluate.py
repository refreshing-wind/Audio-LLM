import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from src.model import AudioQwenModel
import numpy as np
import json
from tqdm import tqdm
import re
import argparse

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(base_dir, "output", "audio_qwen_merged")
    default_test_file = os.path.join(base_dir, "data", "processed", "test.json")

    parser = argparse.ArgumentParser(description="Evaluate Audio-LLM on Test Set")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the merged model")
    parser.add_argument("--test_file", type=str, default=default_test_file, help="Path to test.json")
    args = parser.parse_args()

    # Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = AudioQwenModel(args.model_path)
    
    # Check for projector weights (Merged Model case)
    projector_pt_path = os.path.join(args.model_path, "projector.pt")
    if os.path.exists(projector_pt_path):
        print(f"Loading projector weights from {projector_pt_path}...")
        model.projector.load_state_dict(torch.load(projector_pt_path, map_location="cpu"))
    else:
        print("Warning: projector.pt not found in model path. Make sure the model is loaded correctly.")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        model.to("mps")
        print("Using MPS")
    else:
        print("Using CPU")
        
    # Load Test Data
    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    print(f"Found {len(test_data)} test samples.")
    
    # Prompt
    prompt = "判断音频的情感分类，需要从['angry', 'happy', 'neutral', 'sad', 'surprise']中选择"
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    encodings = tokenizer(input_text, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
        
    correct = 0
    total = 0
    
    results = []

    for item in tqdm(test_data):
        feature_path = item['audio_feature_path']
        true_label = item['label']
        
        # Load features
        try:
            if not os.path.exists(feature_path):
                print(f"Feature file not found: {feature_path}")
                continue

            audio_features = np.load(feature_path)
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
            
            # Reshape to [1, 1, Dim] as expected by model
            if len(audio_features.shape) == 1:
                audio_features = audio_features.unsqueeze(0)
            if len(audio_features.shape) == 2:
                 # If [1, Dim], make it [1, 1, Dim]
                 audio_features = audio_features.unsqueeze(0)
            
            audio_features = audio_features.to(device)
                
            with torch.no_grad():
                output_ids = model.generate(
                    audio_features=audio_features,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50
                )
            
            # Decode the generated tokens (output_ids contains only new tokens when inputs_embeds is used)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract label from generated text
            # Expected format: "这个音频为<label>的情感"
            match = re.search(r'<([^>]+)>', generated_text)
            if match:
                predicted_label = match.group(1)
            else:
                # Fallback: check if any label is in the text
                labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']
                predicted_label = "unknown"
                for l in labels:
                    if l in generated_text:
                        predicted_label = l
                        break
            
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "feature_path": feature_path,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "generated_text": generated_text,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"Error processing {feature_path}: {e}")
            
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation Complete.")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Save detailed results
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Detailed results saved to eval_results.json")

if __name__ == "__main__":
    main()
