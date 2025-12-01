import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
from datasets import load_dataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from src.config import CHISER5_PATH, EMOTION2VEC_MODEL_PATH
from tqdm import tqdm
import shutil

def process_dataset():
    # 1. Load Dataset
    print(f"Loading dataset from {CHISER5_PATH}...")
    dataset = load_dataset(CHISER5_PATH)
    
    # 2. Initialize Emotion2Vec Pipeline
    print(f"Loading emotion2vec pipeline from {EMOTION2VEC_MODEL_PATH}...")
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model=EMOTION2VEC_MODEL_PATH,
        device='cpu' # Use CPU for stability if MPS is tricky with some ops, or 'mps' if supported
    )

    # 3. Prepare Output Directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, "data", "processed")
    feature_dir = os.path.join(output_base, "features")
    
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(feature_dir, exist_ok=True)

    # 4. Process Data
    all_data = []
    
    # Label mapping
    label_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    
    print("Processing audio files and extracting features...")
    # The dataset only has 'train' split based on previous inspection
    ds = dataset['train']
    
    for idx, item in tqdm(enumerate(ds), total=len(ds)):
        # Extract Audio
        audio_data = item['audio']['array']
        sr = item['audio']['sampling_rate']
        label_int = item['label']
        label_str = label_names[label_int]
        
        # Save temporary wav file for pipeline (pipeline expects path or bytes, but path is safer)
        # Or we can pass numpy array directly if pipeline supports it. 
        # Modelscope pipeline usually supports input as path.
        # Let's try passing the array directly if supported, otherwise save temp file.
        # Looking at previous code, it used a URL. 
        # Let's save to a temp file to be safe.
        temp_wav_path = f"temp_{idx}.wav"
        import soundfile as sf
        sf.write(temp_wav_path, audio_data, sr)
        
        try:
            # Extract Feature
            rec_result = inference_pipeline(temp_wav_path, granularity="utterance", extract_embedding=True)
            
            if isinstance(rec_result, list) and len(rec_result) > 0:
                feats = rec_result[0].get('feats')
                if feats is not None:
                    # Save Feature (.npy)
                    feature_filename = f"feat_{idx}.npy"
                    feature_path = os.path.join(feature_dir, feature_filename)
                    # Convert to float32 to save space and match torch default
                    np.save(feature_path, np.array(feats, dtype=np.float32))
                    
                    # Construct Text (Prompt) - REMOVED <audio> tag
                    # Input: "判断音频的情感分类，需要从['angry', 'happy', 'neutral', 'sad', 'surprise']中选择"
                    # Output: "这个音频为<分类>的情感"
                    
                    prompt = "判断音频的情感分类，需要从['angry', 'happy', 'neutral', 'sad', 'surprise']中选择"
                    response = f"这个音频为<{label_str}>的情感"
                    
                    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
                    
                    all_data.append({
                        "audio_feature_path": os.path.abspath(feature_path),
                        "text": text,
                        "label": label_str
                    })
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

    # 5. Split Dataset (70% Train, 20% Test, 10% Val - or just drop the rest?)
    # User asked for 70% Train, 20% Test. I'll use the remaining 10% as Validation.
    
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    total_len = len(all_data)
    train_len = int(total_len * 0.7)
    test_len = int(total_len * 0.2)
    # val_len = total_len - train_len - test_len
    
    train_data = all_data[:train_len]
    test_data = all_data[train_len:train_len+test_len]
    val_data = all_data[train_len+test_len:]
    
    print(f"Total samples: {total_len}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # 6. Save JSONs
    with open(os.path.join(output_base, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(output_base, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(output_base, 'valid.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
        
    print("Data processing complete.")

if __name__ == "__main__":
    process_dataset()
