from torch.utils.data import Dataset
import torch
import json
import numpy as np

class AudioTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio feature
        # Assuming audio_feature_path points to a .npy file
        audio_feature = np.load(item['audio_feature_path'])
        
        # Average if it's a sequence (Batch, Seq_Len, Dim) -> (Dim,)
        # If it's (Seq_Len, Dim), mean(0) -> (Dim,)
        if len(audio_feature.shape) > 1:
             audio_feature = np.mean(audio_feature, axis=0)
             
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
        
        # Tokenize text
        text = item['text']
        
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        # Labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'audio_features': audio_feature,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
