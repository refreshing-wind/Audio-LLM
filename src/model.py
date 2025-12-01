import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class AudioQwenModel(nn.Module):
    def __init__(self, llm_model_path, audio_dim=768):
        super().__init__()
        
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)
        
        # Projector
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, audio_features, input_ids, attention_mask, labels=None):
        """
        audio_features: (Batch, Audio_Dim) or (Batch, Seq_Len, Audio_Dim)
        input_ids: (Batch, Text_Len)
        attention_mask: (Batch, Text_Len)
        labels: (Batch, Text_Len)
        """
        
        # Project audio features to LLM embedding space
        # audio_features shape: [Batch, Audio_Dim] -> [Batch, 1, LLM_Dim]
        # If audio_features is [Batch, Seq_Len, Audio_Dim], then [Batch, Seq_Len, LLM_Dim]
        
        audio_embeds = self.projector(audio_features)
        
        if len(audio_embeds.shape) == 2:
            audio_embeds = audio_embeds.unsqueeze(1)
            
        # Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Concatenate audio and text embeddings
        # [Audio, Text]
        inputs_embeds = torch.cat([audio_embeds, inputs_embeds], dim=1)
        
        # Update attention mask
        # Add 1s for audio part
        audio_len = audio_embeds.shape[1]
        batch_size = audio_embeds.shape[0]
        
        audio_mask = torch.ones(batch_size, audio_len, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
        
        # Update labels if provided
        if labels is not None:
            # Audio part should be ignored in loss (label = -100)
            audio_labels = torch.full((batch_size, audio_len), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([audio_labels, labels], dim=1)
            
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, audio_features, input_ids, attention_mask, **kwargs):
        # Prepare embeddings
        audio_embeds = self.projector(audio_features)
        if len(audio_embeds.shape) == 2:
            audio_embeds = audio_embeds.unsqueeze(1)
            
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([audio_embeds, inputs_embeds], dim=1)
        
        # Update attention mask
        audio_len = audio_embeds.shape[1]
        batch_size = audio_embeds.shape[0]
        audio_mask = torch.ones(batch_size, audio_len, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
        
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
