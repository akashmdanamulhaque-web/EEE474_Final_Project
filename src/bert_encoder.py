import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextBERTEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, texts):
        # Determine the target device from BERT's own parameters
        device = next(self.parameters()).device
        
        # Tokenize text inputs
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # MOVE TOKENIZED INPUT TENSORS TO GPU/CPU DEVICE
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        outputs = self.bert(**inputs)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :] 
        return self.fc(cls_output)