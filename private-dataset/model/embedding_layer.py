"""
Unified Embedding Layer: RoBERTa + Word2Vec static + Positional embeddings
According to the paper: "RoBERTa-based contextualized embeddings, Word2Vec static embeddings, 
and positional embeddings are combined to capture context-dependent semantics, fixed lexical 
features, and sequential structure."
"""
import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoTokenizer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Gehring et al., 2017)"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class UnifiedEmbeddingLayer(nn.Module):
    """
    Combines RoBERTa contextualized embeddings, Word2Vec static embeddings, and positional embeddings.
    """
    def __init__(self, embed_dim=256, max_len=128, dropout=0.3, 
                 roberta_model='hfl/chinese-roberta-wwm-ext', 
                 static_embedding_matrix=None, freeze_roberta=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # RoBERTa for contextualized embeddings
        self.roberta = AutoModel.from_pretrained(roberta_model)
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.roberta_dim = self.roberta.config.hidden_size  # 768
        
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Static (Word2Vec) embedding - use RoBERTa vocab for alignment
        self.roberta_vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
        self.static_embedding = nn.Embedding(
            self.roberta_vocab_size, embed_dim, 
            padding_idx=self.tokenizer.pad_token_id or 0
        )
        if static_embedding_matrix is not None:
            if static_embedding_matrix.shape[0] == self.roberta_vocab_size:
                self.static_embedding.weight.data.copy_(static_embedding_matrix)
            elif static_embedding_matrix.shape[1] == embed_dim:
                nn.init.normal_(self.static_embedding.weight, 0, 0.02)
                self.static_embedding.weight.data[self.tokenizer.pad_token_id or 0] = 0
        
        # Projection: RoBERTa(768) + static(256) -> embed_dim(256)
        self.projection = nn.Linear(self.roberta_dim + embed_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def tokenize(self, texts, device=None):
        """Tokenize texts with RoBERTa tokenizer"""
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        if device is not None:
            encodings = {k: v.to(device) for k, v in encodings.items()}
        return encodings
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: RoBERTa tokenizer output (batch_size, seq_len)
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
        Returns:
            combined embeddings (batch_size, seq_len, embed_dim)
        """
        # RoBERTa contextualized embeddings
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        roberta_emb = roberta_outputs.last_hidden_state  # (B, L, 768)
        
        # Static (Word2Vec) embeddings
        static_emb = self.static_embedding(input_ids)  # (B, L, embed_dim)
        
        # Combine RoBERTa + static
        combined = torch.cat([roberta_emb, static_emb], dim=-1)  # (B, L, 768+256)
        combined = self.projection(combined)  # (B, L, embed_dim)
        
        # Add positional encoding
        combined = self.pos_encoding(combined)
        
        combined = self.dropout(combined)
        
        if attention_mask is not None:
            combined = combined * attention_mask.unsqueeze(-1)
        
        return combined
