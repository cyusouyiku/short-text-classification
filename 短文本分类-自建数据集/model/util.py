import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoTokenizer
def positional_encoding(seq_len, d_model, device='cpu'):
    """
    Create positional encoding matrix for sequence inputs
    
    Args:
        seq_len: length of input sequence
        d_model: dimension of the model
        device: device to place tensor on
        
    Returns:
        Tensor of shape (1, seq_len, d_model) with positional encodings
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # (1, seq_len, d_model)

class PositionalEncoding(nn.Module):
    """
    Module for adding positional encoding to embeddings
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encodings added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class RoBERTaEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        """
        Args:
            embed_dim: 目标词向量维度，用于与静态词向量保持一致
            max_len: 最大序列长度
        """
        super(RoBERTaEmbedding, self).__init__()
        self.roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.max_len = max_len
        
        # 将RoBERTa的768维和静态词向量的256维拼接后降维到256维
        self.dim_reduction = nn.Linear(768 + embed_dim, embed_dim)
        
    def forward(self, text_list, static_embeddings):
        """
        Args:
            text_list: 文本列表 [batch_size]
            static_embeddings: 静态词向量 (batch_size, seq_len, embed_dim)
        Returns:
            降维后的组合词向量 (batch_size, seq_len, embed_dim)
        """
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.roberta.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.roberta(**inputs)
        
        # 获取动态词向量
        dynamic_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        
        # 确保序列长度匹配
        min_len = min(dynamic_embeddings.size(1), static_embeddings.size(1))
        dynamic_embeddings = dynamic_embeddings[:, :min_len, :]
        static_embeddings = static_embeddings[:, :min_len, :]
        
        # 拼接动态和静态词向量
        combined = torch.cat([dynamic_embeddings, static_embeddings], dim=-1)  # (batch_size, seq_len, 768+embed_dim)
        
        # 降维到指定维度
        final_embeddings = self.dim_reduction(combined)  # (batch_size, seq_len, embed_dim)
        return final_embeddings
