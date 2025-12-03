import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=None, dropout=0.1):
        super(SelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim])).cuda() if torch.cuda.is_available() else \
                     torch.sqrt(torch.FloatTensor([attention_dim]))
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(attention_dim, input_dim)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.query(x)  # (batch_size, seq_len, attention_dim)
        k = self.key(x)    # (batch_size, seq_len, attention_dim)
        v = self.value(x)  # (batch_size, seq_len, attention_dim)
        
        # Attention scores
        # (batch_size, seq_len, attention_dim) x (batch_size, attention_dim, seq_len)
        # -> (batch_size, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Convert attention mask to a binary mask
            binary_mask = mask.unsqueeze(1).repeat(1, seq_len, 1)
            # Apply mask by setting masked positions to a large negative number
            attn_scores = attn_scores.masked_fill(binary_mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, attention_dim)
        # -> (batch_size, seq_len, attention_dim)
        output = torch.matmul(attn_weights, v)
        
        # Apply output projection
        output = self.output_layer(output)
        
        return output, attn_weights
        
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, head_dim=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        
        # If head_dim is not specified, divide input_dim by num_heads
        self.head_dim = head_dim if head_dim is not None else input_dim // num_heads
        self.total_dim = self.head_dim * num_heads
        
        # Linear projections
        self.query = nn.Linear(input_dim, self.total_dim)
        self.key = nn.Linear(input_dim, self.total_dim)
        self.value = nn.Linear(input_dim, self.total_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda() if torch.cuda.is_available() else \
                     torch.sqrt(torch.FloatTensor([self.head_dim]))
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.total_dim, input_dim)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and split into multiple heads
        # (batch_size, seq_len, total_dim) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        # (batch_size, num_heads, seq_len, head_dim) x (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Convert attention mask to a binary mask and expand for multiple heads
            binary_mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, seq_len, 1)
            # Apply mask by setting masked positions to a large negative number
            attn_scores = attn_scores.masked_fill(binary_mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2)
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, total_dim)
        output = output.contiguous().view(batch_size, seq_len, self.total_dim)
        
        # Apply output projection
        output = self.output_layer(output)
        
        return output
