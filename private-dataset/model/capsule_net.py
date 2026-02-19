import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)

class CapsuleNet(nn.Module):
    """Accepts pre-computed embeddings (RoBERTa+Word2Vec+Position)."""
    def __init__(self, embedding_dim, num_primary_units=8, 
                 num_primary_channels=64, num_classes=9, num_routing=3, 
                 dropout=0.5):
        super(CapsuleNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_primary_units = num_primary_units
        self.num_primary_channels = num_primary_channels
        self.num_classes = num_classes
        self.num_routing = num_routing
        
        # Primary Capsules
        self.conv1 = nn.Conv1d(embedding_dim, num_primary_channels, kernel_size=9, stride=1, padding=4)
        self.primary_capsules = nn.Conv1d(
            num_primary_channels, 
            num_primary_channels * num_primary_units,
            kernel_size=9, 
            stride=2, 
            padding=4
        )
        
        # Text Capsules - ensure dimension alignment
        self.W = nn.Parameter(torch.randn(
            1, num_primary_units, num_classes,
            num_primary_channels, num_primary_channels
        ))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # x is pre-computed embeddings (batch_size, seq_len, embedding_dim)
        batch_size = x.size(0)
        embedded = x
        
        # Apply mask if provided
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)
            
        # Transpose for conv1d
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # First convolution layer
        x = F.relu(self.conv1(embedded))  # (batch_size, num_primary_channels, seq_len)
        
        # Primary capsules
        primary = self.primary_capsules(x)  # (batch_size, num_primary_channels * num_primary_units, seq_len/2)
        
        # Reshape and squash primary capsules
        primary = primary.view(batch_size, self.num_primary_units, self.num_primary_channels, -1)
        primary = primary.permute(0, 1, 3, 2)  # (batch_size, num_primary_units, seq_len/2, num_primary_channels)
        primary = squash(primary, dim=-1)
        
        # Get sequence length after convolutions
        seq_len = primary.size(2)
        
        # Prepare input for routing
        u_hat = torch.matmul(
            primary.unsqueeze(2),  # (batch_size, num_primary_units, 1, seq_len/2, num_primary_channels)
            self.W.repeat(batch_size, 1, 1, 1, 1)  # (batch_size, num_primary_units, num_classes, num_primary_channels, num_primary_channels)
        )  # (batch_size, num_primary_units, num_classes, seq_len/2, num_primary_channels)
        
        # Initialize routing logits
        b = torch.zeros(batch_size, self.num_primary_units, self.num_classes, seq_len).to(x.device)
        
        # Iterative routing
        for r in range(self.num_routing):
            c = F.softmax(b, dim=2)  # (batch_size, num_primary_units, num_classes, seq_len)
            c = c.unsqueeze(-1)  # (batch_size, num_primary_units, num_classes, seq_len, 1)
            
            # Calculate weighted sum
            s = (u_hat * c).sum(dim=1)  # (batch_size, num_classes, seq_len, num_primary_channels)
            v = squash(s, dim=-1)  # (batch_size, num_classes, seq_len, num_primary_channels)
            
            if r < self.num_routing - 1:
                # Update routing logits
                v_expanded = v.unsqueeze(1)  # (batch_size, 1, num_classes, seq_len, num_primary_channels)
                
                # Adjust dimensions for matrix multiplication alignment
                u_hat_flat = u_hat.view(batch_size, self.num_primary_units, self.num_classes, -1)  # (batch_size, num_primary_units, num_classes, seq_len * num_primary_channels)
                v_expanded_flat = v_expanded.view(batch_size, 1, self.num_classes, -1)  # (batch_size, 1, num_classes, seq_len * num_primary_channels)
                
                # Compute agreement
                agreement = torch.matmul(u_hat_flat, v_expanded_flat.transpose(-2, -1))  # (batch_size, num_primary_units, num_classes, num_classes)
                agreement = agreement.diagonal(dim1=-2, dim2=-1)  # Keep only diagonal elements
                agreement = agreement.unsqueeze(-1).expand(-1, -1, -1, seq_len)  # Expand to original dimensions
                
                b = b + agreement
        
        # Average over sequence length
        output = v.mean(dim=2)  # (batch_size, num_classes, num_primary_channels)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output

