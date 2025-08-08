import torch
import torch.nn as nn
import math
from torchviz import make_dot
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear transformations
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiscaleTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiScaleAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.mha(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class MultiscaleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads=6, 
                 num_layers=6, dropout=0.1, max_len=5000, embedding_matrix=None):
        super().__init__()
        
        # Automatically adjust num_heads to be compatible with embedding_dim
        self.num_heads = min(6, embedding_dim // 64)  # Ensure at least 64 dimensions per head
        
        # Ensure embedding_dim is divisible by num_heads
        if embedding_dim % self.num_heads != 0:
            # Adjust num_heads to be the largest factor of embedding_dim that's <= original num_heads
            for i in range(self.num_heads, 0, -1):
                if embedding_dim % i == 0:
                    self.num_heads = i
                    break
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # Multiple transformer layers with different scales
        self.transformer_layers = nn.ModuleList([
            MultiscaleTransformerLayer(
                d_model=embedding_dim,
                num_heads=self.num_heads,  # Use adjusted num_heads
                d_ff=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 添加输出层，确保输出维度与hidden_dim匹配
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Transform to final hidden dimension
        x = self.output_layer(x)
            
        return x
    
    def visualize(self, save_path='model_architecture', batch_size=1, seq_length=128):
        """
        Visualize the model architecture using torchviz
        
        Args:
            save_path: Path to save the visualization (without extension)
            batch_size: Batch size for dummy input
            seq_length: Sequence length for dummy input
        """
        try:
            # Create dummy input
            x = torch.randint(0, 1000, (batch_size, seq_length))
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Forward pass
            output = self(x)
            
            # Create visualization
            dot = make_dot(output, params=dict(self.named_parameters()),
                         show_attrs=True, show_saved=True)
            
            # Customize graph appearance
            dot.attr(rankdir='TB')  # Top to bottom layout
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Add model summary to graph
            summary = f"""MultiscaleTransformer Architecture
            Vocab Size: {self.embedding.num_embeddings}
            Embedding Dim: {self.embedding.embedding_dim}
            Num Heads: {self.num_heads}
            Num Layers: {len(self.transformer_layers)}
            """
            dot.attr(label=summary)
            
            # Save visualization
            dot.render(save_path, format='png', cleanup=True)
            
            # Generate detailed text summary
            summary_path = f"{save_path}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("MultiscaleTransformer Architecture Summary\n")
                f.write("=====================================\n\n")
                
                # Model configuration
                f.write("Model Configuration:\n")
                f.write(f"- Vocabulary Size: {self.embedding.num_embeddings}\n")
                f.write(f"- Embedding Dimension: {self.embedding.embedding_dim}\n")
                f.write(f"- Number of Attention Heads: {self.num_heads}\n")
                f.write(f"- Number of Transformer Layers: {len(self.transformer_layers)}\n\n")
                
                # Layer details
                f.write("Layer Structure:\n")
                for name, module in self.named_children():
                    f.write(f"\n{name}:\n")
                    f.write(str(module))
                    f.write("\n")
                
                # Parameter statistics
                total_params = sum(p.numel() for p in self.parameters())
                trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
                f.write(f"\nParameter Statistics:\n")
                f.write(f"- Total Parameters: {total_params:,}\n")
                f.write(f"- Trainable Parameters: {trainable_params:,}\n")
                f.write(f"- Non-trainable Parameters: {total_params - trainable_params:,}\n")
            
            print(f"Model visualization saved to {save_path}.png")
            print(f"Model summary saved to {summary_path}")
            
        except Exception as e:
            print(f"Error visualizing model: {str(e)}") 