import torch
import torch.nn as nn
import math
from torchviz import make_dot
import os
import torch.nn.functional as F

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
        
        # 动态调整num_heads确保能整除embedding_dim
        self.num_heads = 1  # 固定为1个头
        assert embedding_dim % self.num_heads == 0, "embedding_dim必须能被num_heads整除"
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # 多尺度Transformer层
        self.transformer_layers = nn.ModuleList([
            MultiscaleTransformerLayer(
                d_model=embedding_dim,
                num_heads=min(i + 1, 3),  # 逐层增加注意力头数：1, 2, 3
                d_ff=hidden_dim * (2 ** min(i // 2, 2)),  # 逐层增加前馈网络维度
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # 多尺度卷积特征提取
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2**i, 
                         stride=1, padding=2**(i-1)),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(1, 4)  # kernel_size: 2, 4, 8
        ])
        
        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 输出投影
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _extract_multiscale_features(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        batch_size, seq_len, _ = x.size()
        
        # 转换维度用于卷积
        x_conv = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            # 应用卷积获取不同尺度特征
            scale_feature = conv(x_conv)
            # 确保输出长度与输入相同
            if scale_feature.size(-1) != seq_len:
                scale_feature = F.interpolate(
                    scale_feature, 
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                )
            multi_scale_features.append(scale_feature)
        
        # 堆叠多尺度特征
        stacked_features = torch.stack(multi_scale_features, dim=1)  # [batch_size, num_scales, embedding_dim, seq_len]
        
        # 计算尺度注意力权重
        scale_weights = []
        for i in range(stacked_features.size(1)):
            feature = stacked_features[:, i].transpose(1, 2)  # [batch_size, seq_len, embedding_dim]
            weight = self.scale_attention(feature)  # [batch_size, seq_len, 1]
            scale_weights.append(weight)
        
        scale_weights = torch.cat(scale_weights, dim=-1)  # [batch_size, seq_len, num_scales]
        scale_weights = F.softmax(scale_weights, dim=-1).unsqueeze(2)  # [batch_size, seq_len, 1, num_scales]
        
        # 加权融合不同尺度的特征
        stacked_features = stacked_features.permute(0, 3, 2, 1)  # [batch_size, seq_len, embedding_dim, num_scales]
        weighted_features = (stacked_features * scale_weights).sum(-1)  # [batch_size, seq_len, embedding_dim]
        
        return weighted_features
        
    def forward(self, x, attention_mask=None):
        # 词嵌入和位置编码
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 保存原始特征
        original_features = x
        
        # Transformer层处理
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
            
        # 提取多尺度特征
        multi_scale_features = self._extract_multiscale_features(original_features)
        
        # 特征融合：transformer输出和多尺度特征
        x = torch.cat([x, multi_scale_features], dim=-1)
        x = self.feature_fusion(x)
        
        # 输出层变换
        output = self.output_layer(x)
            
        return output

    def visualize(self, save_path='model_architecture', batch_size=1, seq_length=128):
        try:
            # Create dummy input
            x = torch.randint(0, 1000, (batch_size, seq_length))
            
            # Forward pass
            output = self(x)
            
            # Create visualization
            dot = make_dot(output, params=dict(self.named_parameters()),
                         show_attrs=True, show_saved=True)
            
            # Customize graph appearance
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Add model summary
            summary = f"""MultiscaleTransformer Architecture
            Vocab Size: {self.embedding.num_embeddings}
            Embedding Dim: {self.embedding.embedding_dim}
            Num Heads: {self.num_heads}
            Num Layers: {len(self.transformer_layers)}
            """
            dot.attr(label=summary)
            
            # Save visualization
            dot.render(save_path, format='png', cleanup=True)
            
            print(f"Model visualization saved to {save_path}.png")
            
        except Exception as e:
            print(f"Error visualizing model: {str(e)}") 