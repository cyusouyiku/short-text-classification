import torch
import torch.nn as nn
import torch.nn.functional as F
from .capsule_net import CapsuleNet
from .multiscale_transformer import MultiscaleTransformer
from .self_attention import SelfAttention, MultiHeadAttention
from .policy_net import PolicyNetwork
from .reward_net import RewardNetwork
from .util import positional_encoding, PositionalEncoding

class TextClassificationModel(nn.Module):
    def __init__(self, config):
        super(TextClassificationModel, self).__init__()
        
        # Extract configuration
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config['num_classes']
        self.dropout = config.get('dropout', 0.5)
        
        # Word embedding
        self.word_embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=0
        )
        
        # Load pretrained embedding if provided
        if 'embedding_matrix' in config:
            self.word_embedding.weight.data.copy_(config['embedding_matrix'])
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.embedding_dim, 
            max_len=config.get('max_seq_len', 512),
            dropout=self.dropout
        )
        
        # CapsuleNet module
        self.capsule = CapsuleNet(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_primary_units=config.get('num_primary_units', 8),
            num_primary_channels=config.get('num_primary_channels', 64),
            num_classes=self.num_classes,
            num_routing=config.get('num_routing', 3),
            dropout=self.dropout
        )
        
        # MultiscaleTransformer module
        self.transformer = MultiscaleTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            dropout=self.dropout,
            max_len=config.get('max_seq_len', 512),
            embedding_matrix=config.get('embedding_matrix', None)
        )
        
        # Calculate feature dimensions
        self.capsule_feature_dim = config.get('num_primary_channels', 64) * self.num_classes
        self.transformer_feature_dim = self.hidden_dim
        
        # Add projection layers to match dimensions
        self.capsule_projection = nn.Linear(self.capsule_feature_dim, self.hidden_dim)
        
        # Final output dimension will be hidden_dim * 2
        self.combined_feature_dim = self.hidden_dim * 2
        
        # Self attention for combining features
        self.self_attention = SelfAttention(
            input_dim=self.combined_feature_dim,
            attention_dim=config.get('attention_dim', self.combined_feature_dim),
            dropout=self.dropout
        )
        
        # Policy network
        self.policy_net = PolicyNetwork(
            input_dim=self.combined_feature_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout
        )
        
        # Reward network
        self.reward_net = RewardNetwork(
            input_dim=self.combined_feature_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
    def _extract_features(self, input_ids, attention_mask=None, positions=None):
        """Extract features from input using CapsuleNet and Transformer paths"""
        batch_size, seq_len = input_ids.size()
        
        # Get transformer features first
        transformer_features = self.transformer(input_ids, attention_mask)  # (batch_size, seq_len, hidden_dim)
        
        # Get capsule features
        capsule_features = self.capsule(input_ids, attention_mask)  # (batch_size, num_classes, num_primary_channels)
        
        # Flatten and project capsule features
        capsule_features = capsule_features.view(batch_size, -1)  # (batch_size, num_classes * num_primary_channels)
        capsule_features = self.capsule_projection(capsule_features)  # (batch_size, hidden_dim)
        
        # Expand capsule features to match sequence length
        capsule_features = capsule_features.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        # Concatenate features along the last dimension
        combined_features = torch.cat([capsule_features, transformer_features], dim=-1)  # (batch_size, seq_len, hidden_dim * 2)
        
        return combined_features, seq_len
    
    def forward(self, input_ids, attention_mask=None, positions=None, labels=None):
        """Forward pass through the model"""
        # Extract features
        features, seq_len = self._extract_features(input_ids, attention_mask, positions)
        
        # Apply self-attention
        attended_features, attention_weights = self.self_attention(features, attention_mask)
        
        # Global pooling to get sequence representation
        if attention_mask is not None:
            # Apply mask for proper averaging
            mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            masked_features = attended_features * mask
            seq_rep = masked_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # Simple mean pooling
            seq_rep = attended_features.mean(dim=1)
        
        # Get policy (classification) output
        logits, policy_probs = self.policy_net(seq_rep)
        
        # Get reward prediction
        predicted_reward = self.reward_net(seq_rep)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Cross entropy loss for policy network
            policy_loss = F.cross_entropy(logits, labels)
            
            # Calculate true reward (1 if prediction is correct, 0 otherwise)
            predictions = torch.argmax(logits, dim=-1)
            true_reward = (predictions == labels).float()
            
            # MSE loss for reward network
            reward_loss = F.mse_loss(predicted_reward, true_reward)
            
            # Combined loss
            loss = policy_loss + reward_loss
        
        return {
            'logits': logits,
            'policy_probs': policy_probs,
            'predicted_reward': predicted_reward,
            'loss': loss,
            'attention_weights': attention_weights
        }
    
    def predict(self, input_ids, attention_mask=None, positions=None):
        """Make a prediction"""
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(input_ids, attention_mask, positions)
            
            # Get predictions
            predictions = torch.argmax(outputs['logits'], dim=-1)
        
        return predictions
    
    def predict_with_rewards(self, input_ids, attention_mask=None, positions=None):
        """Make a prediction and return logits and predicted rewards"""
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(input_ids, attention_mask, positions)
            
            # Get logits and predicted rewards
            logits = outputs['logits']
            rewards = outputs['predicted_reward']
        
        return logits, rewards
