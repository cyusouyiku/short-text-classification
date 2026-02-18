"""
RLHNN: Reinforcement Learning-Enhanced Hybrid Neural Network
Integrates RoBERTa+Word2Vec+Positional embeddings, CapsuleNet, MultiscaleTransformer, and Actor-Critic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding_layer import UnifiedEmbeddingLayer
from .capsule_net import CapsuleNet
from .multiscale_transformer import MultiscaleTransformer
from .self_attention import SelfAttention, MultiHeadAttention
from .policy_net import PolicyNetwork
from .reward_net import RewardNetwork


class TextClassificationModel(nn.Module):
    def __init__(self, config):
        super(TextClassificationModel, self).__init__()
        
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config['num_classes']
        self.dropout = config.get('dropout', 0.3)
        self.lambda_value = config.get('lambda_value', 1.0)
        
        self.embedding_layer = UnifiedEmbeddingLayer(
            embed_dim=self.embedding_dim,
            max_len=config.get('max_seq_len', 128),
            dropout=self.dropout,
            roberta_model=config.get('roberta_model', 'hfl/chinese-roberta-wwm-ext'),
            static_embedding_matrix=config.get('static_embedding_matrix', None),
            freeze_roberta=config.get('freeze_roberta', False)
        )
        
        self.capsule = CapsuleNet(
            embedding_dim=self.embedding_dim,
            num_primary_units=config.get('num_primary_units', 8),
            num_primary_channels=config.get('num_primary_channels', 64),
            num_classes=self.num_classes,
            num_routing=config.get('num_routing', 3),
            dropout=self.dropout
        )
        
        self.transformer = MultiscaleTransformer(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            dropout=self.dropout,
            max_len=config.get('max_seq_len', 512)
        )
        
        self.capsule_feature_dim = config.get('num_primary_channels', 64) * self.num_classes
        self.capsule_projection = nn.Linear(self.capsule_feature_dim, self.hidden_dim)
        self.combined_feature_dim = self.hidden_dim * 2
        
        self.self_attention = MultiHeadAttention(
            input_dim=self.combined_feature_dim,
            num_heads=config.get('fusion_num_heads', 8),
            head_dim=None,
            dropout=self.dropout
        )
        
        self.policy_net = PolicyNetwork(
            input_dim=self.combined_feature_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout
        )
        
        self.reward_net = RewardNetwork(
            input_dim=self.combined_feature_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
    
    def _extract_features(self, input_ids, attention_mask=None):
        embeddings = self.embedding_layer(input_ids, attention_mask)
        batch_size, seq_len, _ = embeddings.size()
        
        transformer_features = self.transformer(embeddings, attention_mask)
        capsule_features = self.capsule(embeddings, attention_mask)
        
        capsule_features = capsule_features.view(batch_size, -1)
        capsule_features = self.capsule_projection(capsule_features)
        capsule_features = capsule_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined_features = torch.cat([capsule_features, transformer_features], dim=-1)
        return combined_features, seq_len
    
    def forward(self, input_ids, attention_mask=None, positions=None, labels=None):
        features, seq_len = self._extract_features(input_ids, attention_mask)
        attended_features, attention_weights = self.self_attention(features, attention_mask)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            masked_features = attended_features * mask
            seq_rep = masked_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            seq_rep = attended_features.mean(dim=1)
        
        logits, policy_probs = self.policy_net(seq_rep)
        predicted_reward = self.reward_net(seq_rep)
        
        loss = None
        if labels is not None:
            dist = torch.distributions.Categorical(probs=policy_probs)
            sampled_action = dist.sample()
            true_reward = (sampled_action == labels).float()
            advantage = true_reward - predicted_reward.squeeze(-1)
            log_prob = dist.log_prob(sampled_action)
            policy_loss = -(log_prob * advantage.detach()).mean()
            value_loss = F.mse_loss(predicted_reward.squeeze(-1), true_reward)
            loss = policy_loss + self.lambda_value * value_loss
        
        return {
            'logits': logits,
            'policy_probs': policy_probs,
            'predicted_reward': predicted_reward,
            'loss': loss,
            'attention_weights': attention_weights
        }
    
    def predict(self, input_ids, attention_mask=None, positions=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, positions)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions
    
    def predict_reward(self, input_ids, attention_mask=None, positions=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, positions)
        return outputs['predicted_reward']
    
    def predict_with_rewards(self, input_ids, attention_mask=None, positions=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, positions)
        return outputs['logits'], outputs['predicted_reward']
