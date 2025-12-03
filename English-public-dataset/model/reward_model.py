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
            num_classes=config.get('num_classes', 4),
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
        self.capsule_feature_dim = config.get('num_primary_channels', 64) * config.get('num_classes', 4)
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
            input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones((batch_size, seq_length))
            
            # Forward pass
            outputs = self(input_ids, attention_mask)
            
            # Create visualization using torchviz
            from torchviz import make_dot
            dot = make_dot(outputs['logits'], params=dict(self.named_parameters()),
                         show_attrs=True, show_saved=True)
            
            # Customize graph appearance
            dot.attr(rankdir='TB')  # Top to bottom layout
            dot.attr('node', shape='box', style='filled', fontname='Arial')
            
            # Add model summary to graph
            summary = f"""TextClassificationModel Architecture
            Vocab Size: {self.vocab_size}
            Embedding Dim: {self.embedding_dim}
            Hidden Dim: {self.hidden_dim}
            Num Classes: {self.num_classes}
            """
            dot.attr(label=summary)
            
            # Save visualization
            dot.render(save_path, format='png', cleanup=True)
            
            # Generate detailed text summary
            summary_path = f"{save_path}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("TextClassificationModel Architecture Summary\n")
                f.write("=====================================\n\n")
                
                # Model configuration
                f.write("Model Configuration:\n")
                f.write(f"- Vocabulary Size: {self.vocab_size}\n")
                f.write(f"- Embedding Dimension: {self.embedding_dim}\n")
                f.write(f"- Hidden Dimension: {self.hidden_dim}\n")
                f.write(f"- Number of Classes: {self.num_classes}\n")
                f.write(f"- Number of Transformer Heads: {self.transformer.num_heads}\n")
                f.write(f"- Number of Transformer Layers: {len(self.transformer.transformer_layers)}\n\n")
                
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
