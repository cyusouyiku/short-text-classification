import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        logits = self.fc3(x)
        
        # Return logits and policy probabilities
        return logits, F.softmax(logits, dim=-1)
    
    def get_action(self, x, sample=True):
        # Get policy probabilities
        logits, policy = self.forward(x)
        
        # Sample action or take greedy action
        if sample:
            # Sample action according to probability distribution
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
        else:
            # Take greedy action
            action = torch.argmax(policy, dim=-1)
            
        # Calculate log probability of the action
        log_prob = F.log_softmax(logits, dim=-1)
        selected_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, selected_log_prob, policy
