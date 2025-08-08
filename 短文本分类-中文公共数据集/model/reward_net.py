import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(RewardNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output a single reward value
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, action=None):
        # x: (batch_size, input_dim)
        # action: (batch_size, )
        
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        reward = self.fc3(x).squeeze(-1)  # (batch_size, )
        
        return reward
