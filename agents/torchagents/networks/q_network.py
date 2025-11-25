"""Q-Network for DQN agent."""

import torch
import torch.nn as nn

from agents.torchagents.networks.encoder import Encoder


class QNetwork(nn.Module):
    """Q-Network: Encoder + Q-value head.
    
    Output: Q-values for each action (SELL, BUY, HOLD)
    """
    
    def __init__(self, num_features: int, window_size: int, num_actions: int = 3):
        """Initialize Q-Network.
        
        Args:
            num_features: Number of input features
            window_size: Length of time window
            num_actions: Number of discrete actions (default 3)
        """
        super().__init__()
        
        self.encoder = Encoder(num_features, window_size)
        self.q_head = nn.Linear(256, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action, shape (Batch, num_actions)
        """
        features = self.encoder(x)
        q_values = self.q_head(features)
        return q_values
