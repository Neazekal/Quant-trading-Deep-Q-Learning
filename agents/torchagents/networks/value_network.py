"""Value Network (Critic) for PPO agent."""

import torch
import torch.nn as nn

from agents.torchagents.networks.encoder import Encoder


class ValueNetwork(nn.Module):
    """Value Network (Critic): Encoder + Value head.
    
    Outputs scalar state value V(s) for advantage estimation.
    
    Architecture:
        Encoder (Conv1d + FC layers) -> Linear(256, 1)
    """
    
    def __init__(self, num_features: int, window_size: int):
        """Initialize Value Network.
        
        Args:
            num_features: Number of input features (channels)
            window_size: Length of time window (sequence length)
        """
        super().__init__()
        
        self.num_features = num_features
        self.window_size = window_size
        
        # Shared encoder backbone
        self.encoder = Encoder(num_features, window_size)
        
        # Value head: outputs scalar state value
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value.
        
        Args:
            x: Input state tensor of shape (Batch, Window_Size, Num_Features)
               or (Batch, Num_Features, Window_Size)
               
        Returns:
            State values of shape (Batch,)
        """
        # Encode state
        features = self.encoder(x)
        
        # Get state value
        value = self.value_head(features)
        
        # Squeeze to remove last dimension: (Batch, 1) -> (Batch,)
        return value.squeeze(-1)
