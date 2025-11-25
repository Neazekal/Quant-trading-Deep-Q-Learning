"""Actor Network (Policy) for PPO agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.torchagents.networks.encoder import Encoder


class ActorNetwork(nn.Module):
    """Actor Network: Encoder + Policy head with Softmax output.
    
    Outputs action probabilities for the discrete action space.
    
    Architecture:
        Encoder (Conv1d + FC layers) -> Linear(256, num_actions) -> Softmax
    """
    
    def __init__(self, num_features: int, window_size: int, num_actions: int = 3):
        """Initialize Actor Network.
        
        Args:
            num_features: Number of input features (channels)
            window_size: Length of time window (sequence length)
            num_actions: Number of discrete actions (default 3: SELL, BUY, HOLD)
        """
        super().__init__()
        
        self.num_features = num_features
        self.window_size = window_size
        self.num_actions = num_actions
        
        # Shared encoder backbone
        self.encoder = Encoder(num_features, window_size)
        
        # Policy head: outputs action logits
        self.policy_head = nn.Linear(256, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action probabilities.
        
        Args:
            x: Input state tensor of shape (Batch, Window_Size, Num_Features)
               or (Batch, Num_Features, Window_Size)
               
        Returns:
            Action probabilities of shape (Batch, num_actions), sums to 1
        """
        # Encode state
        features = self.encoder(x)
        
        # Get action logits
        logits = self.policy_head(features)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (useful for numerical stability in loss computation).
        
        Args:
            x: Input state tensor
            
        Returns:
            Action logits of shape (Batch, num_actions)
        """
        features = self.encoder(x)
        logits = self.policy_head(features)
        return logits
