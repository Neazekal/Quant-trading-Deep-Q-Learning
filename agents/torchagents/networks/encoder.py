"""Shared encoder network for feature extraction."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional encoder for time-series feature extraction.
    
    Architecture:
        Conv1d(in_features, 32, kernel=3) -> GELU -> Flatten -> 
        Linear(256) -> GELU -> Linear(256) -> GELU
    
    Input shape: (Batch, Num_Features, Timeframe_Size)
    Output shape: (Batch, 256)
    """
    
    def __init__(self, num_features: int, window_size: int):
        """Initialize encoder.
        
        Args:
            num_features: Number of input features (channels)
            window_size: Length of time window (sequence length)
        """
        super().__init__()
        
        self.num_features = num_features
        self.window_size = window_size
        
        # Conv1d: (Batch, num_features, window_size) -> (Batch, 32, window_size)
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        
        # Calculate flattened size after conv
        self.flat_size = 32 * window_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (Batch, Num_Features, Window_Size)
               OR (Batch, Window_Size, Num_Features) - will be transposed
               
        Returns:
            Encoded features of shape (Batch, 256)
        """
        # Handle both input formats
        # Expected: (Batch, Num_Features, Window_Size)
        # Common:   (Batch, Window_Size, Num_Features)
        if x.shape[-1] == self.num_features and x.shape[-2] == self.window_size:
            # Input is (Batch, Window_Size, Num_Features), transpose
            x = x.transpose(-1, -2)
        
        # Conv layer
        x = self.conv1(x)
        x = self.activation(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        
        return x
