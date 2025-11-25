"""Base agent interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np


class Agent(ABC):
    """Abstract base class for trading agents."""
    
    @abstractmethod
    def get_action(self, state: np.ndarray, training: bool = False) -> int:
        """Select an action given the current state.
        
        Args:
            state: Current observation from environment
            training: Whether in training mode (enables exploration)
            
        Returns:
            Selected action (0=SELL, 1=BUY, 2=HOLD)
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        """Perform one training step.
        
        Args:
            batch: Dictionary containing training batch data
            
        Returns:
            Dictionary containing loss values and metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save agent weights to disk.
        
        Args:
            path: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load agent weights from disk.
        
        Args:
            path: Path to checkpoint file
        """
        pass
