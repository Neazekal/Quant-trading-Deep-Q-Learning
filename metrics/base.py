"""Base metric interface."""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Metric(ABC):
    """Abstract base class for trading metrics."""
    
    @abstractmethod
    def calculate(self, returns: Union[List[float], np.ndarray]) -> float:
        """Calculate the metric from a series of returns.
        
        Args:
            returns: Array of period returns (e.g., daily returns)
            
        Returns:
            Calculated metric value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name for display."""
        pass
