"""Cumulative return metric."""

from typing import List, Union

import numpy as np

from metrics.base import Metric


class CumulativeReturn(Metric):
    """Cumulative Return metric.
    
    Measures total return over the period.
    """
    
    @property
    def name(self) -> str:
        return "Cumulative Return"
    
    def calculate(self, returns: Union[List[float], np.ndarray]) -> float:
        """Calculate cumulative return from period returns.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Cumulative return as percentage (e.g., 0.5 = 50% gain)
        """
        returns = np.asarray(returns)
        
        if len(returns) == 0:
            return 0.0
        
        # Cumulative return = product of (1 + r) - 1
        cumulative = np.prod(1 + returns) - 1
        
        return float(cumulative)
