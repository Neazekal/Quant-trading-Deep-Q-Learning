"""Maximum Drawdown metric."""

from typing import List, Union

import numpy as np

from metrics.base import Metric


class MaxDrawdown(Metric):
    """Maximum Drawdown (MDD) metric.
    
    Measures the largest peak-to-trough decline in portfolio value.
    Lower is better (0 = no drawdown).
    """
    
    @property
    def name(self) -> str:
        return "Max Drawdown"
    
    def calculate(self, returns: Union[List[float], np.ndarray]) -> float:
        """Calculate maximum drawdown from returns.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Maximum drawdown as a positive percentage (0 to 1)
        """
        returns = np.asarray(returns)
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative wealth (assuming starting value of 1)
        cumulative = np.cumprod(1 + returns)
        
        # Running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Drawdown at each point
        drawdowns = (running_max - cumulative) / running_max
        
        return float(np.max(drawdowns))
    
    def calculate_from_equity(self, equity_curve: Union[List[float], np.ndarray]) -> float:
        """Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Array of portfolio values over time
            
        Returns:
            Maximum drawdown as a positive percentage
        """
        equity = np.asarray(equity_curve)
        
        if len(equity) < 2:
            return 0.0
        
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / (running_max + 1e-10)
        
        return float(np.max(drawdowns))
