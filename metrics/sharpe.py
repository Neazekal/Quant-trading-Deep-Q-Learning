"""Sharpe Ratio metric."""

from typing import List, Union

import numpy as np

from metrics.base import Metric


class SharpeRatio(Metric):
    """Sharpe Ratio metric.
    
    Measures risk-adjusted return: (mean return - risk_free) / std(return).
    Higher is better.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, annualize: bool = True, periods_per_year: int = 8760):
        """Initialize Sharpe Ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0)
            annualize: Whether to annualize the ratio
            periods_per_year: Number of periods per year (8760 for hourly data)
        """
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize
        self.periods_per_year = periods_per_year
    
    @property
    def name(self) -> str:
        return "Sharpe Ratio"
    
    def calculate(self, returns: Union[List[float], np.ndarray]) -> float:
        """Calculate Sharpe Ratio from returns.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Sharpe ratio (can be negative)
        """
        returns = np.asarray(returns)
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return < 1e-10:
            return 0.0
        
        # Per-period risk-free rate
        rf_per_period = self.risk_free_rate / self.periods_per_year
        
        sharpe = (mean_return - rf_per_period) / std_return
        
        if self.annualize:
            sharpe *= np.sqrt(self.periods_per_year)
        
        return float(sharpe)
