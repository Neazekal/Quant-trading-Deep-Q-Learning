"""Base rule interface for action filtering."""

from abc import ABC, abstractmethod


class Rule(ABC):
    """Abstract base class for safety rules."""
    
    @abstractmethod
    def filter(self, action: int) -> int:
        """Filter an action based on the rule.
        
        Args:
            action: Proposed action (0=SELL, 1=BUY, 2=HOLD)
            
        Returns:
            Filtered action (may be changed to HOLD)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset rule state for new episode."""
        pass
    
    @abstractmethod
    def update(self, action: int) -> None:
        """Update internal state after action execution.
        
        Args:
            action: Action that was actually executed
        """
        pass
