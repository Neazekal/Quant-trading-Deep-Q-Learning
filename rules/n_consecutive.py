"""N-Consecutive rule to prevent repeated actions."""

from collections import deque

from rules.base import Rule


class NConsecutiveRule(Rule):
    """Prevent the agent from repeating the same action N times consecutively.
    
    If the agent tries to execute the same action N times in a row,
    force a HOLD action instead.
    
    Example with N=2:
        Actions: BUY, BUY, BUY -> BUY, BUY, HOLD (third BUY blocked)
    """
    
    HOLD_ACTION = 2
    
    def __init__(self, n: int = 2):
        """Initialize N-Consecutive rule.
        
        Args:
            n: Maximum consecutive same actions allowed (default 2)
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        self.n = n
        self._action_history: deque = deque(maxlen=n)
    
    def filter(self, action: int) -> int:
        """Filter action if it would exceed N consecutive.
        
        Args:
            action: Proposed action
            
        Returns:
            Original action or HOLD if blocked
        """
        # Not enough history yet
        if len(self._action_history) < self.n:
            return action
        
        # Check if all recent actions are the same as proposed
        if all(a == action for a in self._action_history):
            return self.HOLD_ACTION
        
        return action
    
    def update(self, action: int) -> None:
        """Record the executed action.
        
        Args:
            action: Action that was executed
        """
        self._action_history.append(action)
    
    def reset(self) -> None:
        """Clear action history for new episode."""
        self._action_history.clear()
    
    @property
    def history(self) -> list:
        """Get current action history."""
        return list(self._action_history)
