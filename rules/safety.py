"""Safety Rules Module for DeepRL-Trading-System.

This module implements the SafetyController and various safety rules
to prevent the agent from taking risky actions.
"""

from collections import deque
from enum import IntEnum
from typing import Optional, Deque

class Actions(IntEnum):
    """Trading actions."""
    SELL = 0
    BUY = 1
    HOLD = 2

class SafetyController:
    """Controller to enforce safety rules on agent actions."""
    
    def __init__(self, n_consecutive: int = 2):
        """Initialize SafetyController.
        
        Args:
            n_consecutive: Max number of consecutive identical actions allowed.
        """
        self.n_consecutive = n_consecutive
        self.action_history: Deque[int] = deque(maxlen=n_consecutive + 1)
        
    def check(self, action: int) -> Optional[int]:
        """Check if the proposed action violates any safety rules.
        
        Args:
            action: The action proposed by the agent.
            
        Returns:
            SAFE_ACTION (int) if a violation is detected (usually HOLD).
            None if the action is safe to proceed.
        """
        # Update history with the proposed action
        self.action_history.append(action)
        
        # Check N-Consecutive Rule
        if self._check_n_consecutive():
            return int(Actions.HOLD)
            
        return None
    
    def _check_n_consecutive(self) -> bool:
        """Check if the last N+1 actions are identical.
        
        If we have N+1 identical actions, it means the agent has already done
        the action N times, and is trying to do it for the (N+1)-th time.
        We should block this (N+1)-th attempt.
        """
        if len(self.action_history) < self.n_consecutive + 1:
            return False
            
        # Check if all actions in history are the same
        first_action = self.action_history[0]
        
        # If the action is HOLD, we usually don't limit consecutive holds
        if first_action == Actions.HOLD:
            return False
            
        for act in self.action_history:
            if act != first_action:
                return False
                
        return True

    def reset(self):
        """Reset the safety controller state."""
        self.action_history.clear()
