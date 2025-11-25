"""Replay Buffer for off-policy algorithms (DQN)."""

from collections import deque
from typing import Dict
import random

import numpy as np


class ReplayBuffer:
    """Experience replay buffer for DQN training.
    
    Stores transitions (s, a, r, s', done) and provides random sampling.
    """
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with batched arrays
        """
        transitions = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms (PPO).
    
    Stores trajectories and computes returns/advantages.
    """
    
    def __init__(self):
        """Initialize rollout buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a step to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate V(s)
            log_prob: Log probability of action
            done: Whether episode terminated
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data as numpy arrays.
        
        Returns:
            Dictionary with arrays (without returns/advantages)
        """
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.states)
