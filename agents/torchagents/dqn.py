"""Deep Q-Network (DQN) Agent."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import Agent
from agents.torchagents.networks.q_network import QNetwork


class DQNAgent(Agent):
    """Deep Q-Network agent with target network and epsilon-greedy exploration.
    
    Features:
        - Double DQN (uses online network to select, target to evaluate)
        - Soft target updates (Polyak averaging)
        - Epsilon-greedy exploration with decay
    """
    
    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_actions: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: Optional[str] = None,
    ):
        """Initialize DQN agent.
        
        Args:
            num_features: Number of input features
            window_size: Observation window size
            num_actions: Number of actions (default 3)
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay per episode
            device: Device to use (auto-detect if None)
        """
        self.num_features = num_features
        self.window_size = window_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = QNetwork(num_features, window_size, num_actions).to(self.device)
        self.target_network = QNetwork(num_features, window_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    
    def get_action(self, state: np.ndarray, training: bool = False) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current observation, shape (Window_Size, Num_Features)
            training: Enable exploration if True
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def train_step(self, batch: dict) -> dict:
        """Perform one training step.
        
        Args:
            batch: Dictionary with keys:
                - states: (Batch, Window_Size, Num_Features)
                - actions: (Batch,)
                - rewards: (Batch,)
                - next_states: (Batch, Window_Size, Num_Features)
                - dones: (Batch,)
                
        Returns:
            Dictionary with 'loss' key
        """
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        return {"loss": loss.item()}
    
    def _soft_update(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save agent checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
