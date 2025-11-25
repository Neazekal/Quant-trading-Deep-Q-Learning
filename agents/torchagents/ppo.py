"""Proximal Policy Optimization (PPO) Agent."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.base import Agent
from agents.torchagents.networks.actor_network import ActorNetwork
from agents.torchagents.networks.value_network import ValueNetwork


class PPOAgent(Agent):
    """Proximal Policy Optimization agent with clipped surrogate objective.
    
    Features:
        - Actor-Critic architecture with separate networks
        - Clipped surrogate objective for stable policy updates
        - Generalized Advantage Estimation (GAE)
        - Entropy bonus for exploration
    
    Reference:
        Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    """
    
    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_actions: int = 3,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        """Initialize PPO agent.
        
        Args:
            num_features: Number of input features
            window_size: Observation window size
            num_actions: Number of actions (default 3: SELL, BUY, HOLD)
            learning_rate: Optimizer learning rate
            gamma: Discount factor for rewards
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_ratio: PPO clipping parameter (epsilon)
            entropy_coef: Coefficient for entropy bonus
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use (auto-detect if None)
        """
        self.num_features = num_features
        self.window_size = window_size
        self.num_actions = num_actions
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(num_features, window_size, num_actions).to(self.device)
        self.critic = ValueNetwork(num_features, window_size).to(self.device)
        
        # Single optimizer for both networks (common in PPO implementations)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )
    
    def get_action(
        self,
        state: np.ndarray,
        training: bool = False,
    ) -> Union[int, Tuple[int, float, float]]:
        """Select action using the policy.
        
        Args:
            state: Current observation, shape (Window_Size, Num_Features)
            training: If True, return additional info for training
            
        Returns:
            If training=False: action (int)
            If training=True: (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities from actor
            probs = self.actor(state_tensor)
            
            # Create categorical distribution
            dist = Categorical(probs)
            
            # Sample action
            action = dist.sample()
            
            if not training:
                return int(action.item())
            
            # Get log probability and value for training
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
            
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given states.
        
        Used during training to compute new log probabilities and values.
        
        Args:
            states: Batch of states, shape (Batch, Window_Size, Num_Features)
            actions: Batch of actions, shape (Batch,)
            
        Returns:
            log_probs: Log probabilities of actions, shape (Batch,)
            values: State values, shape (Batch,)
            entropy: Policy entropy, shape (Batch,)
        """
        # Get action probabilities
        probs = self.actor(states)
        
        # Create distribution
        dist = Categorical(probs)
        
        # Compute log probabilities for taken actions
        log_probs = dist.log_prob(actions)
        
        # Compute entropy
        entropy = dist.entropy()
        
        # Get state values
        values = self.critic(states)
        
        return log_probs, values, entropy
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards, shape (T,)
            values: Array of state values, shape (T,)
            dones: Array of done flags, shape (T,)
            next_value: Value of the state after the last step
            
        Returns:
            advantages: GAE advantages, shape (T,)
            returns: Discounted returns (advantages + values), shape (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        
        # Compute GAE backwards
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # Mask for non-terminal states
            next_non_terminal = 1.0 - dones[t]
            
            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # GAE: delta + gamma * lambda * GAE(t+1)
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # Returns = advantages + values (for value function target)
        returns = advantages + values
        
        return advantages, returns
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform one PPO training step.
        
        Args:
            batch: Dictionary with keys:
                - states: (Batch, Window_Size, Num_Features)
                - actions: (Batch,)
                - old_log_probs: (Batch,) - log probs from collection policy
                - returns: (Batch,) - discounted returns (GAE targets)
                - advantages: (Batch,) - GAE advantages
                
        Returns:
            Dictionary with loss values and metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch["old_log_probs"]).to(self.device)
        returns = torch.FloatTensor(batch["returns"]).to(self.device)
        advantages = torch.FloatTensor(batch["advantages"]).to(self.device)
        
        # Normalize advantages (common practice for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Evaluate current policy
        new_log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # =====================================================================
        # PPO Clipped Surrogate Objective
        # =====================================================================
        
        # Probability ratio: exp(log_prob_new - log_prob_old)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        
        # Surrogate objectives
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        
        # Policy loss: negative because we want to maximize
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # =====================================================================
        # Value Loss (MSE)
        # =====================================================================
        value_loss = nn.functional.mse_loss(values, returns)
        
        # =====================================================================
        # Entropy Bonus (for exploration)
        # =====================================================================
        entropy_loss = -entropy.mean()  # Negative because we add it to total loss
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        
        self.optimizer.step()
        
        # Compute additional metrics
        with torch.no_grad():
            # Approximate KL divergence for monitoring
            approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
            
            # Clip fraction: how often the ratio was clipped
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
            )
        
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),  # Return positive entropy
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save agent checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load agent checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
