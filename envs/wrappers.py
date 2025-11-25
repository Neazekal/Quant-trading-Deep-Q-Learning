"""Environment Wrappers for DeepRL-Trading-System."""

import gymnasium as gym
from envs.crypto_env import Actions

class SmurfRewardWrapper(gym.Wrapper):
    """Wrapper to modify rewards for Smurf Agent training.
    
    The Smurf Agent is a supervisor that learns to be "safe".
    It is rewarded for HOLDING and penalized less for avoiding losses.
    
    Logic:
        - If Action is HOLD: Reward = +0.001 (Incentivize safety)
        - If Action is BUY/SELL: Reward = Original PnL (Real consequences)
    """
    
    def __init__(self, env: gym.Env, hold_reward: float = 0.001):
        """Initialize SmurfRewardWrapper.
        
        Args:
            env: The environment to wrap.
            hold_reward: The positive reward given for HOLD actions.
        """
        super().__init__(env)
        self.hold_reward = hold_reward
        
    def step(self, action):
        """Step the environment with modified reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Modify reward
        if action == Actions.HOLD:
            # Overwrite PnL reward (which is usually 0 for HOLD) with incentive
            reward = self.hold_reward
            
        return obs, reward, terminated, truncated, info
