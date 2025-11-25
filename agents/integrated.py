"""Integrated Agent for DeepRL-Trading-System.

This agent combines:
1. Primary Agent (PPO/DQN): The main profit seeker.
2. Smurf Agent (DQN): The supervisor that can veto risky actions.
3. Safety Controller: Hard-coded rules (N-Consecutive) to prevent spamming.
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch

from agents.base import Agent
from agents.torchagents.ppo import PPOAgent
from agents.torchagents.dqn import DQNAgent
from rules.safety import SafetyController, Actions

class IntegratedAgent:
    """A composite agent that wraps Primary, Smurf, and Safety layers."""
    
    def __init__(
        self,
        primary_agent: Union[PPOAgent, DQNAgent],
        smurf_agent: DQNAgent,
        safety_controller: SafetyController,
    ):
        """Initialize IntegratedAgent.
        
        Args:
            primary_agent: The main trading agent.
            smurf_agent: The supervisor agent.
            safety_controller: The rule-based safety controller.
        """
        self.primary = primary_agent
        self.smurf = smurf_agent
        self.safety = safety_controller
        
    def get_action(
        self,
        state: np.ndarray,
        training: bool = False
    ) -> Tuple[int, dict]:
        """Get the final safe action.
        
        Logic:
            1. Primary Agent proposes an action.
            2. Smurf Agent checks state. If Smurf says HOLD, override to HOLD.
            3. Safety Controller checks history. If unsafe, override to HOLD.
        
        Returns:
            final_action (int)
            info (dict): Debug info about who made the decision.
        """
        # 1. Get Primary Action
        # Note: We always use training=False for inference in integrated mode
        if isinstance(self.primary, PPOAgent):
            primary_action = self.primary.get_action(state, training=False)
        else:
            primary_action = self.primary.get_action(state, training=False)
            
        # 2. Get Smurf Action
        smurf_action = self.smurf.get_action(state, training=False)
        
        # 3. Integration Logic
        final_action = primary_action
        decision_source = "PRIMARY"
        
        # Smurf Override: If Smurf wants to HOLD, we force HOLD
        # (Assuming Smurf is trained to be cautious)
        if smurf_action == Actions.HOLD and primary_action != Actions.HOLD:
            final_action = Actions.HOLD
            decision_source = "SMURF_OVERRIDE"
            
        # 4. Safety Rule Check
        # We check the *proposed* final action against history
        safety_override = self.safety.check(final_action)
        
        if safety_override is not None:
            final_action = safety_override
            decision_source = "SAFETY_RULE_OVERRIDE"
            
        return final_action, {
            "primary_action": primary_action,
            "smurf_action": smurf_action,
            "source": decision_source
        }
    
    def reset(self):
        """Reset internal states (safety controller history)."""
        self.safety.reset()
