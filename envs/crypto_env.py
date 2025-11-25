"""Gymnasium-compatible trading environment for cryptocurrency futures.

This environment expects PRE-PROCESSED data with features already computed.
Use data_processor.py to generate features before loading data into this env.

Usage:
    import pandas as pd
    from envs.crypto_env import CryptoTradingEnv, EnvConfig, Actions

    # Load pre-processed data (with all features)
    df = pd.read_csv("data/DOGEUSDT_1h_processed.csv")
    env = CryptoTradingEnv(df, EnvConfig(window_size=12, leverage=1.0, fee_rate=0.001))
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(Actions.BUY)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class Actions(IntEnum):
    """Trading actions."""
    SELL = 0
    BUY = 1
    HOLD = 2


class Positions(IntEnum):
    """Position states."""
    SHORT = -1
    FLAT = 0
    LONG = 1


# Default feature columns (must match data_processor.py output)
DEFAULT_FEATURE_COLUMNS = [
    # Log Returns (6)
    "open_log_returns",
    "high_log_returns",
    "low_log_returns",
    "close_log_returns",
    "volume_log_returns",
    "trades_log_returns",
    # Time (1)
    "hour",
    # Technical Indicators (15)
    "macd_signal_diffs",
    "stoch",
    "aroon_up",
    "aroon_down",
    "rsi",
    "adx",
    "cci",
    "close_dema",
    "close_vwap",
    "bband_up_close",
    "close_bband_down",
    "adl_diffs2",
    "obv_diffs2",
    # Optional (1)
    "trends",
]


@dataclass
class EnvConfig:
    """Environment configuration."""
    window_size: int = 12              # Lookback window for observations
    leverage: float = 1.0              # Trading leverage (1.0 = no leverage)
    fee_rate: float = 0.001            # Transaction fee rate (0.1% default)
    initial_balance: float = 10000.0   # Starting capital in USDT
    max_position_size: float = 1.0     # Max position as fraction of balance
    feature_columns: List[str] = field(default_factory=lambda: DEFAULT_FEATURE_COLUMNS.copy())


class CryptoTradingEnv(gym.Env):
    """Cryptocurrency trading environment with PnL-based rewards.
    
    Expects pre-processed DataFrame with feature columns computed by data_processor.py.
    
    State: Rolling window of pre-computed features
    Actions: SELL (0), BUY (1), HOLD (2)
    Reward: Realized PnL after transaction costs
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV + pre-computed feature columns
            config: Environment configuration
            render_mode: Rendering mode (optional)
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Validate required columns
        required_cols = ["close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        self.df = df.reset_index(drop=True)
        self.prices = self.df["close"].values.astype(np.float32)
        
        # Extract feature columns
        self._setup_features()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size, self.num_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        
        # Trading state (initialized in reset)
        self._current_step: int = 0
        self._position: Positions = Positions.FLAT
        self._entry_price: float = 0.0
        self._balance: float = self.config.initial_balance
        self._total_pnl: float = 0.0
        self._trade_count: int = 0
        
    def _setup_features(self) -> None:
        """Extract feature columns from DataFrame."""
        # Filter to columns that exist in the DataFrame
        available_features = [
            col for col in self.config.feature_columns 
            if col in self.df.columns
        ]
        
        if not available_features:
            raise ValueError(
                f"No feature columns found in DataFrame. "
                f"Expected columns like: {self.config.feature_columns[:5]}... "
                f"Available columns: {list(self.df.columns)}"
            )
        
        self.feature_names = available_features
        self.num_features = len(available_features)
        
        # Extract and convert to numpy array
        self.features = self.df[available_features].values.astype(np.float32)
        
        # Handle NaN and infinity
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clip extreme values
        self.features = np.clip(self.features, -10.0, 10.0)
        
        print(f"Loaded {self.num_features} features: {self.feature_names}")
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation window."""
        start_idx = self._current_step - self.config.window_size + 1
        end_idx = self._current_step + 1
        
        if start_idx < 0:
            # Pad with zeros at the beginning
            pad_size = -start_idx
            obs = np.vstack([
                np.zeros((pad_size, self.num_features), dtype=np.float32),
                self.features[:end_idx]
            ])
        else:
            obs = self.features[start_idx:end_idx]
        
        return obs.astype(np.float32)
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """Calculate PnL for closing a position."""
        if self._position == Positions.FLAT:
            return 0.0
        
        price_diff = exit_price - self._entry_price
        if self._position == Positions.SHORT:
            price_diff = -price_diff
        
        pnl_pct = price_diff / self._entry_price
        leveraged_pnl_pct = pnl_pct * self.config.leverage
        fee_cost = 2 * self.config.fee_rate
        net_pnl_pct = leveraged_pnl_pct - fee_cost
        
        position_value = self._balance * self.config.max_position_size
        pnl = position_value * net_pnl_pct
        
        return pnl
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward."""
        current_price = self.prices[self._current_step]
        reward = 0.0
        
        if action == Actions.BUY:
            if self._position == Positions.SHORT:
                reward = self._calculate_pnl(current_price)
                self._balance += reward
                self._total_pnl += reward
                self._position = Positions.FLAT
                self._trade_count += 1
            
            if self._position == Positions.FLAT:
                self._position = Positions.LONG
                self._entry_price = current_price
                
        elif action == Actions.SELL:
            if self._position == Positions.LONG:
                reward = self._calculate_pnl(current_price)
                self._balance += reward
                self._total_pnl += reward
                self._position = Positions.FLAT
                self._trade_count += 1
            
            if self._position == Positions.FLAT:
                self._position = Positions.SHORT
                self._entry_price = current_price
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        reward = self._execute_action(action)
        self._current_step += 1
        
        terminated = self._current_step >= len(self.prices) - 1
        truncated = False
        
        # Force close at end
        if terminated and self._position != Positions.FLAT:
            final_price = self.prices[self._current_step]
            final_pnl = self._calculate_pnl(final_price)
            reward += final_pnl
            self._balance += final_pnl
            self._total_pnl += final_pnl
            self._position = Positions.FLAT
            self._trade_count += 1
        
        # Bankruptcy check
        if self._balance <= 0:
            terminated = True
            reward = -1.0
        
        obs = self._get_observation()
        
        info = {
            "balance": self._balance,
            "total_pnl": self._total_pnl,
            "position": int(self._position),
            "entry_price": self._entry_price,
            "current_price": self.prices[self._current_step],
            "trade_count": self._trade_count,
            "step": self._current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self._balance = self.config.initial_balance
        self._total_pnl = 0.0
        self._position = Positions.FLAT
        self._entry_price = 0.0
        self._trade_count = 0
        
        if options and "start_step" in options:
            self._current_step = options["start_step"]
        else:
            self._current_step = self.config.window_size - 1
        
        obs = self._get_observation()
        info = {
            "balance": self._balance,
            "total_pnl": self._total_pnl,
            "position": int(self._position),
            "entry_price": self._entry_price,
            "current_price": self.prices[self._current_step],
            "trade_count": self._trade_count,
            "step": self._current_step,
        }
        
        return obs, info
    
    def render(self) -> None:
        """Render current state."""
        if self.render_mode == "human":
            pos_str = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[int(self._position)]
            print(
                f"Step {self._current_step:5d} | "
                f"Price: {self.prices[self._current_step]:10.6f} | "
                f"Position: {pos_str:5s} | "
                f"Balance: {self._balance:12.2f} | "
                f"PnL: {self._total_pnl:+10.2f}"
            )
    
    def close(self) -> None:
        """Clean up resources."""
        pass


# Convenience alias
BinanceTradingEnv = CryptoTradingEnv
