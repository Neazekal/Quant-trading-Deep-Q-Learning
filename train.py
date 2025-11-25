"""Training script for DRL trading agents.

Usage:
    python train.py --config config.yaml --agent ppo
    python train.py --config config.yaml --agent dqn
    python train.py --config config.yaml --agent ppo --resume checkpoints/ppo_best.pt
"""

from __future__ import annotations

import argparse
import random
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from agents.torchagents import DQNAgent, PPOAgent
from agents.replay_buffer import ReplayBuffer, RolloutBuffer
from envs.crypto_env import CryptoTradingEnv, EnvConfig
from metrics import CumulativeReturn, MaxDrawdown, SharpeRatio
from envs.wrappers import SmurfRewardWrapper
from agents.integrated import IntegratedAgent
from rules.safety import SafetyController


def setup_logging(save_dir: Path, agent_type: str, mode: str):
    """Configure logging to file and console."""
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{agent_type}_{mode}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Saving to {log_file}")



def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device from config string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def create_env(df: pd.DataFrame, config: dict) -> CryptoTradingEnv:
    """Create trading environment from config."""
    env_config = EnvConfig(
        window_size=config["data"]["window_size"],
        leverage=config["env"]["leverage"],
        fee_rate=config["env"]["fee_rate"],
        initial_balance=config["env"]["initial_balance"],
        max_position_size=config["env"]["max_position_size"],
    )
    return CryptoTradingEnv(df, env_config)


def create_agent(
    agent_type: str,
    num_features: int,
    window_size: int,
    config: dict,
    device: torch.device,
) -> Union[DQNAgent, PPOAgent]:
    """Create agent based on type and config."""
    if agent_type == "ppo":
        return PPOAgent(
            num_features=num_features,
            window_size=window_size,
            num_actions=3,
            learning_rate=config["training"]["learning_rate"],
            gamma=config["training"]["gamma"],
            gae_lambda=config["ppo"]["gae_lambda"],
            clip_ratio=config["ppo"]["clip_ratio"],
            entropy_coef=config["ppo"]["entropy_coef"],
            value_coef=config["ppo"]["value_coef"],
            max_grad_norm=config["ppo"]["max_grad_norm"],
            device=str(device),
        )
    elif agent_type == "dqn":
        return DQNAgent(
            num_features=num_features,
            window_size=window_size,
            num_actions=3,
            learning_rate=config["training"]["learning_rate"],
            gamma=config["training"]["gamma"],
            tau=config["dqn"]["tau"],
            epsilon_start=config["dqn"]["epsilon_start"],
            epsilon_end=config["dqn"]["epsilon_end"],
            epsilon_decay=config["dqn"]["epsilon_decay"],
            device=str(device),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_agent(
    agent: Union[DQNAgent, PPOAgent],
    env: CryptoTradingEnv,
    num_episodes: int = 1,
) -> Dict[str, float]:
    """Evaluate agent on environment.
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    total_pnls = []
    episode_returns = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        step_returns = []
        
        done = False
        while not done:
            action = agent.get_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Track step-by-step returns for metrics
            if reward != 0:
                step_returns.append(reward / (info.get("balance", 10000) + 1e-10))
        
        total_rewards.append(episode_reward)
        total_pnls.append(info.get("total_pnl", 0))
        episode_returns.extend(step_returns if step_returns else [0.0])
    
    # Calculate metrics
    returns_array = np.array(episode_returns) if episode_returns else np.array([0.0])
    
    cum_return = CumulativeReturn()
    max_dd = MaxDrawdown()
    sharpe = SharpeRatio(periods_per_year=8760)  # Hourly data
    
    return {
        "mean_reward": float(np.mean(total_rewards)),
        "mean_pnl": float(np.mean(total_pnls)),
        "cumulative_return": cum_return.calculate(returns_array),
        "max_drawdown": max_dd.calculate(returns_array),
        "sharpe_ratio": sharpe.calculate(returns_array),
    }


def train_ppo(
    agent: PPOAgent,
    train_env: CryptoTradingEnv,
    val_env: Optional[CryptoTradingEnv],
    config: dict,
    save_dir: Path,
) -> None:
    """Train PPO agent.
    
    PPO collects trajectories, then performs multiple epochs of updates.
    """
    total_timesteps = config["training"]["total_timesteps"]
    n_steps = config["ppo"]["n_steps"]
    update_epochs = config["ppo"]["update_epochs"]
    batch_size = config["training"]["batch_size"]
    log_freq = config["training"]["log_freq"]
    eval_freq = config["training"]["eval_freq"]
    save_freq = config["training"]["save_freq"]
    
    # Initialize tracking
    global_step = 0
    episode_count = 0
    best_eval_reward = float("-inf")
    
    # Rollout buffer
    buffer = RolloutBuffer()
    
    # Reset environment
    obs, info = train_env.reset()
    episode_reward = 0.0
    
    # Metrics tracking
    recent_rewards: List[float] = []
    recent_losses: List[float] = []
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting PPO Training")
    logging.info(f"Total Timesteps: {total_timesteps:,}")
    logging.info(f"N-Steps per Update: {n_steps}")
    logging.info(f"Update Epochs: {update_epochs}")
    logging.info(f"{'='*60}\n")
    
    pbar = tqdm(total=total_timesteps, desc="Training", unit="step")
    
    while global_step < total_timesteps:
        # =====================================================================
        # Collection Phase: Collect n_steps of experience
        # =====================================================================
        buffer.clear()
        
        for _ in range(n_steps):
            # Get action with training info
            action, log_prob, value = agent.get_action(obs, training=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.push(obs, action, reward, value, log_prob, done)
            
            # Update tracking
            obs = next_obs
            episode_reward += reward
            global_step += 1
            pbar.update(1)
            
            # Handle episode end
            if done:
                recent_rewards.append(episode_reward)
                episode_count += 1
                
                # Reset for new episode
                obs, info = train_env.reset()
                episode_reward = 0.0
            
            if global_step >= total_timesteps:
                break
        
        # =====================================================================
        # Update Phase: Train on collected experience
        # =====================================================================
        
        # Get rollout data
        rollout = buffer.get()
        
        # Compute returns and advantages using GAE
        with torch.no_grad():
            # Get value of last state for bootstrapping
            last_obs = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            next_value = agent.critic(last_obs).item()
        
        advantages, returns = agent.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            next_value,
        )
        
        # Prepare batch data
        batch_data = {
            "states": rollout["states"],
            "actions": rollout["actions"],
            "old_log_probs": rollout["log_probs"],
            "returns": returns,
            "advantages": advantages,
        }
        
        # Multiple epochs of updates
        n_samples = len(rollout["states"])
        indices = np.arange(n_samples)
        
        metrics = {"loss": 0, "entropy": 0}
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                mini_batch = {
                    key: value[batch_indices] for key, value in batch_data.items()
                }
                
                metrics = agent.train_step(mini_batch)
                recent_losses.append(metrics["loss"])
        
        # =====================================================================
        # Logging
        # =====================================================================
        if global_step % log_freq < n_steps and recent_rewards:
            avg_reward = np.mean(recent_rewards[-100:]) if recent_rewards else 0
            avg_loss = np.mean(recent_losses[-100:]) if recent_losses else 0
            
            pbar.set_postfix({
                "ep": episode_count,
                "avg_rew": f"{avg_reward:.2f}",
                "loss": f"{avg_loss:.4f}",
            })
        
        # =====================================================================
        # Evaluation
        # =====================================================================
        if val_env is not None and global_step % eval_freq < n_steps:
            eval_metrics = evaluate_agent(agent, val_env, num_episodes=1)
            
            logging.info(f"\n[Eval @ Step {global_step:,}]")
            logging.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
            logging.info(f"  Mean PnL: ${eval_metrics['mean_pnl']:.2f}")
            logging.info(f"  Sharpe Ratio: {eval_metrics['sharpe_ratio']:.3f}")
            logging.info(f"  Max Drawdown: {eval_metrics['max_drawdown']:.2%}")
            
            # Save best model
            if config["training"]["save_best"] and eval_metrics["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["mean_reward"]
                agent.save(save_dir / "ppo_best.pt")
                logging.info(f"  New best model saved!")
        
        # =====================================================================
        # Checkpointing
        # =====================================================================
        if global_step % save_freq < n_steps:
            agent.save(save_dir / f"ppo_step_{global_step}.pt")
    
    pbar.close()
    
    # Final save
    agent.save(save_dir / "ppo_final.pt")
    logging.info(f"\nTraining complete! Final model saved to {save_dir / 'ppo_final.pt'}")


def train_dqn(
    agent: DQNAgent,
    train_env: CryptoTradingEnv,
    val_env: Optional[CryptoTradingEnv],
    config: dict,
    save_dir: Path,
) -> None:
    """Train DQN agent.
    
    DQN uses experience replay and target networks.
    """
    total_timesteps = config["training"]["total_timesteps"]
    batch_size = config["training"]["batch_size"]
    buffer_size = config["dqn"]["buffer_size"]
    learning_starts = config["dqn"]["learning_starts"]
    train_freq = config["dqn"]["train_freq"]
    log_freq = config["training"]["log_freq"]
    eval_freq = config["training"]["eval_freq"]
    save_freq = config["training"]["save_freq"]
    
    # Initialize tracking
    global_step = 0
    episode_count = 0
    best_eval_reward = float("-inf")
    
    # Replay buffer
    buffer = ReplayBuffer(buffer_size)
    
    # Reset environment
    obs, info = train_env.reset()
    episode_reward = 0.0
    
    # Metrics tracking
    recent_rewards: List[float] = []
    recent_losses: List[float] = []
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting DQN Training")
    logging.info(f"Total Timesteps: {total_timesteps:,}")
    logging.info(f"Buffer Size: {buffer_size:,}")
    logging.info(f"Learning Starts: {learning_starts:,}")
    logging.info(f"{'='*60}\n")
    
    pbar = tqdm(total=total_timesteps, desc="Training", unit="step")
    
    while global_step < total_timesteps:
        # =====================================================================
        # Collection Phase: Step once and store
        # =====================================================================
        
        # Get action (with exploration during training)
        action = agent.get_action(obs, training=True)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        
        # Store transition
        buffer.push(obs, action, reward, next_obs, done)
        
        # Update tracking
        obs = next_obs
        episode_reward += reward
        global_step += 1
        pbar.update(1)
        
        # Handle episode end
        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            
            # Decay epsilon at end of each episode
            agent.decay_epsilon()
            
            # Reset for new episode
            obs, info = train_env.reset()
            episode_reward = 0.0
        
        # =====================================================================
        # Update Phase: Train on sampled batch
        # =====================================================================
        if global_step >= learning_starts and global_step % train_freq == 0:
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                metrics = agent.train_step(batch)
                recent_losses.append(metrics["loss"])
        
        # =====================================================================
        # Logging
        # =====================================================================
        if global_step % log_freq == 0 and recent_rewards:
            avg_reward = np.mean(recent_rewards[-100:]) if recent_rewards else 0
            avg_loss = np.mean(recent_losses[-100:]) if recent_losses else 0
            
            pbar.set_postfix({
                "ep": episode_count,
                "avg_rew": f"{avg_reward:.2f}",
                "loss": f"{avg_loss:.4f}",
                "eps": f"{agent.epsilon:.3f}",
            })
        
        # =====================================================================
        # Evaluation
        # =====================================================================
        if val_env is not None and global_step % eval_freq == 0 and global_step > 0:
            eval_metrics = evaluate_agent(agent, val_env, num_episodes=1)
            
            logging.info(f"\n[Eval @ Step {global_step:,}]")
            logging.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
            logging.info(f"  Mean PnL: ${eval_metrics['mean_pnl']:.2f}")
            logging.info(f"  Sharpe Ratio: {eval_metrics['sharpe_ratio']:.3f}")
            logging.info(f"  Max Drawdown: {eval_metrics['max_drawdown']:.2%}")
            logging.info(f"  Epsilon: {agent.epsilon:.4f}")
            
            # Save best model
            if config["training"]["save_best"] and eval_metrics["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["mean_reward"]
                agent.save(save_dir / "dqn_best.pt")
                logging.info(f"  New best model saved!")
        
        # =====================================================================
        # Checkpointing
        # =====================================================================
        if global_step % save_freq == 0 and global_step > 0:
            agent.save(save_dir / f"dqn_step_{global_step}.pt")
    
    pbar.close()
    
    # Final save
    agent.save(save_dir / "dqn_final.pt")
    logging.info(f"\nTraining complete! Final model saved to {save_dir / 'dqn_final.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train DRL trading agent")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        choices=["ppo", "dqn"],
        required=True,
        help="Agent type to train",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["train", "smurf", "evaluate_integrated"],
        default="train",
        help="Operation mode: train (normal), smurf (train supervisor), evaluate_integrated (test safety)",
    )
    parser.add_argument("--primary_path", type=str, help="Path to primary agent checkpoint (for integrated eval)")
    parser.add_argument("--smurf_path", type=str, help="Path to smurf agent checkpoint (for integrated eval)")
    args = parser.parse_args()
    
    # Load config
    logging.info(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Get device
    device = get_device(config["system"]["device"])
    logging.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(config["system"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(save_dir, args.agent, args.mode)
    
    # Save config copy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(save_dir / f"config_{timestamp}.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Load data
    logging.info(f"Loading training data from {config['data']['train_path']}...")
    train_df = pd.read_csv(config["data"]["train_path"])
    logging.info(f"Loaded {len(train_df)} rows")
    
    # Split data for validation
    val_df = None
    if config["data"].get("val_path"):
        logging.info(f"Loading validation data from {config['data']['val_path']}...")
        val_df = pd.read_csv(config["data"]["val_path"])
    elif config["data"].get("val_split", 0) > 0:
        split_idx = int(len(train_df) * (1 - config["data"]["val_split"]))
        val_df = train_df.iloc[split_idx:].reset_index(drop=True)
        train_df = train_df.iloc[:split_idx].reset_index(drop=True)
        logging.info(f"Split: {len(train_df)} train, {len(val_df)} val rows")
    
    # Create environments
    logging.info("Creating environments...")
    train_env = create_env(train_df, config)
    val_env = create_env(val_df, config) if val_df is not None else None
    
    # Get feature dimensions from environment
    num_features = train_env.num_features
    window_size = config["data"]["window_size"]
    logging.info(f"State shape: ({window_size}, {num_features})")
    
    # Create agent
    logging.info(f"Creating {args.agent.upper()} agent...")
    agent = create_agent(args.agent, num_features, window_size, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        logging.info(f"Loading checkpoint from {args.resume}...")
        agent.load(args.resume)
    
    # Train
    # Train
    if args.mode == "train":
        if args.agent == "ppo":
            train_ppo(agent, train_env, val_env, config, save_dir)
        else:
            train_dqn(agent, train_env, val_env, config, save_dir)
            
    elif args.mode == "smurf":
        logging.info("Training in SMURF mode (Incentivizing HOLD)...")
        # Wrap environment with Smurf Reward
        train_env = SmurfRewardWrapper(train_env, hold_reward=0.001)
        if val_env:
            val_env = SmurfRewardWrapper(val_env, hold_reward=0.001)
            
        # Smurf must be DQN (usually)
        if args.agent != "dqn":
            logging.warning("WARNING: Smurf is typically a DQN agent. Proceeding anyway.")
            
        train_dqn(agent, train_env, val_env, config, save_dir)
        
    elif args.mode == "evaluate_integrated":
        logging.info("Evaluating INTEGRATED System...")
        if not args.primary_path or not args.smurf_path:
            raise ValueError("Must provide --primary_path and --smurf_path for integrated evaluation")
            
        # Load Primary Agent
        logging.info(f"Loading Primary Agent from {args.primary_path}...")
        # We need to create a fresh agent instance to load weights into
        # Assuming Primary is PPO for now (can be made configurable)
        primary_agent = PPOAgent(num_features, window_size, 3, device=str(device))
        primary_agent.load(args.primary_path)
        
        # Load Smurf Agent
        logging.info(f"Loading Smurf Agent from {args.smurf_path}...")
        smurf_agent = DQNAgent(num_features, window_size, 3, device=str(device))
        smurf_agent.load(args.smurf_path)
        
        # Create Safety Controller
        safety = SafetyController(n_consecutive=2)
        
        # Create Integrated Agent
        integrated_agent = IntegratedAgent(primary_agent, smurf_agent, safety)
        
        # Evaluate
        # Use validation env for evaluation
        eval_env = val_env if val_env else train_env
        
        metrics = evaluate_agent(integrated_agent, eval_env, num_episodes=5)
        
        logging.info(f"\n[Integrated Evaluation Results]")
        logging.info(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        logging.info(f"  Mean PnL: ${metrics['mean_pnl']:.2f}")
        logging.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logging.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")


import gc

def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("GPU memory cleaned.")

if __name__ == "__main__":
    try:
        cleanup_gpu()  # Clean before start
        main()
    finally:
        cleanup_gpu()  # Clean after finish (even on error)
