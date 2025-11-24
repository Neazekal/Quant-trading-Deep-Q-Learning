# Quant-trading-Deep-Q-Learning

## Quick start (futures environment scaffold)

1. Install basics (PyTorch, Gymnasium, NumPy, Pandas):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch gymnasium numpy pandas
   ```
2. Load some OHLCV data (numpy array `[T, 6]` with columns `[ts, open, high, low, close, volume]`) and spin up the env:
   ```python
   import numpy as np
   from envs.crypto_env import BinanceTradingEnv, EnvConfig, Actions

   ohlcv = np.load("data/btcusdt_1h.npy")  # placeholder path
   env = BinanceTradingEnv(ohlcv, EnvConfig(window_size=50, leverage=5.0))
   obs, _ = env.reset()
   obs, reward, done, _, info = env.step(Actions.OPEN_LONG)
   ```
3. Wire this env into your DQN training loop (e.g., with a replay buffer and target network).

## TODO
- Data loader for Binance USDT-M futures (REST or websocket capture, incl. funding).
- Training script with Dueling Double DQN + prioritized replay.
- Evaluation harness with equity curve, Sharpe, drawdown, turnover, fee breakdown.

## Utilities
- `scripts/download_ohlcv.py`: download historical klines to `data/`.
  Example:
  ```bash
  python scripts/download_ohlcv.py --symbol BTCUSDT --interval 1h \
         --start "2024-01-01" --end "2024-06-01"
  # Add --spot to use spot data; default is USDT-M futures.
  ```
