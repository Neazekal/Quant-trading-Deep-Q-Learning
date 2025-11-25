# TraderNet-CRv2: Master Design Document & Roadmap
> **Purpose:** This document serves as the "Source of Truth" for the TraderNet-CRv2 project. It contains sufficient detail to reconstruct the system from scratch, even in an empty repository.
---
## 1. Project Overview
**System Name:** TraderNet-CRv2 (PyTorch Modernization)
**Objective:** Build a robust, safety-first Deep Reinforcement Learning (DRL) trading system for cryptocurrencies.
**Core Philosophy:** **"Defense in Depth"**. High-performance agents (PPO/DQN) are wrapped in multiple safety layers to prevent catastrophic losses (Maximum Drawdown).
---
## 2. System Architecture
The system consists of three main layers:
1.  **Environment Layer:** Simulates the market, handles data, and calculates rewards.
2.  **Agent Layer (The Brain):** PyTorch-based DRL agents (PPO, DQN) that make trading decisions.
3.  **Safety Layer (The Shield):** Rules and Supervisor agents that override risky actions.
---
## 3. Component Specifications (The Blueprints)
### 3.1. Data & Input
*   **Raw Data:** OHLCV (Open, High, Low, Close, Volume) + Google Trends data.
*   **Preprocessing:**
    *   Log Returns of prices and volume.
    *   Technical Indicators: MACD, RSI, Bollinger Bands, AROON, CCI, ADX.
    *   Normalization: MinMax Scaling (0, 1) or Standardization.
*   **State Representation (Input to Agents):**
    *   **Shape:** `(Batch_Size, Timeframe_Size, Num_Features)`
    *   **Timeframe_Size:** Typically `12` (looking back 12 steps).
    *   **Num_Features:** ~21 features (Log returns + Indicators).
### 3.2. Trading Environment Logic
*   **Action Space:** Discrete(3) -> `{0: SELL, 1: BUY, 2: HOLD}`.
*   **Reward Functions:**
    *   **Market Order:** Profit/Loss (PnL) based on price difference between entry and exit.
    *   **Smurf Reward:** Modified reward where `HOLD` gives a small positive reward to encourage safety.
*   **Transaction Costs:** Must simulate trading fees (e.g., 0.1%) to prevent spam-trading.
### 3.3. Neural Network Architectures (PyTorch)
All networks use a shared **Encoder** backbone.
#### **A. The Encoder (Feature Extractor)**
*   **Input:** `(Batch, Num_Features, Timeframe_Size)` (Note: PyTorch Conv1d expects channels first).
*   **Layers:**
    1.  **Conv1d:** Filters=32, Kernel=3, Stride=1, Padding=Same. Activation: GELU.
    2.  **Flatten:** Convert to 1D vector.
    3.  **Dense (FC):** 256 units. Activation: GELU.
    4.  **Dense (FC):** 256 units. Activation: GELU.
#### **B. DQN Agent Networks**
*   **Q-Network:**
    *   **Body:** Encoder.
    *   **Head:** `Linear(256, 3)` (Output: Q-values for Sell, Buy, Hold).
*   **Logic:**
    *   **Policy:** Epsilon-Greedy.
    *   **Loss:** MSE or Huber Loss between `Q(s, a)` and `Target_Q`.
    *   **Target Update:** Soft update (Polyak averaging).
#### **C. PPO Agent Networks**
*   **Actor Network (Policy):**
    *   **Body:** Encoder.
    *   **Head:** `Linear(256, 3)` -> `Softmax`. (Output: Probability distribution).
*   **Critic Network (Value):**
    *   **Body:** Encoder.
    *   **Head:** `Linear(256, 1)`. (Output: State Value `V(s)`).
*   **Logic:**
    *   **Policy:** Stochastic sampling.
    *   **Loss:** PPO Clipped Surrogate Objective + Value Loss + Entropy Bonus.
### 3.4. Safety Mechanisms
#### **A. N-Consecutive Rule**
*   **Logic:** Prevent the agent from repeating the same action `N` times in a row.
*   **Implementation:**
    *   Maintain a queue of last `N` actions.
    *   If `Action_New == Action_{N-1} == ... == Action_{0}` (all same), force `Action = HOLD`.
    *   **Default N:** 2.
#### **B. Smurfing (Supervisor)**
*   **Logic:** A secondary "Smurf" agent validates the primary agent's action.
*   **Rule:**
    *   If `Smurf_Action == HOLD`: Force `Final_Action = HOLD`.
    *   Else: `Final_Action = Primary_Agent_Action`.
---
## 4. Implementation Roadmap
### Phase 1: PyTorch Migration (The Foundation)
**Goal:** Rebuild the Agent Layer using PyTorch.
1.  **Setup Directory:**
    *   `agents/torchagents/`
    *   `agents/torchagents/networks/`
2.  **Implement Networks:**
    *   Create `EncodingNetwork` (Conv1d + FCs).
    *   Create `QNetwork` (Encoder + Q-Head).
    *   Create `ActorNetwork` & `ValueNetwork`.
3.  **Implement Agents:**
    *   `DQNAgent`: Implement `get_action`, `train_step`, `save`, `load`.
    *   `PPOAgent`: Implement `get_action`, `train_step` (with GAE calculation), `save`, `load`.
4.  **Training Loop:**
    *   Create `train_torch.ipynb`.
    *   Implement a custom training loop: `Env Step -> Store in Buffer -> Sample Batch -> Agent Train`.
### Phase 2: Safety Integration
**Goal:** Re-implement the Safety Layer.
1.  **Port Rules:** Convert `rules/nconsecutive.py` to work with the new PyTorch agents.
2.  **Smurf Integration:**
    *   Train a separate `DQNAgent` as the Smurf (using `SmurfReward`).
    *   Create a `SmurfWrapper` or `IntegratedAgent` class that holds both the Main Agent and Smurf Agent to execute the overriding logic.
### Phase 3: Live Trading Infrastructure
**Goal:** Connect to the real world.
1.  **Exchange Interface:**
    *   Install `ccxt`.
    *   Create `ExchangeConnector` class to handle API keys and Order execution.
2.  **Data Streamer:**
    *   Create `WebSocketClient` to fetch live OHLCV.
    *   Implement `RealTimeFeatureEng` to calculate indicators on the fly (must match training data preprocessing exactly).
3.  **Bot Runner:**
    *   Create `main.py` loop:
        *   `while True:`
        *   `data = stream.get_latest()`
        *   `action = agent.get_action(data)`
        *   `safe_action = safety_layer.check(action)`
        *   `exchange.execute(safe_action)`
        *   `sleep(timeframe)`
---
## 5. Operational Guide
### How to Train
1.  Prepare data: `python download_datasets.py` (if not exists).
2.  Run training: Open `train_torch.ipynb`, select Agent (PPO/DQN), and run cells.
3.  Checkpoints: Models are saved in `database/storage/checkpoints/`.
### How to Evaluate
1.  Load model from checkpoint.
2.  Run evaluation loop on Test Set.
3.  Check metrics: Cumulative Return, Sharpe Ratio, **Maximum Drawdown**.