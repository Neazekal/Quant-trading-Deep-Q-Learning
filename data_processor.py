"""Pre-calculate technical indicators and prepare features for training.

This script processes raw OHLCV data and generates all required features
BEFORE feeding into the Neural Network. Uses the `ta` library for indicators.

Usage:
    python data_processor.py --input data/DOGEUSDT_1h_20200101_20251125.csv \
                             --output data/DOGEUSDT_1h_20200101_20251125_processed.csv

Required Input Columns:
    open_time_dt, open, high, low, close, volume

Output Features:
    - Log Returns: open, high, low, close, volume, trades (if available)
    - Time: hour
    - Technical Indicators: MACD, Stochastic, Aroon, RSI, ADX, CCI, DEMA, VWAP,
                           Bollinger Bands signals, ADL diffs, OBV diffs
    - Optional: Google Trends (if available)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, AroonIndicator, CCIIndicator, MACD, DPOIndicator
from ta.volatility import BollingerBands
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def generate_features(
    df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate all required features from OHLCV data.
    
    Args:
        df: DataFrame with columns [open_time_dt, open, high, low, close, volume]
        trends_df: Optional DataFrame with Google Trends data (columns: date, trends)
    
    Returns:
        DataFrame with all computed features
    """
    df = df.copy()
    eps = 1e-10
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # =========================================================================
    # 1. LOG RETURNS
    # =========================================================================
    df["open_log_returns"] = np.log(df["open"] / df["open"].shift(1) + eps)
    df["high_log_returns"] = np.log(df["high"] / df["high"].shift(1) + eps)
    df["low_log_returns"] = np.log(df["low"] / df["low"].shift(1) + eps)
    df["close_log_returns"] = np.log(df["close"] / df["close"].shift(1) + eps)
    df["volume_log_returns"] = np.log(df["volume"] / df["volume"].shift(1) + eps)
    
    # Trades log returns (if column exists)
    if "trades" in df.columns:
        df["trades_log_returns"] = np.log(df["trades"] / df["trades"].shift(1) + eps)
    else:
        df["trades_log_returns"] = 0.0
    
    # =========================================================================
    # 2. TIME FEATURES
    # =========================================================================
    if "open_time_dt" in df.columns:
        df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
        df["hour"] = df["open_time_dt"].dt.hour
    else:
        df["hour"] = 0
    
    # =========================================================================
    # 3. TECHNICAL INDICATORS (using `ta` library)
    # =========================================================================
    
    # --- MACD Signal Differences (Histogram) ---
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_signal_diffs"] = macd.macd_diff()  # MACD Line - Signal Line
    
    # --- Stochastic Oscillator ---
    stoch = StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3
    )
    df["stoch"] = stoch.stoch()
    
    # --- Aroon Indicator ---
    aroon = AroonIndicator(high=df["high"], low=df["low"], window=25)
    df["aroon_up"] = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()
    
    # --- RSI (Relative Strength Index) ---
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()
    
    # --- ADX (Average Directional Index) ---
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    
    # --- CCI (Commodity Channel Index) ---
    cci = CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20)
    df["cci"] = cci.cci()
    
    # --- DEMA (Double Exponential Moving Average) ---
    # ta doesn't have DEMA directly, compute manually: DEMA = 2*EMA - EMA(EMA)
    ema_period = 20
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    ema_of_ema = ema.ewm(span=ema_period, adjust=False).mean()
    df["close_dema"] = 2 * ema - ema_of_ema
    # Normalize by close price for scale invariance
    df["close_dema"] = (df["close_dema"] - df["close"]) / (df["close"] + eps)
    
    # --- VWAP (Volume Weighted Average Price) ---
    vwap = VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14
    )
    df["close_vwap"] = vwap.volume_weighted_average_price()
    # Normalize: difference from close as percentage
    df["close_vwap"] = (df["close_vwap"] - df["close"]) / (df["close"] + eps)
    
    # --- Bollinger Bands Signals ---
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    # Binary signals: 1 if price crosses band, else 0
    df["bband_up_close"] = (df["close"] > bb_upper).astype(float)
    df["close_bband_down"] = (df["close"] < bb_lower).astype(float)
    
    # --- ADL (Accumulation/Distribution Line) 2nd Order Difference ---
    adl = AccDistIndexIndicator(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
    )
    adl_line = adl.acc_dist_index()
    # 2nd order difference: diff of diff
    df["adl_diffs2"] = adl_line.diff().diff()
    # Normalize
    adl_std = df["adl_diffs2"].rolling(window=20, min_periods=1).std()
    df["adl_diffs2"] = df["adl_diffs2"] / (adl_std + eps)
    
    # --- OBV (On-Balance Volume) 2nd Order Difference ---
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    obv_line = obv.on_balance_volume()
    # 2nd order difference
    df["obv_diffs2"] = obv_line.diff().diff()
    # Normalize
    obv_std = df["obv_diffs2"].rolling(window=20, min_periods=1).std()
    df["obv_diffs2"] = df["obv_diffs2"] / (obv_std + eps)
    
    # =========================================================================
    # 4. GOOGLE TRENDS (Optional)
    # =========================================================================
    if trends_df is not None and not trends_df.empty:
        # Merge trends data by date
        trends_df = trends_df.copy()
        trends_df["date"] = pd.to_datetime(trends_df["date"]).dt.date
        df["date"] = df["open_time_dt"].dt.date
        df = df.merge(trends_df[["date", "trends"]], on="date", how="left")
        df["trends"] = df["trends"].fillna(0.0)
        df = df.drop(columns=["date"])
    else:
        df["trends"] = 0.0
    
    # =========================================================================
    # 5. NORMALIZATION & CLEANUP
    # =========================================================================
    
    # Normalize percentage-based indicators to [0, 1] range
    df["stoch"] = df["stoch"] / 100.0
    df["aroon_up"] = df["aroon_up"] / 100.0
    df["aroon_down"] = df["aroon_down"] / 100.0
    df["rsi"] = df["rsi"] / 100.0
    df["adx"] = df["adx"] / 100.0
    df["cci"] = df["cci"] / 200.0  # CCI typically ranges -200 to 200
    
    # Normalize MACD by close price
    df["macd_signal_diffs"] = df["macd_signal_diffs"] / (df["close"] + eps)
    
    # Fill NaN values (from indicators with warmup periods)
    df = df.fillna(0.0)
    
    # Replace infinities
    df = df.replace([np.inf, -np.inf], 0.0)
    
    # Clip extreme values
    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].clip(-10.0, 10.0)
    
    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature columns expected by the Neural Network."""
    return [
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


def process_ohlcv_file(
    input_path: str,
    output_path: str,
    trends_path: Optional[str] = None,
) -> pd.DataFrame:
    """Process a raw OHLCV CSV file and save processed features.
    
    Args:
        input_path: Path to input CSV with OHLCV data
        output_path: Path to save processed CSV
        trends_path: Optional path to Google Trends CSV
    
    Returns:
        Processed DataFrame
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Load trends if provided
    trends_df = None
    if trends_path:
        print(f"Loading trends from {trends_path}...")
        trends_df = pd.read_csv(trends_path)
    
    # Generate features
    print("Generating features...")
    df = generate_features(df, trends_df)
    
    # Select columns to save (OHLCV + features)
    feature_cols = get_feature_columns()
    output_cols = ["open_time_dt", "open", "high", "low", "close", "volume"] + feature_cols
    
    # Filter to existing columns
    output_cols = [col for col in output_cols if col in df.columns]
    df_out = df[output_cols]
    
    # Save
    print(f"Saving processed data to {output_path}...")
    df_out.to_csv(output_path, index=False)
    print(f"Saved {len(df_out)} rows with {len(feature_cols)} features")
    
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Process OHLCV data with technical indicators")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--trends", "-t", help="Optional Google Trends CSV file path")
    
    args = parser.parse_args()
    
    process_ohlcv_file(args.input, args.output, args.trends)
    print("Done!")


if __name__ == "__main__":
    main()
