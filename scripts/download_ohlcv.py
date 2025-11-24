"""Download Binance OHLCV (klines) to the local data/ folder.

Usage:
    python scripts/download_ohlcv.py --symbol BTCUSDT --interval 1h \
           --start "2024-01-01" --end "2024-06-01"

Notes:
- Uses Binance USDT-M futures by default. Pass --spot to use spot klines.
- Reads API keys from env BINANCE_API_KEY / BINANCE_API_SECRET (recommended) or CLI flags.
- Requires: python-binance, pandas.
- Output CSV is written to data/<symbol>_<interval>_<start>_<end>.csv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from binance import Client
from dotenv import load_dotenv
from tqdm import tqdm

# Interval mapping to milliseconds for pagination increments.
# Binance limits klines per request; we page forward using this increment.
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Binance OHLCV to data/")
    p.add_argument("--symbol", required=True, help="Trading pair, e.g., BTCUSDT")
    p.add_argument(
        "--interval",
        default="1h",
        choices=INTERVAL_MS.keys(),
        help="Kline interval",
    )
    p.add_argument("--start", required=True, help='Start date/time (e.g., "2024-01-01")')
    p.add_argument("--end", default=None, help='End date/time (optional, e.g., "2024-06-01")')
    p.add_argument(
        "--spot",
        action="store_true",
        help="Use spot klines instead of USDT-M futures (default futures)",
    )
    p.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet endpoints (both spot and futures have separate testnets)",
    )
    return p.parse_args()


def to_ts_ms(date_str: str) -> int:
    return int(pd.to_datetime(date_str, utc=True).timestamp() * 1000)


def fetch_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None,
    spot: bool,
) -> List[List]:
    """Paginate through klines until end_ms or data exhausted.

    We use small page steps (limit=1500) to avoid hitting server caps and to
    maintain predictable memory usage for long date ranges. Progress bar uses
    the number of candles fetched (approximated when end_ms is known).
    """
    data: List[List] = []
    curr = start_ms
    increment = INTERVAL_MS[interval]
    total_steps = None
    if end_ms is not None:
        total_steps = max(1, int((end_ms - start_ms) // increment))

    with tqdm(total=total_steps, unit="bar", desc=f"{symbol} {interval}") as pbar:
        while True:
            if spot:
                klines = client.get_klines(
                    symbol=symbol, interval=interval, startTime=curr, endTime=end_ms, limit=1500
                )
            else:
                klines = client.futures_klines(
                    symbol=symbol, interval=interval, startTime=curr, endTime=end_ms, limit=1500
                )

            if not klines:
                break

            data.extend(klines)
            pbar.update(len(klines))

            last_open_time = klines[-1][0]
            curr = last_open_time + increment
            if end_ms is not None and curr > end_ms:
                break

    return data


def klines_to_df(raw: Iterable[List], symbol: str, interval: str) -> pd.DataFrame:
    cols = [
        "open_time",  # milliseconds
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",  # milliseconds
        "quote_volume",
        "trade_count",
        "taker_base_volume",
        "taker_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    # Ensure numeric types
    numeric_cols = [c for c in cols if c not in ("open_time", "close_time")]
    df[numeric_cols] = df[numeric_cols].astype(float)
    # Add human-friendly timestamps for inspection.
    df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["interval"] = interval
    return df


def main():
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("API key/secret required via .env or environment variables.")

    client = Client(api_key, api_secret, testnet=args.testnet)

    start_ms = to_ts_ms(args.start)
    end_ms = to_ts_ms(args.end) if args.end else None

    print(f"Downloading {args.symbol} {args.interval} klines from {args.start} to {args.end or 'latest'}...")
    raw = fetch_klines(client, args.symbol, args.interval, start_ms, end_ms, spot=args.spot)
    if not raw:
        raise SystemExit("No klines returned. Check symbol/interval/date range.")

    df = klines_to_df(raw, args.symbol, args.interval)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    start_tag = pd.to_datetime(args.start).strftime("%Y%m%d")
    end_tag = (pd.to_datetime(args.end).strftime("%Y%m%d") if args.end else "latest")
    dest = data_dir / f"{args.symbol}_{args.interval}_{start_tag}_{end_tag}.csv"
    df.to_csv(dest, index=False)

    print(f"Saved {len(df)} rows to {dest}")


if __name__ == "__main__":
    main()
