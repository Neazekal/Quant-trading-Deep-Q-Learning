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
    p.add_argument(
        "--with-funding",
        action="store_true",
        help="Include funding rates (futures only); merged and forward-filled onto klines.",
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

    We use small page steps (limit=1000) to stay well under API rate limits and
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
                    symbol=symbol, interval=interval, startTime=curr, endTime=end_ms, limit=1000
                )
            else:
                klines = client.futures_klines(
                    symbol=symbol, interval=interval, startTime=curr, endTime=end_ms, limit=1000
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


def fetch_funding_rates(
    client: Client, symbol: str, start_ms: int, end_ms: int | None
) -> pd.DataFrame:
    """Fetch futures funding rates and return as DataFrame."""
    rows: List[dict] = []
    curr = start_ms
    while True:
        res = client.futures_funding_rate(
            symbol=symbol, startTime=curr, endTime=end_ms, limit=1000
        )
        if not res:
            break
        rows.extend(res)
        last_ts = int(res[-1]["fundingTime"])
        curr = last_ts + 1  # move past last record
        if end_ms is not None and curr > end_ms:
            break

    if not rows:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    df = pd.DataFrame(rows)
    df["fundingTime"] = df["fundingTime"].astype(int)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["funding_dt"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df


def klines_to_df(raw: Iterable[List], symbol: str, interval: str) -> pd.DataFrame:
    """Convert raw kline array to a lean OHLCV DataFrame."""
    cols = [
        "open_time",  # ms
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",  # ms
        "quote_volume",
        "trade_count",
        "taker_base_volume",
        "taker_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)

    # Cast numerics and add human-friendly timestamp.
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # Keep only the essentials to reduce memory and file size.
    df = df[["open_time_dt", "open", "high", "low", "close", "volume"]]
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

    # Funding rates (futures only)
    if args.with_funding:
        if args.spot:
            print("Skipping funding: spot market selected.")
        else:
            fr = fetch_funding_rates(client, args.symbol, start_ms, end_ms)
            if fr.empty:
                print("Warning: no funding rates returned for range; funding_rate set to 0.")
                df["funding_rate"] = 0.0
            else:
                # Align funding to klines: latest funding rate effective until next funding event.
                fr = fr.sort_values("funding_dt")
                df = df.sort_values("open_time_dt")
                merged = pd.merge_asof(
                    df,
                    fr[["funding_dt", "fundingRate"]],
                    left_on="open_time_dt",
                    right_on="funding_dt",
                    direction="backward",
                )
                merged["fundingRate"] = merged["fundingRate"].fillna(0.0)
                merged = merged.rename(columns={"fundingRate": "funding_rate"})
                df = merged.drop(columns=["funding_dt"])

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    start_tag = pd.to_datetime(args.start).strftime("%Y%m%d")
    end_tag = (pd.to_datetime(args.end).strftime("%Y%m%d") if args.end else "latest")
    dest = data_dir / f"{args.symbol}_{args.interval}_{start_tag}_{end_tag}.csv"
    df.to_csv(dest, index=False)

    print(f"Saved {len(df)} rows to {dest}")


if __name__ == "__main__":
    main()
