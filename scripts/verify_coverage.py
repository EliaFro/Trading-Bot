#!/usr/bin/env python3
"""
Data-coverage verification: per symbol/timeframe, report exact date range,
bar count, expected bar count, completeness %, and any interior gaps.

Usage: python scripts/verify_coverage.py [--months 36] [--max-gap-report 5]
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import text

from src.utils.database import DatabaseManager
from src.data.market_data import TIMEFRAME_SECONDS

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', type=int, default=36)
    parser.add_argument('--max-gap-report', type=int, default=5)
    args = parser.parse_args()

    db = DatabaseManager('./data/trading_system.db')
    target_start = datetime.now(timezone.utc) - timedelta(days=args.months * 30.5)

    print(f"{'symbol':9s} {'tf':4s} {'bars':>9s} {'coverage':41s} "
          f"{'complete':>8s} {'gaps>1.5x':>9s}")
    all_ok = True
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            tf_sec = TIMEFRAME_SECONDS[tf]
            with db.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT timestamp FROM ohlcv "
                    "WHERE symbol=:s AND timeframe=:tf ORDER BY timestamp"),
                    {'s': symbol, 'tf': tf}).fetchall()
            if not rows:
                print(f"{symbol:9s} {tf:4s} {'0':>9s} MISSING")
                all_ok = False
                continue
            stamps = [r[0] for r in rows]
            lo = datetime.fromtimestamp(stamps[0], tz=timezone.utc)
            hi = datetime.fromtimestamp(stamps[-1], tz=timezone.utc)
            expected = int((stamps[-1] - stamps[0]) / tf_sec) + 1
            completeness = len(stamps) / expected
            gaps = sum(1 for a, b in zip(stamps, stamps[1:])
                       if b - a > tf_sec * 1.5)
            reaches_target = lo <= target_start + timedelta(days=2)
            ok = completeness > 0.995 and gaps == 0 and reaches_target
            all_ok &= ok
            flag = '' if ok else ('  <-- INCOMPLETE' if not reaches_target
                                  else '  <-- GAPS')
            print(f"{symbol:9s} {tf:4s} {len(stamps):>9,d} "
                  f"{lo:%Y-%m-%d %H:%M} -> {hi:%Y-%m-%d %H:%M} "
                  f"{completeness:>7.2%} {gaps:>9d}{flag}")

    print(f"\n{'ALL COMBOS COMPLETE AND GAP-FREE' if all_ok else 'COVERAGE PROBLEMS FOUND'}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
