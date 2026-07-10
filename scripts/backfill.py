#!/usr/bin/env python3
"""
Historical OHLCV backfill from Binance public endpoints.

Downloads at least 12 months of candles per symbol/timeframe into the ohlcv
table. Resumable: re-running continues from the newest stored bar.

Usage:
    python scripts/backfill.py                          # config symbols/timeframes, 12 months
    python scripts/backfill.py --months 18
    python scripts/backfill.py --symbols BTC/USDT --timeframes 5m 15m
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.config import Config
from src.utils.database import DatabaseManager
from src.data.market_data import MarketData

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('backfill')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--months', type=int, default=12)
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--timeframes', nargs='+', default=None)
    args = parser.parse_args()

    config = Config.load()
    symbols = args.symbols or config.trading.get('symbols', ['BTC/USDT'])
    timeframes = args.timeframes or config.trading.get('timeframes', ['5m'])

    db = DatabaseManager(config.database)
    md = MarketData(db=db)

    started = time.time()
    grand_total = 0
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Backfilling {symbol} {timeframe} "
                        f"({args.months} months)...")
            grand_total += md.backfill(symbol, timeframe, months=args.months)
            # Close any interior holes left by interrupted earlier runs
            grand_total += md.repair_gaps(symbol, timeframe, months=args.months)

    elapsed = time.time() - started
    logger.info(f"Done. {grand_total} bars stored in {elapsed / 60:.1f} minutes.")

    # Coverage report
    from sqlalchemy import text
    with db.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT symbol, timeframe, COUNT(*),
                   MIN(timestamp), MAX(timestamp)
            FROM ohlcv GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)).fetchall()
    from datetime import datetime, timezone
    print(f"\n{'symbol':10s} {'tf':4s} {'bars':>9s}  coverage")
    for symbol, tf, n, lo, hi in rows:
        lo_s = datetime.fromtimestamp(lo, tz=timezone.utc).strftime('%Y-%m-%d')
        hi_s = datetime.fromtimestamp(hi, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        print(f"{symbol:10s} {tf:4s} {n:>9,d}  {lo_s} -> {hi_s}")


if __name__ == '__main__':
    main()
