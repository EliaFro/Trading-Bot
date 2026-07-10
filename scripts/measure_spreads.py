#!/usr/bin/env python3
"""
Measure real bid-ask spreads from the Binance order book (public endpoint).

Samples the top of book N times per symbol, reports mean/median/p95 relative
spread, and the depth-weighted spread for a small marketable order. Output
feeds the Fast Lab execution model (docs/spread_measurements.json).

Honesty notes baked into the output:
  * spreads are sampled at ONE point in time — calm-market values. Stressed
    spreads are wider, so the execution model applies a conservative floor
    (>= p95 measured, never below 1bp for majors / 2bp for SOL).
  * for our order sizes ($10-$3,300) top-of-book depth on these pairs is
    hundreds of times larger, so depth impact beyond the spread is covered
    by the separate 0.05% slippage term (kept unchanged from Phase 2).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import ccxt

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
N_SAMPLES = 30
INTERVAL_S = 2.0
# conservative floors (relative spread) applied regardless of measurement
FLOORS = {'BTC/USDT': 0.0001, 'ETH/USDT': 0.0001, 'SOL/USDT': 0.0002}


def main():
    exchange = ccxt.binance({'enableRateLimit': True})
    samples = {s: [] for s in SYMBOLS}
    depth_usd = {s: [] for s in SYMBOLS}

    print(f"sampling top-of-book {N_SAMPLES}x per symbol "
          f"({INTERVAL_S}s apart)...")
    for i in range(N_SAMPLES):
        for symbol in SYMBOLS:
            try:
                ob = exchange.fetch_order_book(symbol, limit=5)
                bid, ask = ob['bids'][0][0], ob['asks'][0][0]
                mid = (bid + ask) / 2
                samples[symbol].append((ask - bid) / mid)
                depth_usd[symbol].append(
                    min(ob['bids'][0][1], ob['asks'][0][1]) * mid)
            except Exception as e:
                print(f"  sample {i} {symbol}: {e}")
        time.sleep(INTERVAL_S)

    out = {'measured_at': datetime.now(timezone.utc).isoformat(),
           'n_samples': N_SAMPLES, 'symbols': {}}
    print(f"\n{'symbol':10s} {'mean':>9s} {'median':>9s} {'p95':>9s} "
          f"{'floor':>9s} {'USED':>9s} {'ToB depth $':>12s}")
    for symbol in SYMBOLS:
        arr = np.array(samples[symbol])
        floor = FLOORS[symbol]
        used = float(max(np.percentile(arr, 95), floor))
        out['symbols'][symbol] = {
            'mean': float(arr.mean()), 'median': float(np.median(arr)),
            'p95': float(np.percentile(arr, 95)), 'max': float(arr.max()),
            'floor': floor, 'used_spread': used,
            'top_of_book_depth_usd_median': float(np.median(depth_usd[symbol])),
        }
        print(f"{symbol:10s} {arr.mean():>9.5%} {np.median(arr):>9.5%} "
              f"{np.percentile(arr, 95):>9.5%} {floor:>9.5%} {used:>9.5%} "
              f"{np.median(depth_usd[symbol]):>12,.0f}")

    Path('docs/spread_measurements.json').write_text(
        json.dumps(out, indent=2))
    print("\n-> docs/spread_measurements.json")


if __name__ == '__main__':
    main()
