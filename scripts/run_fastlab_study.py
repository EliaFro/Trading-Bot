#!/usr/bin/env python3
"""
Fast Lab Part B study: Elder triple-screen multi-timeframe family (N=8)
through the identical walk-forward harness with the full anti-data-mining
kit and the upgraded execution model (measured spreads + fee decomposition).

Pre-registered predictions and design: docs/FASTLAB_PLAN.md (locked before
this script first ran).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.database import DatabaseManager
from src.backtesting import walkforward as wf
from src.backtesting.walkforward import SignalArrays, simulate
from src.backtesting import mtf_library as mtf
from scripts.run_daily_momentum import window_calendar, classify_regimes, daily_frame
from scripts.run_signal_library import (window_returns, t_and_p,
                                        reality_check)

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
FEES = {'taker': 0.001, 'bnb': 0.00075}
ALPHA = 0.05
SPREADS = json.loads(Path('docs/spread_measurements.json').read_text())
SPREAD_USED = {s: SPREADS['symbols'][s]['used_spread'] for s in SYMBOLS}
N_NOISE_5M, N_NOISE_1M = 60, 40


def load_frames(db):
    frames = {}
    for symbol in SYMBOLS:
        for tf in ('1m', '5m', '1h'):
            df = db.get_ohlcv_data(symbol, tf)
            df = df.set_index('timestamp').sort_index()
            frames[(symbol, tf)] = df[['open', 'high', 'low', 'close', 'volume']]
        print(f"{symbol}: 1m={len(frames[(symbol,'1m')]):,} "
              f"5m={len(frames[(symbol,'5m')]):,} bars")
    return frames


def run_variant(builder, entry_tf, frames, fee, oos_start, oos_end):
    wf.COMMISSION = fee
    wf.MAX_POSITION_SIZE = 0.10 + 1e-9
    sims = {}
    for symbol in SYMBOLS:
        ltf = frames[(symbol, entry_tf)]
        h1 = frames[(symbol, '1h')]
        sig = builder(ltf, h1)
        start_idx = int(ltf.index.searchsorted(oos_start))
        end_idx = int(ltf.index.searchsorted(oos_end))
        sims[symbol] = simulate(ltf, sig, start_idx, end_idx,
                                initial_equity=10_000.0,
                                spread=SPREAD_USED[symbol])
    curves = [s.equity / s.equity.iloc[0] for s in sims.values()
              if s.equity is not None and len(s.equity) > 1]
    curve = (pd.concat(curves, axis=1).sort_index().ffill().mean(axis=1)
             * 10_000.0) if curves else pd.Series(dtype=float)
    trades = [t for s in sims.values() for t in s.trades]
    return curve, trades, sum(s.fees for s in sims.values())


def decompose(trades):
    """Per-trade means of the honest cost decomposition."""
    if not trades:
        return {}
    return {
        'n_trades': len(trades),
        'mean_gross_pct': float(np.mean([t['gross_pct'] for t in trades])),
        'mean_fee_pct': float(np.mean([t['fee_pct'] for t in trades])),
        'mean_spread_pct': float(np.mean([t['spread_pct'] for t in trades])),
        'mean_slip_pct': float(np.mean([t['slip_pct'] for t in trades])),
        'mean_total_cost_pct': float(np.mean([t['total_cost_pct'] for t in trades])),
        'mean_net_pct': float(np.mean([t['pnl_pct'] for t in trades])),
        'median_hold_bars': float(np.median([t['bars_held'] for t in trades])),
    }


def econ(curve, trades, fees_paid, windows):
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gw, gl = sum(t['pnl'] for t in wins), -sum(t['pnl'] for t in losses)
    pf = gw / gl if gl > 0 else (np.inf if gw else 0.0)
    if curve.empty or len(curve) < 10:
        return None
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    wr = window_returns(curve, windows)
    return {'total_return': float(curve.iloc[-1] / curve.iloc[0] - 1),
            'profit_factor': pf, 'max_drawdown': dd,
            'n_trades': len(trades), 'fees': fees_paid,
            'positive_windows': int((wr > 0).sum()),
            'n_windows': len(wr), 'window_returns': wr}


def noise_intraday(seed, p_enter, p_exit):
    def build(ltf, h1, size=0.10):
        rng = np.random.default_rng(seed)
        n = len(ltf)
        return SignalArrays(
            entry=rng.random(n) < p_enter,
            exit_=rng.random(n) < p_exit,
            confidence=np.full(n, 0.99),
            size=np.full(n, size),
            stop_loss=np.full(n, np.nan),
            take_profit=np.full(n, np.nan))
    return build


def main():
    # Pre-registered kill rule: strategy search at this horizon closes
    # permanently after 2026-08-07 (src/trading/kill_rule.py). No bypass.
    from src.trading.kill_rule import assert_search_allowed
    assert_search_allowed()

    db = DatabaseManager('./data/trading_system.db')
    frames = load_frames(db)
    calendar_index = daily_frame(db, 'BTC/USDT').index
    windows = window_calendar(calendar_index)
    regimes = classify_regimes(daily_frame(db, 'BTC/USDT'), windows)
    oos_start, oos_end = windows[0][0], windows[-1][1]
    variants = mtf.build_variants()
    n_var = len(variants)
    print(f"OOS {oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d}, "
          f"{len(windows)} windows, regimes {Counter(regimes)}")
    print(f"N = {n_var} variants; spreads used: "
          + ", ".join(f"{s} {v:.3%}" for s, v in SPREAD_USED.items()))

    results, wr_cols = {}, []
    for name, entry_tf, builder in variants:
        row = {'name': name, 'entry_tf': entry_tf}
        for fee_name, fee in FEES.items():
            curve, trades, fees_paid = run_variant(
                builder, entry_tf, frames, fee, oos_start, oos_end)
            m = econ(curve, trades, fees_paid, windows)
            row[fee_name] = {k: v for k, v in (m or {}).items()
                             if k != 'window_returns'}
            if fee_name == 'taker' and m:
                row['t_stat'], row['p_naive'] = t_and_p(m['window_returns'])
                wr_cols.append(m['window_returns'])
                row['decomposition'] = decompose(trades)
        results[name] = row
        d = row.get('decomposition', {})
        t = row.get('taker', {})
        print(f"{name:36s} ret {t.get('total_return', 0):+7.1%} "
              f"PF {t.get('profit_factor', 0):5.2f} n {t.get('n_trades', 0):5d} "
              f"| gross/trade {d.get('mean_gross_pct', 0):+.3%} vs "
              f"cost {d.get('mean_total_cost_pct', 0):.3%} "
              f"| t {row.get('t_stat', 0):+.2f}")

    wr_matrix = np.column_stack(wr_cols)
    bonferroni = ALPHA / n_var
    max_t, rc_p = reality_check(wr_matrix)
    naive_sig = [n for n, r in results.items() if r.get('p_naive', 1) < ALPHA]
    bonf_sig = [n for n, r in results.items() if r.get('p_naive', 1) < bonferroni]
    print(f"\nnaive p<{ALPHA}: {len(naive_sig)} {naive_sig}")
    print(f"Bonferroni p<{bonferroni:.4f}: {len(bonf_sig)} {bonf_sig}")
    print(f"Reality Check: best t={max_t:.2f}, p={rc_p:.3f}")

    # ── Noise control at matched frequency ──
    med_trades = {tf: np.median([r['taker']['n_trades'] / 3
                                 for r in results.values()
                                 if r['entry_tf'] == tf])
                  for tf in ('1m', '5m')}
    years = (oos_end - oos_start).days / 365.25
    noise_stats = []
    for tf, n_noise, base_seed in (('5m', N_NOISE_5M, 2000),
                                   ('1m', N_NOISE_1M, 3000)):
        bars = len(frames[('BTC/USDT', tf)].loc[oos_start:oos_end])
        trades_target = med_trades[tf]
        p_enter = min(1.5 * trades_target / bars, 0.05)
        p_exit = 2 * p_enter
        print(f"\nnoise@{tf}: {n_noise} strategies, matched to "
              f"~{trades_target:.0f} trades/symbol (p_enter={p_enter:.5f})")
        for i in range(n_noise):
            curve, trades, fees_paid = run_variant(
                noise_intraday(base_seed + i, p_enter, p_exit), tf,
                frames, FEES['taker'], oos_start, oos_end)
            m = econ(curve, trades, fees_paid, windows)
            if m:
                t, p = t_and_p(m['window_returns'])
                noise_stats.append({'tf': tf, 't': t, 'p': p,
                                    'pf': m['profit_factor'],
                                    'ret': m['total_return']})
    n_noise_sig = sum(1 for s in noise_stats if s['p'] < ALPHA)
    best_noise = max(noise_stats, key=lambda s: s['t'])
    print(f"noise with naive p<{ALPHA}: {n_noise_sig}/{len(noise_stats)}")
    print(f"best noise: tf={best_noise['tf']} t={best_noise['t']:.2f} "
          f"PF={best_noise['pf']:.2f} ret={best_noise['ret']:+.1%}")

    dump = {
        'plan': 'docs/FASTLAB_PLAN.md (pre-registered)',
        'oos_span': f"{oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d}",
        'n_windows': len(windows), 'regimes': dict(Counter(regimes)),
        'spreads_used': SPREAD_USED,
        'n_variants': n_var, 'bonferroni_bar': bonferroni,
        'naive_significant': naive_sig, 'bonferroni_significant': bonf_sig,
        'reality_check': {'best_t': max_t, 'p': rc_p},
        'noise': {'n': len(noise_stats), 'naive_significant': n_noise_sig,
                  'best': best_noise,
                  'all_t': [s['t'] for s in noise_stats]},
        'variants': results,
    }
    Path('docs/fastlab_partB_metrics.json').write_text(
        json.dumps(dump, indent=2, default=str))
    print("\nmetrics -> docs/fastlab_partB_metrics.json")

    # chart: ceiling check
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = list(results)
    gross = [results[n]['decomposition']['mean_gross_pct'] * 100 for n in names]
    cost = [results[n]['decomposition']['mean_total_cost_pct'] * 100 for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, gross, 0.4, label='mean GROSS edge per trade %', color='tab:blue')
    ax.bar(x + 0.2, cost, 0.4, label='mean round-trip COST %', color='tab:red')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('triple_', '') for n in names],
                       rotation=30, ha='right', fontsize=7)
    ax.set_title('The ceiling check: gross edge vs cost, per trade (taker fee)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('docs/phase2_charts/fastlab_partB.png', dpi=110)
    print("chart -> docs/phase2_charts/fastlab_partB.png")


if __name__ == '__main__':
    main()
