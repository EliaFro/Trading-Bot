#!/usr/bin/env python3
"""
Published-strategy library study with explicit anti-data-mining guards.

Pipeline (identical to the TSMOM study): daily bars, 36 months, continuous
OOS simulation on the same 51-window calendar, both fee tiers, both sizing
variants, long-only, conservative fills.

Anti-data-mining guards (the point of this study):
  1. N is counted and reported. Naive expectation: N x alpha false positives.
  2. Bonferroni: a strategy's window-return t-test must clear p < 0.05/N.
  3. White's Reality Check: moving-block bootstrap of the demeaned window-
     return matrix (preserving cross-strategy correlation); the observed
     best strategy must beat the bootstrap distribution of the best.
  4. Pure-noise control: 100 random strategies with matched trade frequency
     run through the IDENTICAL pipeline; their naive "pass" count is the
     luck baseline any real result must clearly exceed.
"""

import json
import sys
from collections import Counter
from datetime import timedelta
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
from src.backtesting.walkforward import simulate
from src.backtesting.signal_library import (LIBRARY, N_STRATEGIES,
                                            noise_strategy)
from scripts.run_daily_momentum import (SYMBOLS, daily_frame,
                                        window_calendar, classify_regimes)

FEES = {'taker': 0.001, 'bnb': 0.00075}
SIZINGS = {'rail_10pct': 0.10, 'equal_weight_33pct': 1.0 / 3}
N_NOISE = 100
ALPHA = 0.05
GATE = {'profit_factor': 1.15, 'max_drawdown': 0.15,
        'min_trades': 100, 'positive_windows': 0.60}
# NOTE on min_trades: the Phase 2 gate said 200 for intraday; the TSMOM study
# established ~100 as the honest daily-horizon expectation. We keep 100 here
# and report every count — no strategy below it can claim significance anyway
# because the t-test does the real work.


def run_strategy(builder, frames, size, fee, oos_start, oos_end):
    wf.COMMISSION = fee
    wf.MAX_POSITION_SIZE = size + 1e-9
    sims = {}
    for symbol in SYMBOLS:
        df = frames[symbol]
        start_idx = int(df.index.searchsorted(oos_start))
        end_idx = int(df.index.searchsorted(oos_end))
        sims[symbol] = simulate(df, builder(df, size=size),
                                start_idx=start_idx, end_idx=end_idx,
                                initial_equity=10_000.0)
    curves = [s.equity / s.equity.iloc[0] for s in sims.values()
              if s.equity is not None and len(s.equity) > 1]
    curve = (pd.concat(curves, axis=1).sort_index().ffill().mean(axis=1)
             * 10_000.0) if curves else pd.Series(dtype=float)
    trades = [t for s in sims.values() for t in s.trades]
    fees_paid = sum(s.fees for s in sims.values())
    return curve, trades, fees_paid


def window_returns(curve, windows):
    out = []
    for w_start, w_end in windows:
        seg = curve.loc[w_start:w_end]
        out.append(float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) > 1 else 0.0)
    return np.array(out)


def econ_metrics(curve, trades, fees_paid, windows):
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gw, gl = sum(t['pnl'] for t in wins), -sum(t['pnl'] for t in losses)
    pf = gw / gl if gl > 0 else (np.inf if gw else 0.0)
    if curve.empty or len(curve) < 10:
        return None
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    wr = window_returns(curve, windows)
    return {
        'total_return': float(curve.iloc[-1] / curve.iloc[0] - 1),
        'profit_factor': pf, 'max_drawdown': dd,
        'n_trades': len(trades), 'fees': fees_paid,
        'positive_windows': int((wr > 0).sum()), 'n_windows': len(wr),
        'window_returns': wr,
    }


def passes_econ_gate(m):
    return (m and m['profit_factor'] > GATE['profit_factor']
            and abs(m['max_drawdown']) < GATE['max_drawdown']
            and m['n_trades'] >= GATE['min_trades']
            and m['positive_windows'] / m['n_windows'] >= GATE['positive_windows'])


def t_and_p(wr):
    if wr.std(ddof=1) == 0:
        return 0.0, 1.0
    t = wr.mean() / wr.std(ddof=1) * np.sqrt(len(wr))
    p = float(stats.t.sf(t, df=len(wr) - 1))          # one-sided: mean > 0
    return float(t), p


def reality_check(returns_matrix: np.ndarray, block: int = 4,
                  n_boot: int = 5000, seed: int = 7):
    """White (2000) Reality Check via moving-block bootstrap.
    returns_matrix: (n_windows, n_strategies). Returns (observed max t,
    p-value that the best strategy's t is explainable by luck)."""
    rng = np.random.default_rng(seed)
    n_w, n_s = returns_matrix.shape
    demeaned = returns_matrix - returns_matrix.mean(axis=0, keepdims=True)

    def t_stats(matrix):
        sd = matrix.std(axis=0, ddof=1)
        sd[sd == 0] = np.inf
        return matrix.mean(axis=0) / sd * np.sqrt(n_w)

    observed_t = t_stats(returns_matrix)
    observed_max = float(observed_t.max())

    n_blocks = int(np.ceil(n_w / block))
    count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n_w - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n_w]
        boot_max = t_stats(demeaned[idx]).max()
        if boot_max >= observed_max:
            count += 1
    return observed_max, count / n_boot


def main():
    db = DatabaseManager('./data/trading_system.db')
    frames = {s: daily_frame(db, s) for s in SYMBOLS}
    windows = window_calendar(frames['BTC/USDT'].index)
    regimes = classify_regimes(frames['BTC/USDT'], windows)
    oos_start, oos_end = windows[0][0], windows[-1][1]
    print(f"OOS {oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d}, "
          f"{len(windows)} windows, regimes {Counter(regimes)}")
    print(f"N = {N_STRATEGIES} published strategies; "
          f"expected false positives at naive p<{ALPHA}: "
          f"{N_STRATEGIES * ALPHA:.1f}")

    btc = frames['BTC/USDT'].loc[oos_start:oos_end, 'close']
    hold_btc = float(btc.iloc[-1] / btc.iloc[0] - 1)

    # ── Library pass ─────────────────────────────────────────────────────
    results = {}
    wr_matrix = []
    for name, builder in LIBRARY.items():
        row = {'name': name, 'doc': (builder.__doc__ or '').strip()}
        for fee_name, fee in FEES.items():
            for sizing_name, size in SIZINGS.items():
                curve, trades, fees_paid = run_strategy(
                    builder, frames, size, fee, oos_start, oos_end)
                m = econ_metrics(curve, trades, fees_paid, windows)
                key = f"{fee_name}|{sizing_name}"
                if m:
                    row[key] = {k: v for k, v in m.items()
                                if k != 'window_returns'}
                    row[key]['econ_gate'] = passes_econ_gate(m)
                    if fee_name == 'taker' and sizing_name == 'rail_10pct':
                        wr = m['window_returns']
                        row['t_stat'], row['p_naive'] = t_and_p(wr)
                        wr_matrix.append(wr)
                        row['_curve'] = curve
        results[name] = row
        base = row.get('taker|rail_10pct', {})
        print(f"{name:22s} ret {base.get('total_return', 0):+7.1%} "
              f"PF {base.get('profit_factor', 0):5.2f} "
              f"DD {base.get('max_drawdown', 0):6.1%} "
              f"n {base.get('n_trades', 0):4d} "
              f"+w {base.get('positive_windows', 0)}/{len(windows)} "
              f"t {row.get('t_stat', 0):+.2f} p {row.get('p_naive', 1):.3f} "
              f"{'ECON-PASS' if base.get('econ_gate') else ''}")

    wr_matrix = np.column_stack(wr_matrix)         # (windows, strategies)

    # ── Corrections ──────────────────────────────────────────────────────
    bonferroni = ALPHA / N_STRATEGIES
    max_t, rc_p = reality_check(wr_matrix)
    naive_sig = [n for n, r in results.items() if r.get('p_naive', 1) < ALPHA]
    bonf_sig = [n for n, r in results.items() if r.get('p_naive', 1) < bonferroni]
    econ_pass = [n for n, r in results.items()
                 if r.get('taker|rail_10pct', {}).get('econ_gate')]

    print(f"\nnaive p<{ALPHA}: {len(naive_sig)} {naive_sig}")
    print(f"Bonferroni p<{bonferroni:.4f}: {len(bonf_sig)} {bonf_sig}")
    print(f"Reality Check: best t={max_t:.2f}, p={rc_p:.3f}")
    print(f"econ gate passers (naive): {len(econ_pass)} {econ_pass}")

    # ── Noise control ────────────────────────────────────────────────────
    print(f"\nnoise control: {N_NOISE} random strategies, identical pipeline")
    noise_stats = []
    for i in range(N_NOISE):
        curve, trades, fees_paid = run_strategy(
            noise_strategy(1000 + i), frames, SIZINGS['rail_10pct'],
            FEES['taker'], oos_start, oos_end)
        m = econ_metrics(curve, trades, fees_paid, windows)
        if m:
            t, p = t_and_p(m['window_returns'])
            noise_stats.append({
                'seed': 1000 + i, 't': t, 'p': p,
                'pf': m['profit_factor'],
                'total_return': m['total_return'],
                'econ_gate': passes_econ_gate(m),
                'p_naive_sig': p < ALPHA,
            })
    n_noise_econ = sum(1 for s in noise_stats if s['econ_gate'])
    n_noise_sig = sum(1 for s in noise_stats if s['p_naive_sig'])
    best_noise = max(noise_stats, key=lambda s: s['t'])
    print(f"noise passing naive econ gate: {n_noise_econ}/{N_NOISE}")
    print(f"noise with naive p<{ALPHA}: {n_noise_sig}/{N_NOISE}")
    print(f"best noise strategy: t={best_noise['t']:.2f} "
          f"PF={best_noise['pf']:.2f} ret={best_noise['total_return']:+.1%}")

    # ── Persist ──────────────────────────────────────────────────────────
    dump = {
        'oos_span': f"{oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d}",
        'n_windows': len(windows),
        'regimes': dict(Counter(regimes)),
        'hold_btc': hold_btc,
        'n_strategies': N_STRATEGIES,
        'alpha': ALPHA,
        'bonferroni_bar': bonferroni,
        'expected_false_positives_naive': N_STRATEGIES * ALPHA,
        'reality_check': {'best_t': max_t, 'p': rc_p},
        'naive_significant': naive_sig,
        'bonferroni_significant': bonf_sig,
        'econ_gate_passers_naive': econ_pass,
        'noise': {'n': N_NOISE, 'econ_gate_passers': n_noise_econ,
                  'naive_significant': n_noise_sig,
                  'best': {k: v for k, v in best_noise.items()},
                  'all_t': [s['t'] for s in noise_stats]},
        'strategies': {n: {k: v for k, v in r.items() if k != '_curve'}
                       for n, r in results.items()},
    }
    Path('docs/signal_library_metrics.json').write_text(
        json.dumps(dump, indent=2, default=str))
    print("\nmetrics -> docs/signal_library_metrics.json")

    # ── Charts ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    lib_t = [r['t_stat'] for r in results.values() if 't_stat' in r]
    axes[0].hist([s['t'] for s in noise_stats], bins=25, alpha=0.6,
                 label=f'{N_NOISE} random strategies', color='gray',
                 density=True)
    axes[0].hist(lib_t, bins=15, alpha=0.7,
                 label=f'{N_STRATEGIES} published strategies',
                 color='tab:blue', density=True)
    axes[0].axvline(stats.t.ppf(1 - ALPHA, len(windows) - 1), color='orange',
                    ls='--', label=f'naive p<{ALPHA}')
    axes[0].axvline(stats.t.ppf(1 - bonferroni, len(windows) - 1),
                    color='red', ls='--', label='Bonferroni bar')
    axes[0].set_title('t-statistics: published strategies vs pure noise')
    axes[0].set_xlabel('t (20-day window returns)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    top5 = sorted((r for r in results.values() if '_curve' in r),
                  key=lambda r: -r.get('t_stat', -9))[:5]
    for r in top5:
        c = r['_curve']
        axes[1].plot(c.index, c / 10_000.0,
                     label=f"{r['name']} (t={r['t_stat']:.2f})", lw=1.2)
    axes[1].plot(btc.index, btc / btc.iloc[0], ls='--', color='orange',
                 lw=1, label=f'Hold BTC ({hold_btc:+.0%})')
    axes[1].axhline(1, color='gray', lw=0.8)
    axes[1].set_title('Top-5 by t-stat vs hold-BTC (rail sizing, 0.10% fee)')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('docs/phase2_charts/signal_library.png', dpi=110)
    print("chart -> docs/phase2_charts/signal_library.png")


if __name__ == '__main__':
    main()
