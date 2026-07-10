#!/usr/bin/env python3
"""
Signal family #1: daily-bar time-series momentum (TSMOM), weekly rebalance.

Design (pre-registered, nothing tuned from data):
  * daily UTC bars resampled from stored 1h candles
  * lookbacks: 20 / 40 / 60 / 90 days — ALL reported, none selected
  * decision day: Monday UTC close; execution next day's open
  * long when trailing lookback return > 0, else cash (long-only spot)
  * no stop-loss/take-profit: the exit is the signal flip (canonical TSMOM)
  * execution model identical to Phase 2: marketable LIMIT (lapses on gaps),
    0.05% slippage, fees 0.10% or 0.075% per side
  * sizing variants: 10%/position (Phase 4 rail) and 33%/position
    (equal-weight; would require a deliberate rail change)

Evaluation uses the IDENTICAL Phase 2 walk-forward calendar: the same 51
twenty-day OOS windows (first window starts after the 60d+2d train+embargo
offset) — consistency and regime metrics come from slicing the continuous
OOS equity curve on those exact boundaries. With zero optimized parameters
there is nothing to train, so walk-forward reduces to a pure out-of-sample
evaluation on the same calendar. The gate is unchanged.
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.database import DatabaseManager
from src.backtesting import walkforward as wf
from src.backtesting.walkforward import SignalArrays, simulate

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
LOOKBACKS = [20, 40, 60, 90]
FEES = {'taker_0.10%': 0.001, 'bnb_0.075%': 0.00075}
SIZINGS = {'rail_10pct': 0.10, 'equal_weight_33pct': 1.0 / 3}
REBALANCE_WEEKDAY = 0          # Monday UTC, pre-registered
TRAIN_OFFSET_DAYS = 62         # identical calendar: 60d train + 2d embargo
TEST_DAYS = 20

GATE = {'profit_factor': 1.15, 'max_drawdown': 0.15,
        'min_trades': 200, 'positive_windows': 0.60}


def daily_frame(db, symbol) -> pd.DataFrame:
    """Daily UTC bars from stored 1h candles."""
    h1 = db.get_ohlcv_data(symbol, '1h')
    h1 = h1.set_index('timestamp').sort_index()
    d = pd.DataFrame({
        'open': h1['open'].resample('1D').first(),
        'high': h1['high'].resample('1D').max(),
        'low': h1['low'].resample('1D').min(),
        'close': h1['close'].resample('1D').last(),
        'volume': h1['volume'].resample('1D').sum(),
    }).dropna()
    return d


def tsmom_signals(df: pd.DataFrame, lookback: int, size: float) -> SignalArrays:
    """Long when trailing `lookback`-day return > 0, decided on Mondays."""
    close = df['close'].to_numpy(float)
    momentum = pd.Series(close).pct_change(lookback).to_numpy()
    is_monday = (df.index.dayofweek == REBALANCE_WEEKDAY)

    entry = np.nan_to_num(momentum > 0) & is_monday
    exit_ = np.nan_to_num(momentum <= 0) & is_monday
    n = len(df)
    return SignalArrays(
        entry=entry.astype(bool),
        exit_=exit_.astype(bool),
        confidence=np.where(entry, 0.99, 0.0),
        size=np.full(n, size),
        stop_loss=np.full(n, np.nan),      # exit = signal flip (no stops)
        take_profit=np.full(n, np.nan),
    )


def window_calendar(index: pd.DatetimeIndex):
    """The identical Phase 2 walk-forward OOS windows on this data range."""
    windows = []
    start = index[0] + timedelta(days=TRAIN_OFFSET_DAYS)
    while start + timedelta(days=TEST_DAYS) <= index[-1]:
        windows.append((start, start + timedelta(days=TEST_DAYS)))
        start += timedelta(days=TEST_DAYS)
    return windows


def slice_metrics(curve: pd.Series, windows) -> list:
    """Per-window returns from the continuous OOS equity curve."""
    out = []
    for w_start, w_end in windows:
        seg = curve.loc[w_start:w_end]
        if len(seg) < 2:
            out.append(0.0)
        else:
            out.append(float(seg.iloc[-1] / seg.iloc[0] - 1))
    return out


def classify_regimes(btc_daily: pd.DataFrame, windows) -> list:
    regimes = []
    closes = btc_daily['close']
    for w_start, w_end in windows:
        seg = closes.loc[w_start:w_end]
        if len(seg) < 2:
            regimes.append('unknown')
            continue
        r = float(seg.iloc[-1] / seg.iloc[0] - 1)
        regimes.append('bull' if r > 0.05 else 'bear' if r < -0.05 else 'sideways')
    return regimes


def run_config(frames, lookback, size, fee):
    """One (lookback, sizing, fee) config across all symbols + portfolio."""
    wf.COMMISSION = fee
    wf.MAX_POSITION_SIZE = size + 1e-9      # sizing variant sets the cap

    per_symbol = {}
    for symbol in SYMBOLS:
        df = frames[symbol]
        oos_start_idx = int(df.index.searchsorted(
            df.index[0] + timedelta(days=TRAIN_OFFSET_DAYS)))
        sim = simulate(df, tsmom_signals(df, lookback, size),
                       start_idx=oos_start_idx, end_idx=len(df),
                       initial_equity=10_000.0)
        per_symbol[symbol] = sim

    curves = []
    for symbol, sim in per_symbol.items():
        if sim.equity is not None and len(sim.equity) > 1:
            curves.append(sim.equity / sim.equity.iloc[0])
    portfolio = (pd.concat(curves, axis=1).sort_index().ffill().mean(axis=1)
                 * 10_000.0) if curves else pd.Series(dtype=float)

    return per_symbol, portfolio


def full_metrics(trades, curve, windows, regimes, tf_label='1d'):
    window_rets = slice_metrics(curve, windows) if not curve.empty else []
    positive = sum(1 for r in window_rets if r > 0)

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = -sum(t['pnl'] for t in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else \
        (float('inf') if gross_win else 0.0)

    if not curve.empty and len(curve) > 10:
        daily = curve.resample('1D').last().dropna()
        rets = daily.pct_change().dropna()
        sharpe = float(np.sqrt(365) * rets.mean() / rets.std()) \
            if len(rets) > 2 and rets.std() > 0 else 0.0
        downside = rets[rets < 0]
        sortino = float(np.sqrt(365) * rets.mean() / downside.std()) \
            if len(downside) > 1 and downside.std() > 0 else 0.0
        run_max = curve.cummax()
        max_dd = float(((curve - run_max) / run_max).min())
        total_return = float(curve.iloc[-1] / curve.iloc[0] - 1)
        years = (curve.index[-1] - curve.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        sharpe = sortino = max_dd = total_return = cagr = 0.0

    # significance of window returns (the honest unit at this horizon)
    wr = np.array(window_rets)
    t_stat = float(wr.mean() / wr.std() * np.sqrt(len(wr))) \
        if len(wr) > 2 and wr.std() > 0 else 0.0

    # per-regime breakdown from the same windows
    regime_stats = {}
    for regime in ('bull', 'sideways', 'bear'):
        rs = [r for r, reg in zip(window_rets, regimes) if reg == regime]
        if rs:
            regime_stats[regime] = {
                'n_windows': len(rs),
                'mean_window_return': float(np.mean(rs)),
                'positive_pct': sum(1 for x in rs if x > 0) / len(rs),
                'total_return': float(np.prod([1 + x for x in rs]) - 1),
            }

    gross_profit_total = gross_win - gross_loss
    fees_total = sum(t.get('fees', 0.0) for t in trades)

    hold_days = [t['bars_held'] for t in trades]
    return {
        'total_return': total_return, 'cagr': cagr, 'profit_factor': pf,
        'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd,
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'n_trades': len(trades),
        'avg_hold_days': float(np.mean(hold_days)) if hold_days else 0.0,
        'fees': fees_total,
        'n_windows': len(window_rets), 'positive_windows': positive,
        'positive_window_pct': positive / len(window_rets) if window_rets else 0.0,
        't_stat_windows': t_stat,
        'regimes': regime_stats,
    }


def verdict(m_taker, m_bnb):
    def passes(m):
        return (m['profit_factor'] > GATE['profit_factor']
                and abs(m['max_drawdown']) < GATE['max_drawdown']
                and m['n_trades'] >= GATE['min_trades']
                and m['positive_window_pct'] >= GATE['positive_windows'])
    if passes(m_taker):
        return 'PASS'
    if passes(m_bnb):
        return 'CONDITIONAL PASS (fees)'
    return 'FAIL'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-out', default='docs/daily_momentum_metrics.json')
    args = parser.parse_args()

    db = DatabaseManager('./data/trading_system.db')
    frames = {s: daily_frame(db, s) for s in SYMBOLS}
    for s, f in frames.items():
        print(f"{s}: {len(f)} daily bars {f.index[0]:%Y-%m-%d} -> "
              f"{f.index[-1]:%Y-%m-%d}")

    windows = window_calendar(frames['BTC/USDT'].index)
    regimes = classify_regimes(frames['BTC/USDT'], windows)
    from collections import Counter
    print(f"{len(windows)} OOS windows, regimes: {Counter(regimes)}")

    # Benchmarks over the OOS span
    oos_start = windows[0][0]
    oos_end = windows[-1][1]
    bench = {}
    for s in SYMBOLS:
        seg = frames[s].loc[oos_start:oos_end, 'close']
        bench[s] = float(seg.iloc[-1] / seg.iloc[0] - 1)
    bench['basket'] = float(np.mean(list(bench.values())))
    print(f"benchmarks (hold, OOS span): "
          + ", ".join(f"{k} {v:+.1%}" for k, v in bench.items()))

    results = {}
    curves_for_charts = {}
    for sizing_name, size in SIZINGS.items():
        for lookback in LOOKBACKS:
            fee_metrics = {}
            for fee_name, fee in FEES.items():
                per_symbol, portfolio = run_config(frames, lookback, size, fee)
                all_trades = []
                for symbol, sim in per_symbol.items():
                    for t in sim.trades:
                        t2 = dict(t)
                        # commission both legs, from trade economics
                        entry_val = t['qty'] * t['entry_price']
                        exit_val = t['qty'] * t['exit_price']
                        t2['fees'] = (entry_val + exit_val) * fee
                        t2['symbol'] = symbol
                        all_trades.append(t2)
                m = full_metrics(all_trades, portfolio, windows, regimes)
                # per-symbol trade counts + gross for fee-relevance reporting
                gross_sum = sum(t['pnl'] + t['fees'] for t in all_trades)
                m['gross_pnl'] = gross_sum
                m['fees_pct_of_gross'] = (m['fees'] / abs(gross_sum)
                                          if gross_sum else 0.0)
                fee_metrics[fee_name] = m
                if fee_name == 'taker_0.10%':
                    curves_for_charts[(sizing_name, lookback)] = portfolio

            key = f"tsmom_{lookback}d|{sizing_name}"
            results[key] = {
                'taker': fee_metrics['taker_0.10%'],
                'bnb': fee_metrics['bnb_0.075%'],
                'verdict': verdict(fee_metrics['taker_0.10%'],
                                   fee_metrics['bnb_0.075%']),
            }
            t = fee_metrics['taker_0.10%']
            print(f"TSMOM {lookback:2d}d {sizing_name:18s} | "
                  f"ret {t['total_return']:+8.1%} (CAGR {t['cagr']:+.1%}) "
                  f"PF {t['profit_factor']:5.2f} DD {t['max_drawdown']:6.1%} "
                  f"trades {t['n_trades']:3d} +win {t['positive_windows']}/"
                  f"{t['n_windows']} t={t['t_stat_windows']:+.2f} "
                  f"fees%gross {t['fees_pct_of_gross']:.1%} "
                  f"-> {results[key]['verdict']}")

    Path(args.json_out).write_text(json.dumps(
        {'benchmarks': bench, 'results': results,
         'windows': len(windows),
         'regime_counts': dict(Counter(regimes))},
        indent=2, default=str))
    print(f"\nmetrics -> {args.json_out}")

    # Chart: equal-weight sizing, all lookbacks vs benchmarks
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, sizing_name in zip(axes, SIZINGS):
        for lookback in LOOKBACKS:
            c = curves_for_charts.get((sizing_name, lookback))
            if c is not None and not c.empty:
                ax.plot(c.index, c / c.iloc[0], lw=1.3,
                        label=f'TSMOM {lookback}d')
        btc = frames['BTC/USDT'].loc[oos_start:oos_end, 'close']
        ax.plot(btc.index, btc / btc.iloc[0], color='orange', ls='--', lw=1,
                label=f"Hold BTC ({bench['BTC/USDT']:+.0%})")
        ax.axhline(1.0, color='gray', lw=0.8, label='Cash')
        ax.set_title(f"Daily TSMOM portfolio — {sizing_name} — OOS, fee 0.10%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path('docs/phase2_charts/daily_tsmom.png')
    fig.savefig(out, dpi=110)
    print(f"chart -> {out}")


if __name__ == '__main__':
    main()
