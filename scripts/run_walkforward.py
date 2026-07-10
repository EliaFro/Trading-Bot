#!/usr/bin/env python3
"""
Phase 2 walk-forward study driver.

Runs walk-forward validation (train 60d / embargo 2d / test 20d) for every
strategy x symbol x timeframe combination on real stored history, computes
out-of-sample metrics, renders equity charts, and writes docs/PHASE2_RESULTS.md.

Usage:
    python scripts/run_walkforward.py                     # full study
    python scripts/run_walkforward.py --tag iter1         # tagged run
    python scripts/run_walkforward.py --strategies breakout ensemble
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('walkforward')

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['5m', '15m']
STRATEGIES = ['ma_crossover', 'rsi_mean_reversion', 'breakout', 'ensemble']
TF_MINUTES = {'5m': 5, '15m': 15}

GATE = {'profit_factor': 1.15, 'max_drawdown': 0.15,
        'min_trades': 200, 'positive_windows': 0.60}

# Objective regime classification per OOS window, from BTC's return over the
# window: bull > +5%, bear < -5%, else sideways. (20-day windows; ±5% is a
# meaningful directional move for BTC on that horizon.)
REGIME_BULL = 0.05
REGIME_BEAR = -0.05


def classify_window_regimes(btc_frame: pd.DataFrame, windows) -> dict:
    """window number -> ('bull'|'bear'|'sideways', btc_return, btc_vol)."""
    out = {}
    closes = btc_frame['close']
    for w in windows:
        seg = closes.loc[w.test_start:w.test_end]
        if len(seg) < 2:
            out[w.window] = ('unknown', 0.0, 0.0)
            continue
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1)
        vol = float(seg.pct_change().std() * np.sqrt(len(seg) / 20))
        regime = ('bull' if ret > REGIME_BULL
                  else 'bear' if ret < REGIME_BEAR else 'sideways')
        out[w.window] = (regime, ret, vol)
    return out


def regime_breakdown(windows, regimes) -> dict:
    """Per-regime OOS performance for one configuration."""
    stats = {}
    for regime in ('bull', 'sideways', 'bear'):
        wins_in_regime = [w for w in windows
                          if regimes.get(w.window, ('?',))[0] == regime]
        if not wins_in_regime:
            stats[regime] = None
            continue
        trades = [t for w in wins_in_regime for t in w.trades]
        gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = -sum(t['pnl'] for t in trades if t['pnl'] <= 0)
        pf = gross_win / gross_loss if gross_loss > 0 else \
            (np.inf if gross_win else 0.0)
        rets = [w.oos_return for w in wins_in_regime]
        stats[regime] = {
            'n_windows': len(wins_in_regime),
            'mean_window_return': float(np.mean(rets)),
            'positive_pct': sum(1 for r in rets if r > 0) / len(rets),
            'profit_factor': pf,
            'n_trades': len(trades),
        }
    return stats


def load_frame(db, symbol, timeframe) -> pd.DataFrame:
    df = db.get_ohlcv_data(symbol, timeframe)
    df = df.set_index('timestamp').sort_index()
    return df[['open', 'high', 'low', 'close', 'volume']]


def stitch_equity(windows) -> pd.Series:
    parts = [w.equity_curve for w in windows if w.equity_curve is not None]
    if not parts:
        return pd.Series(dtype=float)
    curve = pd.concat(parts)
    return curve[~curve.index.duplicated(keep='last')].sort_index()


def compute_metrics(windows, timeframe, initial=10_000.0) -> dict:
    all_trades = [t for w in windows for t in w.trades]
    curve = stitch_equity(windows)

    final = windows[-1].equity_after if windows else initial
    total_return = final / initial - 1

    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = -sum(t['pnl'] for t in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win else 0.0)

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
    else:
        sharpe = sortino = max_dd = 0.0

    positive = sum(1 for w in windows if w.oos_return > 0)
    traded = [w for w in windows if w.oos_trades > 0]
    avg_bars = np.mean([t['bars_held'] for t in all_trades]) if all_trades else 0
    train_pfs = [w.train_pf for w in traded if np.isfinite(w.train_pf)]

    return {
        'total_return': total_return,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'win_rate': len(wins) / len(all_trades) if all_trades else 0.0,
        'n_trades': len(all_trades),
        'avg_duration_h': avg_bars * TF_MINUTES[timeframe] / 60,
        'fees': sum(w.oos_fees for w in windows),
        'n_windows': len(windows),
        'positive_windows': positive,
        'positive_window_pct': positive / len(windows) if windows else 0.0,
        'windows_traded': len(traded),
        'is_pf_mean': float(np.mean(train_pfs)) if train_pfs else 0.0,
        'final_equity': final,
        'curve': curve,
    }


def portfolio_metrics(results, windows_all, strategy, tf) -> dict:
    """Aggregate one strategy across all 3 symbols, equal capital weight —
    this is the unit Phase 3 actually deploys. Trades are pooled; the equity
    curve is the mean of per-symbol equity multiples (continuous equal-weight
    allocation); fees and windows aggregate."""
    per_symbol = [results.get((strategy, s, tf)) for s in SYMBOLS]
    per_symbol = [m for m in per_symbol if m is not None]
    if not per_symbol:
        return {}

    curves = [m['curve'] / 10_000.0 for m in per_symbol if not m['curve'].empty]
    if curves:
        aligned = pd.concat(curves, axis=1).sort_index().ffill().dropna(how='all')
        combined = aligned.mean(axis=1) * 10_000.0
    else:
        combined = pd.Series(dtype=float)

    all_windows = []
    for s in SYMBOLS:
        all_windows.extend(windows_all.get((strategy, s, tf), []))

    # window consistency at the portfolio level: sum same-numbered windows
    n_windows = max((w.window for w in all_windows), default=0)
    window_returns = []
    for wno in range(1, n_windows + 1):
        rets = [w.oos_return for w in all_windows if w.window == wno]
        window_returns.append(np.mean(rets) if rets else 0.0)

    all_trades = []
    for s in SYMBOLS:
        for w in windows_all.get((strategy, s, tf), []):
            all_trades.extend(w.trades)

    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = -sum(t['pnl'] for t in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win else 0.0)

    if not combined.empty and len(combined) > 10:
        daily = combined.resample('1D').last().dropna()
        rets = daily.pct_change().dropna()
        sharpe = float(np.sqrt(365) * rets.mean() / rets.std()) \
            if len(rets) > 2 and rets.std() > 0 else 0.0
        downside = rets[rets < 0]
        sortino = float(np.sqrt(365) * rets.mean() / downside.std()) \
            if len(downside) > 1 and downside.std() > 0 else 0.0
        run_max = combined.cummax()
        max_dd = float(((combined - run_max) / run_max).min())
        total_return = float(combined.iloc[-1] / combined.iloc[0] - 1)
    else:
        sharpe = sortino = max_dd = total_return = 0.0

    positive = sum(1 for r in window_returns if r > 0)
    avg_bars = np.mean([t['bars_held'] for t in all_trades]) if all_trades else 0

    return {
        'total_return': total_return,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'win_rate': len(wins) / len(all_trades) if all_trades else 0.0,
        'n_trades': len(all_trades),
        'avg_duration_h': avg_bars * TF_MINUTES[tf] / 60,
        'fees': sum(m['fees'] for m in per_symbol),
        'n_windows': n_windows,
        'positive_windows': positive,
        'positive_window_pct': positive / n_windows if n_windows else 0.0,
        'windows_traded': sum(m['windows_traded'] for m in per_symbol),
        'is_pf_mean': float(np.mean([m['is_pf_mean'] for m in per_symbol])),
        'final_equity': float(combined.iloc[-1]) if not combined.empty else 10_000.0,
        'curve': combined,
    }


def passes_gate(m) -> bool:
    return (m['profit_factor'] > GATE['profit_factor']
            and abs(m['max_drawdown']) < GATE['max_drawdown']
            and m['n_trades'] >= GATE['min_trades']
            and m['positive_window_pct'] >= GATE['positive_windows'])


def benchmark_returns(frames, windows_by_any) -> dict:
    """BTC buy-and-hold over the same stitched OOS calendar."""
    out = {}
    for tf in TIMEFRAMES:
        ref = next((w for key, w in windows_by_any.items()
                    if key[2] == tf and w), None)
        if ref is None:
            continue
        start, end = ref[0].test_start, ref[-1].test_end
        btc = frames[('BTC/USDT', tf)].loc[start:end, 'close']
        out[tf] = {'start': start, 'end': end,
                   'btc_return': float(btc.iloc[-1] / btc.iloc[0] - 1),
                   'btc_series': btc / btc.iloc[0]}
    return out


def make_chart(tag, strategy, tf, results, benchmarks, outdir: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for symbol in SYMBOLS:
        m = results.get((strategy, symbol, tf))
        if m and not m['curve'].empty:
            ax.plot(m['curve'].index, m['curve'] / 10_000.0,
                    label=f"{symbol} (ret {m['total_return']:+.1%})", lw=1.2)
    bench = benchmarks.get(tf)
    if bench is not None:
        ax.plot(bench['btc_series'].index, bench['btc_series'],
                label=f"Hold BTC ({bench['btc_return']:+.1%})",
                color='orange', ls='--', lw=1)
    ax.axhline(1.0, color='gray', lw=0.8, label='Cash (0%)')
    ax.set_title(f"{strategy} — {tf} — out-of-sample equity (walk-forward)")
    ax.set_ylabel('Equity multiple')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / f"{tag}_{strategy}_{tf.replace('m','min')}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_study(strategies, tag='baseline'):
    db = DatabaseManager('./data/trading_system.db')
    charts_dir = Path('docs/phase2_charts')
    charts_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            frames[(symbol, tf)] = load_frame(db, symbol, tf)
            logger.info(f"loaded {symbol} {tf}: {len(frames[(symbol, tf)])} bars "
                        f"({frames[(symbol, tf)].index[0]} -> "
                        f"{frames[(symbol, tf)].index[-1]})")

    results, windows_all = {}, {}
    started = time.time()
    for strategy in strategies:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                t0 = time.time()
                df = frames[(symbol, tf)]
                if strategy == 'ensemble':
                    windows, _ = wf.walk_forward_ensemble(df)
                else:
                    windows, _ = wf.walk_forward(df, strategy)
                m = compute_metrics(windows, tf)
                results[(strategy, symbol, tf)] = m
                windows_all[(strategy, symbol, tf)] = windows
                logger.info(
                    f"{strategy:20s} {symbol:9s} {tf:3s} | ret {m['total_return']:+7.2%} "
                    f"PF {m['profit_factor']:5.2f} DD {m['max_drawdown']:6.2%} "
                    f"trades {m['n_trades']:4d} win-w {m['positive_windows']}/"
                    f"{m['n_windows']} | {time.time()-t0:.1f}s")

    # Portfolio rows: the same strategy across all 3 symbols, equal weight —
    # the actual Phase 3 deployment unit.
    for strategy in strategies:
        for tf in TIMEFRAMES:
            pm = portfolio_metrics(results, windows_all, strategy, tf)
            if pm:
                results[(strategy, 'PORTFOLIO', tf)] = pm
                logger.info(
                    f"{strategy:20s} PORTFOLIO {tf:3s} | ret {pm['total_return']:+7.2%} "
                    f"PF {pm['profit_factor']:5.2f} DD {pm['max_drawdown']:6.2%} "
                    f"trades {pm['n_trades']:4d} win-w {pm['positive_windows']}/"
                    f"{pm['n_windows']}")

    benchmarks = benchmark_returns(frames, windows_all)
    charts = {}
    for strategy in strategies:
        for tf in TIMEFRAMES:
            charts[(strategy, tf)] = make_chart(
                tag, strategy, tf, results, benchmarks, charts_dir)

    logger.info(f"study '{tag}' complete in {(time.time()-started)/60:.1f} min")
    return results, windows_all, benchmarks, charts, frames


def summary_table(results) -> str:
    lines = ["| strategy | symbol | tf | OOS return | PF (fees) | Sharpe | Sortino "
             "| max DD | win rate | trades | avg hold | fees $ | +windows | gate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (strategy, symbol, tf), m in sorted(results.items()):
        gate = '**PASS**' if passes_gate(m) else 'fail'
        pf = f"{m['profit_factor']:.2f}" if np.isfinite(m['profit_factor']) else 'inf'
        lines.append(
            f"| {strategy} | {symbol} | {tf} | {m['total_return']:+.2%} | {pf} "
            f"| {m['sharpe']:.2f} | {m['sortino']:.2f} | {m['max_drawdown']:.2%} "
            f"| {m['win_rate']:.1%} | {m['n_trades']} | {m['avg_duration_h']:.1f}h "
            f"| {m['fees']:.0f} | {m['positive_windows']}/{m['n_windows']} | {gate} |")
    return '\n'.join(lines)


def window_table(windows, regimes=None) -> str:
    lines = ["| w | test period | regime | params | IS PF | OOS ret | OOS PF | trades | fees $ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for w in windows:
        params = json.dumps(w.params) if w.params else 'sat out'
        pf = f"{w.oos_pf:.2f}" if np.isfinite(w.oos_pf) else 'inf'
        regime = ''
        if regimes and w.window in regimes:
            r, btc_ret, _ = regimes[w.window]
            regime = f"{r} ({btc_ret:+.0%})"
        lines.append(
            f"| {w.window} | {w.test_start:%y-%m-%d}->{w.test_end:%m-%d} | {regime} | `{params}` "
            f"| {w.train_pf:.2f} | {w.oos_return:+.2%} | {pf} "
            f"| {w.oos_trades} | {w.oos_fees:.0f} |")
    return '\n'.join(lines)


def regime_table(results_regimes) -> str:
    lines = ["| strategy | symbol | tf | regime | windows | mean win ret | +win% | PF | trades |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (strategy, symbol, tf), stats in sorted(results_regimes.items()):
        for regime in ('bull', 'sideways', 'bear'):
            s = stats.get(regime)
            if not s:
                continue
            pf = f"{s['profit_factor']:.2f}" if np.isfinite(s['profit_factor']) else 'inf'
            lines.append(
                f"| {strategy} | {symbol} | {tf} | {regime} | {s['n_windows']} "
                f"| {s['mean_window_return']:+.2%} | {s['positive_pct']:.0%} "
                f"| {pf} | {s['n_trades']} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES)
    parser.add_argument('--tag', default='baseline')
    parser.add_argument('--flags', nargs='*', default=[],
                        choices=list(wf.FLAGS.keys()),
                        help='iteration feature flags to enable')
    parser.add_argument('--json-out', default=None,
                        help='dump metrics JSON for iteration tracking')
    parser.add_argument('--fee', type=float, default=0.001,
                        help='commission per side (0.001 taker, '
                             '0.00075 BNB discount)')
    args = parser.parse_args()

    for flag in args.flags:
        wf.FLAGS[flag] = True
    wf.COMMISSION = args.fee
    logger.info(f"run tag={args.tag} flags={wf.FLAGS} fee={args.fee}")

    results, windows_all, benchmarks, charts, frames = run_study(
        args.strategies, args.tag)

    # Regime classification per configuration (BTC return over each window)
    results_regimes = {}
    for key, windows in windows_all.items():
        strategy, symbol, tf = key
        regimes = classify_window_regimes(frames[('BTC/USDT', tf)], windows)
        results_regimes[key] = regime_breakdown(windows, regimes)

    print('\n===== SUMMARY =====')
    print(summary_table(results))
    passing = [k for k, m in results.items() if passes_gate(m)]
    print(f"\nGate passers: {passing if passing else 'NONE'}")
    for tf, b in benchmarks.items():
        print(f"benchmark hold-BTC over OOS period ({tf}): {b['btc_return']:+.2%}")

    print('\n===== REGIME BREAKDOWN =====')
    print(regime_table(results_regimes))

    if args.json_out:
        dump = {
            'fee': args.fee,
            'flags': dict(wf.FLAGS),
            'metrics': {f"{s}|{sym}|{tf}": {k: v for k, v in m.items()
                                            if k != 'curve'}
                        for (s, sym, tf), m in results.items()},
            'regimes': {f"{s}|{sym}|{tf}": stats
                        for (s, sym, tf), stats in results_regimes.items()},
        }
        Path(args.json_out).write_text(json.dumps(dump, indent=2, default=str))

    # persist per-window tables for the report appendix
    appendix = [f"# Per-window tables — tag `{args.tag}` "
                f"(fee {args.fee:.4%}/side, flags {wf.FLAGS})\n"]
    for key, windows in windows_all.items():
        strategy, symbol, tf = key
        regimes = classify_window_regimes(frames[('BTC/USDT', tf)], windows)
        appendix.append(f"\n#### {strategy} — {symbol} — {tf}\n")
        appendix.append(window_table(windows, regimes))
    Path(f'docs/phase2_windows_{args.tag}.md').write_text('\n'.join(appendix))
    print(f"\nPer-window tables -> docs/phase2_windows_{args.tag}.md")
    print(f"Charts -> docs/phase2_charts/{args.tag}_*.png")


if __name__ == '__main__':
    main()
