#!/usr/bin/env python3
"""
Stage 1 milestone: walk-forward ML evaluation on 36 months of daily data.

Produces docs/ML_RESULTS.md — verdict first line — plus charts and JSON.
Trading rule (pre-registered): long a symbol while the ensemble's argmax
class is UP; flat otherwise. Executed through the identical Phase 2
simulator at both fee tiers and both sizing variants. Benchmarks (hold-BTC,
hold-basket, TSMOM-60d) recomputed on the ML's exact OOS calendar.
"""

import json
import sys
from collections import Counter
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
from src.ml.dataset import LABEL_UP, assemble_panel
from src.ml import walkforward_ml as wfml
from scripts.run_daily_momentum import (SYMBOLS, daily_frame, tsmom_signals,
                                        classify_regimes)

FEES = {'taker': 0.001, 'bnb': 0.00075}
SIZINGS = {'rail_10pct': 0.10, 'equal_weight_33pct': 1.0 / 3}
TEST_DAYS = wfml.TEST_DAYS


def predictions_to_signals(df: pd.DataFrame, preds: pd.DataFrame,
                           symbol: str, size: float) -> SignalArrays:
    """Level-based rule: long while ensemble argmax == UP (daily decisions)."""
    p = preds[preds['symbol'] == symbol].set_index('date')['pred']
    p = p.reindex(df.index)                      # NaN outside OOS
    is_up = (p == LABEL_UP).to_numpy()
    known = p.notna().to_numpy()
    n = len(df)
    return SignalArrays(
        entry=is_up & known,
        exit_=(~is_up) & known,
        confidence=np.where(is_up & known, 0.99, 0.0),
        size=np.full(n, size),
        stop_loss=np.full(n, np.nan),
        take_profit=np.full(n, np.nan),
    )


def portfolio_curve(sims: dict) -> pd.Series:
    curves = [s.equity / s.equity.iloc[0] for s in sims.values()
              if s.equity is not None and len(s.equity) > 1]
    if not curves:
        return pd.Series(dtype=float)
    return pd.concat(curves, axis=1).sort_index().ffill().mean(axis=1) * 10_000.0


def slice_windows(curve: pd.Series, windows) -> list:
    out = []
    for w_start, w_end in windows:
        seg = curve.loc[w_start:w_end]
        out.append(float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) > 1 else 0.0)
    return out


def strategy_metrics(sims: dict, curve: pd.Series, windows, regimes) -> dict:
    trades = [t for s in sims.values() for t in s.trades]
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gw = sum(t['pnl'] for t in wins)
    gl = -sum(t['pnl'] for t in losses)
    pf = gw / gl if gl > 0 else (float('inf') if gw else 0.0)
    fees = sum(s.fees for s in sims.values())

    if curve.empty or len(curve) < 10:
        return {}
    daily = curve.resample('1D').last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(np.sqrt(365) * rets.mean() / rets.std()) \
        if rets.std() > 0 else 0.0
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    total = float(curve.iloc[-1] / curve.iloc[0] - 1)
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    wr = slice_windows(curve, windows)
    positive = sum(1 for r in wr if r > 0)
    t_stat = float(np.mean(wr) / np.std(wr) * np.sqrt(len(wr))) \
        if len(wr) > 2 and np.std(wr) > 0 else 0.0

    regime_stats = {}
    for regime in ('bull', 'sideways', 'bear'):
        rs = [r for r, reg in zip(wr, regimes) if reg == regime]
        if rs:
            regime_stats[regime] = {
                'n': len(rs),
                'total': float(np.prod([1 + x for x in rs]) - 1),
                'positive_pct': sum(1 for x in rs if x > 0) / len(rs)}

    gross = gw - gl + fees
    return {
        'total_return': total,
        'cagr': (1 + total) ** (1 / years) - 1 if years > 0 else 0.0,
        'profit_factor': pf, 'sharpe': sharpe, 'max_drawdown': dd,
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'n_trades': len(trades),
        'avg_hold_days': float(np.mean([t['bars_held'] for t in trades]))
        if trades else 0.0,
        'fees': fees,
        'fees_pct_of_gross': fees / abs(gross) if gross else 0.0,
        'positive_windows': positive, 'n_windows': len(wr),
        't_stat': t_stat, 'regimes': regime_stats,
    }


def main():
    db = DatabaseManager('./data/trading_system.db')
    frames = {s: daily_frame(db, s) for s in SYMBOLS}

    print("building panel...")
    X, meta = assemble_panel(frames)

    print("walk-forward training...")
    reports = wfml.run_walkforward_ml(X, meta)
    preds = wfml.stitch_predictions(reports)
    stability = wfml.importance_stability(reports)

    oos_start = reports[0].test_start
    oos_end = reports[-1].test_end
    windows = [(r.test_start, r.test_end) for r in reports]
    regimes = classify_regimes(frames['BTC/USDT'], windows)
    print(f"{len(reports)} OOS windows {oos_start:%Y-%m-%d} -> "
          f"{oos_end:%Y-%m-%d}, regimes {Counter(regimes)}")

    # classification aggregates + IS/OOS gap
    clf = {}
    for model in list(wfml.MODEL_SPECS) + ['ensemble']:
        is_bal = [r.model_metrics[model]['is']['balanced_accuracy'] for r in reports]
        oos_bal = [r.model_metrics[model]['oos']['balanced_accuracy'] for r in reports]
        oos_f1 = [r.model_metrics[model]['oos']['macro_f1'] for r in reports]
        clf[model] = {
            'is_balanced_accuracy': float(np.mean(is_bal)),
            'oos_balanced_accuracy': float(np.mean(oos_bal)),
            'oos_macro_f1': float(np.mean(oos_f1)),
            'is_oos_gap': float(np.mean(is_bal) - np.mean(oos_bal)),
        }

    # trading evaluation: ML rule + benchmarks on the SAME OOS calendar
    results = {}
    for fee_name, fee in FEES.items():
        wf.COMMISSION = fee
        for sizing_name, size in SIZINGS.items():
            wf.MAX_POSITION_SIZE = size + 1e-9

            def run(signal_builder):
                sims = {}
                for symbol in SYMBOLS:
                    df = frames[symbol]
                    start_idx = int(df.index.searchsorted(oos_start))
                    end_idx = int(df.index.searchsorted(oos_end))
                    sims[symbol] = simulate(df, signal_builder(df, symbol),
                                            start_idx=start_idx, end_idx=end_idx,
                                            initial_equity=10_000.0)
                curve = portfolio_curve(sims)
                return strategy_metrics(sims, curve, windows, regimes), curve

            ml_m, ml_curve = run(lambda df, s: predictions_to_signals(df, preds, s, size))
            ts_m, ts_curve = run(lambda df, s: tsmom_signals(df, 60, size))
            key = f"{fee_name}|{sizing_name}"
            results[key] = {'ml': ml_m, 'tsmom60': ts_m}
            if fee_name == 'taker':
                results[key]['_curves'] = {'ml': ml_curve, 'tsmom': ts_curve}
            print(f"{key:28s} ML ret {ml_m['total_return']:+7.1%} PF "
                  f"{ml_m['profit_factor']:5.2f} DD {ml_m['max_drawdown']:6.1%} "
                  f"n={ml_m['n_trades']:4d} | TSMOM60 ret "
                  f"{ts_m['total_return']:+7.1%} PF {ts_m['profit_factor']:5.2f}")

    # buy-and-hold benchmarks on the same span
    bench = {}
    for s in SYMBOLS:
        seg = frames[s].loc[oos_start:oos_end, 'close']
        bench[s] = float(seg.iloc[-1] / seg.iloc[0] - 1)
    basket_curves = [frames[s].loc[oos_start:oos_end, 'close'] for s in SYMBOLS]
    basket = pd.concat([c / c.iloc[0] for c in basket_curves], axis=1).mean(axis=1)
    bench['basket'] = float(basket.iloc[-1] - 1)
    bench['basket_max_dd'] = float(((basket - basket.cummax()) / basket.cummax()).min())

    # persist JSON (drop curves)
    dump = {
        'oos_span': f"{oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d}",
        'n_windows': len(reports),
        'regime_counts': dict(Counter(regimes)),
        'classification': clf,
        'benchmarks': bench,
        'importance': {
            'top20': stability['mean_importance'].head(20).round(4).to_dict(),
            'group_share': stability['group_share'],
            'rank_stability_mean': stability['rank_stability_mean'],
            'rank_stability_min': stability['rank_stability_min'],
        },
        'trading': {k: {kk: vv for kk, vv in v.items() if kk != '_curves'}
                    for k, v in results.items()},
        'per_window': [
            {'window': r.window, 'test_start': str(r.test_start.date()),
             'regime': regimes[i],
             'ens_oos_bal_acc': r.model_metrics['ensemble']['oos']['balanced_accuracy'],
             'ens_is_bal_acc': r.model_metrics['ensemble']['is']['balanced_accuracy'],
             'ens_oos_f1': r.model_metrics['ensemble']['oos']['macro_f1'],
             'weights': r.ensemble_weights}
            for i, r in enumerate(reports)],
    }
    Path('docs/ml_metrics.json').write_text(json.dumps(dump, indent=2, default=str))
    print("metrics -> docs/ml_metrics.json")

    # charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    curves = results['taker|equal_weight_33pct']['_curves']
    axes[0].plot(curves['ml'].index, curves['ml'] / 10_000, label='ML ensemble', lw=1.4)
    axes[0].plot(curves['tsmom'].index, curves['tsmom'] / 10_000, label='TSMOM-60d', lw=1.2)
    btc = frames['BTC/USDT'].loc[oos_start:oos_end, 'close']
    axes[0].plot(btc.index, btc / btc.iloc[0], ls='--', color='orange', lw=1,
                 label=f"Hold BTC ({bench['BTC/USDT']:+.0%})")
    axes[0].axhline(1, color='gray', lw=0.8, label='Cash')
    axes[0].set_title('OOS equity — equal-weight sizing, fee 0.10%')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    w_ids = [r.window for r in reports]
    is_acc = [r.model_metrics['ensemble']['is']['balanced_accuracy'] for r in reports]
    oos_acc = [r.model_metrics['ensemble']['oos']['balanced_accuracy'] for r in reports]
    axes[1].plot(w_ids, is_acc, label='in-sample', lw=1, color='gray')
    axes[1].plot(w_ids, oos_acc, label='out-of-sample', lw=1.4, color='tab:red')
    axes[1].axhline(1 / 3, color='black', ls=':', label='3-class chance')
    axes[1].fill_between(w_ids, oos_acc, is_acc, alpha=0.15, color='red',
                         label='IS-OOS gap (overfitting)')
    axes[1].set_title('Learning curve: balanced accuracy per retrain')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    top = stability['mean_importance'].head(15)[::-1]
    err = stability['std_importance'][top.index]
    axes[2].barh(top.index, top.values, xerr=err.values, color='tab:blue')
    axes[2].set_title('Feature importance (mean ± std across retrains)')
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('docs/phase2_charts/ml_study.png', dpi=110)
    print("chart -> docs/phase2_charts/ml_study.png")


if __name__ == '__main__':
    main()
