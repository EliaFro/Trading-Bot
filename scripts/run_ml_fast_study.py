#!/usr/bin/env python3
"""
Fast Lab Part C: the deep models' fair trial at 1m scale.

Pre-registered design (docs/FASTLAB_PLAN.md, locked before first run):
15 test windows of 20 days spread EVENLY across 36 months; per window,
train on the prior 60 days only (31-minute label purge inside a 1-day
embargo), validation = last 5 train days (deep early stopping, ensemble
weights, model selection — reported OOS is never used for decisions).

Models per window: LogReg / RandomForest / GradientBoosting on tabular
features + SmallLSTM / SmallCNN on 60-bar sequences. Trading rule
(pre-registered): long while ensemble argmax == UP at sampled bars;
simulated on the full 1m frame with per-symbol spread and the upgraded
execution model. Fee decomposition reported.
"""

import json
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.database import DatabaseManager
from src.backtesting import walkforward as wf
from src.backtesting.walkforward import SignalArrays, simulate
from src.ml import fast_dataset as fdset
from src.ml.deep_models import TORCH, train_deep, predict_proba_deep
from src.ml.walkforward_ml import MODEL_SPECS
from scripts.run_daily_momentum import classify_regimes, daily_frame
from scripts.run_fastlab_study import decompose

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
SPREADS = {s: json.loads(Path('docs/spread_measurements.json').read_text())
           ['symbols'][s]['used_spread'] for s in SYMBOLS}
N_WINDOWS = 15
TEST_DAYS = 20
TRAIN_DAYS = 60
VAL_DAYS = 5
FEE = 0.001


def metrics(y_true, y_pred):
    return {'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro'))}


def preds_to_signals(frame_1m, dates, pred, size=0.10):
    """Level rule on sampled bars: long while argmax==UP; other bars hold."""
    s = pd.Series(pred, index=dates)
    s = s[~s.index.duplicated(keep='last')].reindex(frame_1m.index)
    known = s.notna().to_numpy()
    is_up = (s == fdset.LABEL_UP).to_numpy()
    n = len(frame_1m)
    return SignalArrays(
        entry=is_up & known,
        exit_=(~is_up) & known,
        confidence=np.where(is_up & known, 0.99, 0.0),
        size=np.full(n, size),
        stop_loss=np.full(n, np.nan),
        take_profit=np.full(n, np.nan))


def main():
    # Pre-registered kill rule: strategy search at this horizon closes
    # permanently after 2026-08-07 (src/trading/kill_rule.py). No bypass.
    from src.trading.kill_rule import assert_search_allowed
    assert_search_allowed()

    t0 = time.time()
    db = DatabaseManager('./data/trading_system.db')
    frames_1m = {}
    for symbol in SYMBOLS:
        df = db.get_ohlcv_data(symbol, '1m')
        frames_1m[symbol] = df.set_index('timestamp').sort_index()[
            ['open', 'high', 'low', 'close', 'volume']]
        print(f"{symbol}: {len(frames_1m[symbol]):,} 1m bars")

    start = max(f.index[0] for f in frames_1m.values())
    end = min(f.index[-1] for f in frames_1m.values())
    first_test = start + timedelta(days=TRAIN_DAYS + 2)
    last_test = end - timedelta(days=TEST_DAYS + 1)
    test_starts = pd.date_range(first_test, last_test, periods=N_WINDOWS)
    print(f"{N_WINDOWS} stratified windows, {test_starts[0]:%Y-%m-%d} -> "
          f"{test_starts[-1]:%Y-%m-%d} | torch={TORCH}")

    daily_btc = daily_frame(db, 'BTC/USDT')
    windows_cal = [(ts, ts + timedelta(days=TEST_DAYS)) for ts in test_starts]
    regimes = classify_regimes(daily_btc, windows_cal)
    print(f"regimes: {Counter(regimes)}")

    model_names = list(MODEL_SPECS) + (['lstm', 'cnn'] if TORCH else []) \
        + ['ensemble']
    clf_acc = {m: {'is': [], 'oos': []} for m in model_names}
    trade_results = {fam: {'trades': [], 'window_rets': []}
                     for fam in ('ensemble', 'best_tree', 'best_deep')}

    for w_no, test_start in enumerate(test_starts, 1):
        w_t0 = time.time()
        test_end = test_start + timedelta(days=TEST_DAYS)
        train_start = test_start - timedelta(days=TRAIN_DAYS + 1)
        val_start = test_start - timedelta(days=VAL_DAYS + 1)
        embargo_end = test_start                    # 1-day embargo inside

        slice_start = train_start - timedelta(days=2)
        slices = {s: f.loc[slice_start:test_end + timedelta(hours=2)]
                  for s, f in frames_1m.items()}
        X_tab, X_seq, meta = fdset.assemble_fast_panel(slices, SPREADS)

        dates = meta['date']
        is_fit = (dates >= train_start) & (dates < val_start)
        is_val = (dates >= val_start) & (dates < embargo_end - timedelta(days=1))
        is_test = (dates >= test_start) & (dates < test_end)
        assert meta.loc[is_fit | is_val, 'date'].max() \
            + timedelta(minutes=fdset.HORIZON + 1) < test_start, "purge violated"
        if is_fit.sum() < 5000 or is_test.sum() < 1000:
            print(f"w{w_no}: insufficient data, skipped")
            continue

        y = meta['label'].to_numpy()
        fit_i, val_i, test_i = (np.where(m)[0] for m in (is_fit, is_val, is_test))

        # tabular models
        scaler = StandardScaler().fit(X_tab[fit_i])
        Xf, Xv, Xe = (scaler.transform(X_tab[i]) for i in (fit_i, val_i, test_i))
        probas_val, probas_test, val_f1 = {}, {}, {}
        for name, make in MODEL_SPECS.items():
            model = make()
            model.fit(Xf, y[fit_i])
            probas_val[name] = model.predict_proba(Xv)
            probas_test[name] = model.predict_proba(Xe)
            val_f1[name] = f1_score(y[val_i], probas_val[name].argmax(1),
                                    average='macro')
            clf_acc[name]['is'].append(
                metrics(y[fit_i][:20000], model.predict(Xf[:20000]))['balanced_accuracy'])
            clf_acc[name]['oos'].append(
                metrics(y[test_i], probas_test[name].argmax(1))['balanced_accuracy'])

        # deep models (channel-standardized on train stats)
        if TORCH:
            mu = X_seq[fit_i].mean(axis=(0, 1), keepdims=True)
            sd = X_seq[fit_i].std(axis=(0, 1), keepdims=True) + 1e-8
            Sf, Sv, Se = ((X_seq[i] - mu) / sd for i in (fit_i, val_i, test_i))
            for name in ('lstm', 'cnn'):
                model, info = train_deep(name, Sf, y[fit_i], Sv, y[val_i])
                pv = predict_proba_deep(model, Sv)
                pe = predict_proba_deep(model, Se)
                pf_ = predict_proba_deep(model, Sf[:20000])
                probas_val[name], probas_test[name] = pv, pe
                val_f1[name] = f1_score(y[val_i], pv.argmax(1), average='macro')
                clf_acc[name]['is'].append(
                    metrics(y[fit_i][:20000], pf_.argmax(1))['balanced_accuracy'])
                clf_acc[name]['oos'].append(
                    metrics(y[test_i], pe.argmax(1))['balanced_accuracy'])

        # ensemble: weights from validation only
        weights = {m: max(f - 1 / 3, 0.0) for m, f in val_f1.items()}
        total = sum(weights.values())
        weights = ({m: w / total for m, w in weights.items()} if total > 0
                   else {m: 1 / len(val_f1) for m in val_f1})
        ens_test = sum(probas_test[m] * w for m, w in weights.items())
        clf_acc['ensemble']['oos'].append(
            metrics(y[test_i], ens_test.argmax(1))['balanced_accuracy'])
        # ensemble IS proxy: weighted mean of member IS scores (members'
        # exact IS probas aren't retained; this slightly UNDERSTATES the
        # ensemble's in-sample fit, i.e. understates the gap — noted)
        clf_acc['ensemble']['is'].append(
            float(sum(clf_acc[m]['is'][-1] * weights[m] for m in val_f1)))

        # trading: ensemble / best tree / best deep on this window
        best_tree = max((m for m in MODEL_SPECS), key=lambda m: val_f1[m])
        fams = {'ensemble': ens_test.argmax(1),
                'best_tree': probas_test[best_tree].argmax(1)}
        if TORCH:
            best_deep = max(('lstm', 'cnn'), key=lambda m: val_f1[m])
            fams['best_deep'] = probas_test[best_deep].argmax(1)

        test_symbols = meta['symbol'].to_numpy()[test_i]
        test_dates = meta['date'].to_numpy()[test_i]
        for fam, pred in fams.items():
            wf.COMMISSION = FEE
            wf.MAX_POSITION_SIZE = 0.10 + 1e-9
            curves = []
            for symbol in SYMBOLS:
                sel = test_symbols == symbol       # positions within test rows
                if not sel.any():
                    continue
                frame = frames_1m[symbol].loc[test_start:test_end]
                sig = preds_to_signals(frame, pd.DatetimeIndex(test_dates[sel]),
                                       pred[sel])
                sim = simulate(frame, sig, 0, len(frame),
                               initial_equity=10_000.0,
                               spread=SPREADS[symbol])
                trade_results[fam]['trades'].extend(sim.trades)
                if sim.equity is not None and len(sim.equity) > 1:
                    curves.append(float(sim.equity.iloc[-1] / sim.equity.iloc[0] - 1))
            trade_results[fam]['window_rets'].append(
                float(np.mean(curves)) if curves else 0.0)

        oos_e = clf_acc['ensemble']['oos'][-1]
        print(f"w{w_no:02d} {test_start:%Y-%m-%d} [{regimes[w_no-1]:8s}] "
              f"ens OOS {oos_e:.3f} | "
              + " ".join(f"{m[:4]}={clf_acc[m]['oos'][-1]:.3f}"
                         for m in model_names if m != 'ensemble')
              + f" | {time.time()-w_t0:.0f}s")

    # aggregate + dump
    out = {'plan': 'docs/FASTLAB_PLAN.md', 'n_windows': N_WINDOWS,
           'regimes': dict(Counter(regimes)), 'spreads': SPREADS,
           'classification': {}, 'trading': {}}
    for m in model_names:
        if clf_acc[m]['oos']:
            is_m, oos_m = np.mean(clf_acc[m]['is']), np.mean(clf_acc[m]['oos'])
            out['classification'][m] = {
                'is_bal_acc': float(is_m), 'oos_bal_acc': float(oos_m),
                'gap': float(is_m - oos_m),
                'oos_per_window': [float(x) for x in clf_acc[m]['oos']]}
    for fam, res in trade_results.items():
        trades = res['trades']
        wr = np.array(res['window_rets'])
        wins = sum(1 for t in trades if t['pnl'] > 0)
        gw = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gl = -sum(t['pnl'] for t in trades if t['pnl'] <= 0)
        net_per_trade = [t['pnl_pct'] for t in trades]
        t_trades = (np.mean(net_per_trade) / np.std(net_per_trade)
                    * np.sqrt(len(net_per_trade))) if len(net_per_trade) > 2 else 0
        out['trading'][fam] = {
            'n_trades': len(trades),
            'profit_factor': gw / gl if gl else 0.0,
            'win_rate': wins / len(trades) if trades else 0.0,
            'mean_window_return': float(wr.mean()) if len(wr) else 0.0,
            'positive_windows': int((wr > 0).sum()),
            't_stat_per_trade': float(t_trades),
            'decomposition': decompose(trades),
        }
    Path('docs/fastlab_partC_metrics.json').write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\ndone in {(time.time()-t0)/60:.1f} min "
          f"-> docs/fastlab_partC_metrics.json")


if __name__ == '__main__':
    main()
