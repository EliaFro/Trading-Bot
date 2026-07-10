"""
Walk-forward validation engine for Phase 2.

Fast vectorized backtester that reproduces the paper engine's execution model
EXACTLY, with conservative resolution of every intrabar ambiguity:

  * signals compute on CLOSED bars, orders execute on the NEXT bar
  * entry = marketable LIMIT at signal_close * (1 + slippage_tolerance);
    fills only if the next bar trades at/through the limit, else the signal
    lapses unfilled (order_timeout 90s < one bar on 5m/15m)
  * fill price = next open * (1 + slippage_rate), capped at the limit
  * 0.1% commission per side, 0.05% slippage per side
  * stop-loss beats take-profit when both are struck in the same bar
  * gap through a stop fills at the (worse) open, not the stop price
  * exit signals (death cross, RSI overbought, breakdown) execute at the
    NEXT bar's open with sell-side slippage
  * engine-level confidence gate (min_confidence) applied identically
  * position sizing = the strategies' Kelly-fraction logic, capped at 10%

Strategy signal generators are 1:1 ports of the implementations in
src/backtesting/backtest_module.py (same formulas, same parameters), written
as vector operations so a full year backtests in milliseconds.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COMMISSION = 0.001          # 0.1% per side (Binance spot)
SLIPPAGE = 0.0005           # 0.05% against the fill
LIMIT_TOLERANCE = 0.001     # marketable limit offset, as in the paper engine
MIN_CONFIDENCE = 0.55       # engine gate (config/trading.yaml strategies.min_confidence)
MAX_POSITION_SIZE = 0.10    # hard cap per position
RISK_PER_TRADE = 0.01       # risk budget for calibrated sizing (1% of equity)

# ── Iteration feature flags (set by the study driver; logged in the report) ──
# calibrated      iter1: probability-scaled confidence + fixed-fractional
#                 risk sizing (the original formulas produce ~0 confidence
#                 and dust-sized positions — see PHASE2_RESULTS.md baseline)
# regime_filter   iter2: DIRECTIONAL regime gate — long-only strategies never
#                 fight the primary trend: trend strategies need ADX>25 AND
#                 close>SMA200; mean reversion needs a range (ADX<25) AND
#                 close>SMA200
# min_edge        iter3: skip signals whose take-profit distance is under
#                 3x the round-trip cost
# mr_completion   iter3: mean-reversion completion — RSI take-profit at the
#                 rolling mean (SMA20) instead of a fixed 1.5% cap, and
#                 entries require 5 points deeper oversold
# direction_only  iter4: trend strategies gate on direction (close>SMA200)
#                 without demanding trend maturity (ADX) at signal time
FLAGS = {
    'calibrated': False,
    'regime_filter': False,
    'min_edge': False,
    'mr_completion': False,
    'mr_deep': False,        # iter3 experiment, reverted by evidence
    'direction_only': False, # iter4 experiment, reverted by evidence
    'passive_entry': False,  # iter5: MR entries rest 0.3% below signal close
}
PASSIVE_OFFSET = 0.003


def _uptrend(c: Dict[str, np.ndarray]) -> np.ndarray:
    sma200 = pd.Series(c['close']).rolling(200).mean().to_numpy()
    return np.nan_to_num(c['close'] > sma200).astype(bool)


def _risk_size(close: np.ndarray, stop_loss: np.ndarray) -> np.ndarray:
    """Fixed-fractional sizing: risk RISK_PER_TRADE of equity per trade,
    based on stop distance, capped at the hard position limit."""
    stop_pct = np.where(close > 0, (close - stop_loss) / close, np.nan)
    stop_pct = np.where(stop_pct > 1e-4, stop_pct, np.nan)
    return np.minimum(np.nan_to_num(RISK_PER_TRADE / stop_pct),
                      MAX_POSITION_SIZE)


def _adx14(c: Dict[str, np.ndarray]) -> np.ndarray:
    from src.utils.indicators import ADX
    return np.nan_to_num(ADX(c['high'], c['low'], c['close'], 14))


def _edge_ok(close: np.ndarray, take_profit: np.ndarray) -> np.ndarray:
    """Expected gross move to TP must exceed 3x round-trip cost."""
    round_trip = 2 * (COMMISSION + SLIPPAGE)          # 0.3%
    tp_dist = np.where(close > 0, (take_profit - close) / close, 0.0)
    return tp_dist >= 3 * round_trip


# ── Rolling helpers (strictly trailing, no lookahead) ───────────────────────

def rolling_slope(y: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope of y over each trailing `window` bars (vectorized
    via convolution). out[i] uses y[i-window+1 .. i]."""
    n = len(y)
    out = np.full(n, np.nan)
    if n < window:
        return out
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    weights = (x - x_mean) / denom               # slope = Σ w_j * y_j
    # convolve with reversed weights -> trailing dot product
    conv = np.convolve(y, weights[::-1], mode='valid')   # length n-window+1
    out[window - 1:] = conv
    return out


def wilder_rsi(close: np.ndarray, period: int) -> np.ndarray:
    from src.utils.indicators import RSI
    return RSI(close, period)


def atr_series(high, low, close, period: int = 14) -> np.ndarray:
    from src.utils.indicators import ATR
    return ATR(high, low, close, period)


# ── Signal container ─────────────────────────────────────────────────────────

@dataclass
class SignalArrays:
    """Per-bar vectors. entry[i]/exit_[i] refer to decisions at the CLOSE of
    bar i; execution happens on bar i+1."""
    entry: np.ndarray          # bool
    exit_: np.ndarray          # bool (signal-based exit while in position)
    confidence: np.ndarray     # float, for the engine gate + ensemble merge
    size: np.ndarray           # position size fraction (Kelly-capped)
    stop_loss: np.ndarray      # absolute price at signal bar (NaN = none)
    take_profit: np.ndarray
    passive_offset: float = 0.0   # >0: entry rests this far BELOW signal close


# ── Strategy signal generators (vector ports of backtest_module) ────────────

def _common(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    close = df['close'].to_numpy(float)
    return {
        'open': df['open'].to_numpy(float),
        'high': df['high'].to_numpy(float),
        'low': df['low'].to_numpy(float),
        'close': close,
        'volume': df['volume'].to_numpy(float),
        'vol20': pd.Series(close).pct_change().rolling(20).std().to_numpy(),
        'volume_sma20': df['volume'].rolling(20).mean().to_numpy(),
    }


def signals_ma_crossover(df: pd.DataFrame, params: Dict) -> SignalArrays:
    c = _common(df)
    fast_p = params.get('fast_period', 20)
    slow_p = params.get('slow_period', 50)
    use_volume = params.get('use_volume_filter', True)

    close = pd.Series(c['close'])
    fast = close.rolling(fast_p).mean().to_numpy()
    slow = close.rolling(slow_p).mean().to_numpy()

    prev_fast = np.roll(fast, 1)
    prev_slow = np.roll(slow, 1)
    prev_fast[0] = np.nan
    prev_slow[0] = np.nan

    golden = (prev_fast <= prev_slow) & (fast > slow)
    death = (prev_fast >= prev_slow) & (fast < slow)
    if use_volume:
        golden &= c['volume'] > c['volume_sma20']

    strength = np.abs(fast - slow) / np.where(slow > 0, slow, np.nan)
    stop_loss = c['close'] * 0.98
    take_profit = c['close'] * 1.02

    if FLAGS['calibrated']:
        # iter1: strength at a fresh cross is ~0 by construction; scale it so
        # a 0.4%+ MA separation maps to full confidence, and size by risk.
        confidence = 0.55 + 0.35 * np.minimum(
            np.nan_to_num(strength) / 0.004, 1.0)
        size = _risk_size(c['close'], stop_loss)
    else:
        confidence = np.minimum(strength * 10, 0.9)
        kelly = strength * 0.25
        size = np.minimum(kelly / (1 + np.nan_to_num(c['vol20'])),
                          MAX_POSITION_SIZE)

    entry = np.nan_to_num(golden).astype(bool)
    if FLAGS['regime_filter']:
        if FLAGS['direction_only']:
            # iter4: crossovers fire at trend inception when ADX is still
            # low — gate on direction only
            entry &= _uptrend(c)
        else:
            entry &= (_adx14(c) > 25) & _uptrend(c)
    if FLAGS['min_edge']:
        entry &= _edge_ok(c['close'], take_profit)

    return SignalArrays(
        entry=entry,
        exit_=np.nan_to_num(death).astype(bool),
        confidence=np.nan_to_num(confidence),
        size=np.nan_to_num(size),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def signals_rsi_mean_reversion(df: pd.DataFrame, params: Dict) -> SignalArrays:
    c = _common(df)
    period = params.get('rsi_period', 14)
    oversold = params.get('oversold_threshold', 30)
    overbought = params.get('overbought_threshold', 70)
    use_divergence = params.get('use_divergence', True)

    rsi = wilder_rsi(c['close'], period)

    # Divergence: trailing 20-bar slopes of price and RSI
    divergence = np.zeros(len(rsi))
    if use_divergence:
        ps = rolling_slope(c['close'], 20)
        rs = rolling_slope(np.nan_to_num(rsi), 20)
        divergence = np.where((ps < 0) & (rs > 0), 1.0,
                              np.where((ps > 0) & (rs < 0), -1.0, 0.0))

    # Historical reversal probability — trailing only: an oversold bar at i
    # resolves at i+5, so it may inform decisions from i+5 onward.
    close_s = pd.Series(c['close'])
    fwd5 = close_s.shift(-5) / close_s - 1
    occurred = pd.Series((rsi < oversold).astype(float))
    succeeded = (occurred > 0) & (fwd5 > 0)
    occ_known = occurred.shift(5).rolling(1000, min_periods=30).sum()
    succ_known = succeeded.astype(float).shift(5).rolling(1000, min_periods=30).sum()
    prob = (succ_known / occ_known).fillna(0.5).clip(0, 1).to_numpy()

    strength = np.where(rsi < oversold,
                        (oversold - rsi) / oversold * (1 + divergence * 0.5), 0.0)
    stop_loss = c['close'] * 0.97
    take_profit = c['close'] * 1.015

    if FLAGS['mr_completion']:
        # iter3: target the mean itself (SMA20), floor 0.5% above entry so a
        # trade can never target less than ~1.7x its round-trip cost
        sma20 = close_s.rolling(20).mean().to_numpy()
        take_profit = np.maximum(np.nan_to_num(sma20, nan=0.0),
                                 c['close'] * 1.005)

    if FLAGS['calibrated']:
        # iter1: deeper oversold + better historical reversal odds -> higher
        # confidence; shallow dips (RSI ~29) stay under the 0.55 gate.
        confidence = np.where(
            strength > 0,
            0.5 + 0.45 * np.minimum(strength * (0.5 + prob) * 3.0, 1.0),
            0.0)
        size = _risk_size(c['close'], stop_loss)
    else:
        confidence = np.nan_to_num(strength * prob)
        kelly = confidence * 0.25
        size = np.minimum(kelly / (1 + np.nan_to_num(c['vol20'])),
                          MAX_POSITION_SIZE)

    # iter3 tried entry deepening (threshold-5): it collapsed sample size and
    # hurt BTC/ETH — reverted in iter4. Completion exits proved their worth.
    entry_threshold = oversold - 5 if FLAGS.get('mr_deep') else oversold
    entry = np.nan_to_num(rsi < entry_threshold).astype(bool)
    if FLAGS['regime_filter']:
        # long-only mean reversion: needs a range AND the primary trend up
        entry &= (_adx14(c) < 25) & _uptrend(c)
    if FLAGS['min_edge']:
        entry &= _edge_ok(c['close'], take_profit)

    return SignalArrays(
        entry=entry,
        exit_=np.nan_to_num(rsi > overbought).astype(bool),
        confidence=np.nan_to_num(confidence),
        size=np.nan_to_num(size),
        stop_loss=stop_loss,
        take_profit=take_profit,
        passive_offset=PASSIVE_OFFSET if FLAGS['passive_entry'] else 0.0,
    )


def signals_breakout(df: pd.DataFrame, params: Dict) -> SignalArrays:
    c = _common(df)
    lookback = params.get('lookback_period', 20)
    vol_mult = params.get('volume_multiplier', 1.5)
    atr_mult = params.get('atr_multiplier', 2.0)

    high_s, low_s = pd.Series(c['high']), pd.Series(c['low'])
    resistance = high_s.rolling(lookback).max().shift(1).to_numpy()
    support = low_s.rolling(lookback).min().shift(1).to_numpy()
    atr = atr_series(c['high'], c['low'], c['close'], 14)

    volume_ok = c['volume'] > c['volume_sma20'] * vol_mult
    entry = (c['close'] > resistance) & volume_ok
    exit_ = (c['close'] < support) & volume_ok          # breakdown -> exit

    strength = np.where(resistance > 0,
                        (c['close'] - resistance) / resistance, 0.0)
    stop_loss = c['close'] - atr * atr_mult
    take_profit = c['close'] + atr * atr_mult * 2

    if FLAGS['calibrated']:
        # iter1: 0.4%+ clearance above resistance maps to full confidence;
        # size by risk against the ATR stop.
        confidence = 0.55 + 0.35 * np.minimum(
            np.maximum(strength, 0) / 0.004, 1.0)
        confidence = np.where(entry, confidence, 0.0)
        size = _risk_size(c['close'], stop_loss)
    else:
        confidence = np.minimum(np.maximum(strength, 0) * 5 + 0.5, 0.9)
        confidence = np.where(entry, confidence, 0.0)
        vol_atr = np.where(c['close'] > 0, atr / c['close'], np.nan)
        kelly = np.maximum(strength, 0) * 2 * 0.25
        size = np.minimum(kelly / (1 + np.nan_to_num(vol_atr)),
                          MAX_POSITION_SIZE)

    entry = np.nan_to_num(entry).astype(bool)
    if FLAGS['regime_filter']:
        if FLAGS['direction_only']:
            # iter4: breakouts occur at range END when ADX is still low —
            # gate on direction only
            entry &= _uptrend(c)
        else:
            entry &= (_adx14(c) > 25) & _uptrend(c)
    if FLAGS['min_edge']:
        entry &= _edge_ok(c['close'], take_profit)

    return SignalArrays(
        entry=entry,
        exit_=np.nan_to_num(exit_).astype(bool),
        confidence=np.nan_to_num(confidence),
        size=np.nan_to_num(size),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def signals_ensemble(df: pd.DataFrame, params: Dict) -> SignalArrays:
    """Confidence-weighted vote across the three strategies — the vector port
    of EnsembleModel._merge (same agreement scaling, same contributor pick)."""
    subs = {
        'ma_crossover': signals_ma_crossover(df, params.get('ma_crossover', {})),
        'rsi_mean_reversion': signals_rsi_mean_reversion(
            df, params.get('rsi_mean_reversion', {})),
        'breakout': signals_breakout(df, params.get('breakout', {})),
    }
    n = len(df)
    n_strats = len(subs)

    buy_conf = np.zeros((n_strats, n))
    sell_flag = np.zeros((n_strats, n))
    for k, (name, s) in enumerate(subs.items()):
        buy_conf[k] = np.where(s.entry, np.maximum(s.confidence, 1e-9), 0.0)
        sell_flag[k] = s.exit_.astype(float)

    buy_score = buy_conf.sum(axis=0)
    sell_score = sell_flag.sum(axis=0) * 0.6      # exit signals carry conf 0.6..0.95;
    # (individual exit confidences vary 0.5-0.95; 0.6 is the merge's typical
    #  weight — conservative: makes exits FIRE MORE EASILY than entries)

    n_buys = (buy_conf > 0).sum(axis=0)
    best_idx = buy_conf.argmax(axis=0)
    best_conf = buy_conf.max(axis=0)

    agreement = np.where(n_buys > 0, n_buys / n_strats, 0.0)
    merged_conf = np.minimum(0.99, best_conf * (0.75 + 0.25 * agreement * n_buys))

    entry = (buy_score > sell_score) & (n_buys > 0)
    exit_ = (sell_score > buy_score) & (sell_flag.sum(axis=0) > 0)

    # size / SL / TP from the highest-confidence contributor at each bar
    sizes = np.stack([s.size for s in subs.values()])
    sls = np.stack([s.stop_loss for s in subs.values()])
    tps = np.stack([s.take_profit for s in subs.values()])
    idx = (best_idx, np.arange(n))

    return SignalArrays(
        entry=entry, exit_=exit_,
        confidence=np.where(entry, merged_conf, 0.0),
        size=sizes[idx], stop_loss=sls[idx], take_profit=tps[idx],
    )


SIGNAL_GENERATORS = {
    'ma_crossover': signals_ma_crossover,
    'rsi_mean_reversion': signals_rsi_mean_reversion,
    'breakout': signals_breakout,
    'ensemble': signals_ensemble,
}

PARAM_GRIDS = {
    'ma_crossover': [
        {'fast_period': f, 'slow_period': s}
        for f in (10, 20, 30) for s in (50, 100) if f < s
    ],
    'rsi_mean_reversion': [
        {'rsi_period': p, 'oversold_threshold': lo, 'overbought_threshold': hi}
        for p in (7, 14) for lo in (25, 30) for hi in (70, 75)
    ],
    'breakout': [
        {'lookback_period': lb, 'volume_multiplier': vm, 'atr_multiplier': am}
        for lb in (20, 55) for vm in (1.5, 2.0) for am in (2.0, 3.0)
    ],
    # ensemble grid is assembled from the sub-strategies' train-window winners
}


# ── Fast simulator ──────────────────────────────────────────────────────────

@dataclass
class SimResult:
    trades: List[Dict] = field(default_factory=list)
    equity: Optional[pd.Series] = None
    final_equity: float = 0.0
    fees: float = 0.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def profit_factor(self) -> float:
        wins = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        losses = -sum(t['pnl'] for t in self.trades if t['pnl'] <= 0)
        if losses <= 0:
            return float('inf') if wins > 0 else 0.0
        return wins / losses


def simulate(df: pd.DataFrame, sig: SignalArrays, start_idx: int,
             end_idx: int, initial_equity: float = 10_000.0,
             min_confidence: float = MIN_CONFIDENCE) -> SimResult:
    """Simulate one long-only, one-position-at-a-time pass with full cash
    accounting and per-bar mark-to-market equity (honest drawdowns).

    Decisions on bar i (close), execution on bar i+1. New entries allowed only
    for signal bars i with start_idx <= i < end_idx (test-window purity);
    an open position may exit after end_idx (its outcome belongs to the
    window that opened it, exactly as the live engine would experience)."""
    open_, high, low, close = (df['open'].to_numpy(float),
                               df['high'].to_numpy(float),
                               df['low'].to_numpy(float),
                               df['close'].to_numpy(float))
    n = len(df)
    cash = initial_equity
    fees_total = 0.0
    trades: List[Dict] = []
    equity_curve = np.full(n, np.nan)

    in_pos = False
    qty = sl = tp = entry_px = 0.0
    entry_i = -1

    def close_position(i: int, exit_px: float, reason: str):
        nonlocal cash, fees_total, in_pos, qty
        proceeds = qty * exit_px
        commission = proceeds * COMMISSION
        pnl = proceeds - commission - qty * entry_px * (1 + COMMISSION)
        cash += proceeds - commission
        fees_total += commission
        trades.append({
            'entry_idx': entry_i, 'exit_idx': i,
            'entry_time': df.index[entry_i], 'exit_time': df.index[i],
            'entry_price': entry_px, 'exit_price': exit_px,
            'qty': qty, 'pnl': pnl,
            'pnl_pct': pnl / (qty * entry_px),
            'reason': reason, 'bars_held': i - entry_i,
        })
        in_pos = False
        qty = 0.0

    for i in range(start_idx, n):
        if in_pos:
            # scan bar i for SL (priority) / TP / signal exit
            exit_px = reason = None
            if not np.isnan(sl) and low[i] <= sl:
                raw = min(open_[i], sl)        # gap through stop -> worse open
                exit_px, reason = raw * (1 - SLIPPAGE), 'stop_loss'
            elif not np.isnan(tp) and high[i] >= tp:
                raw = max(open_[i], tp)        # gap up through TP -> real open
                exit_px, reason = raw * (1 - SLIPPAGE), 'take_profit'
            elif i - 1 >= entry_i and sig.exit_[i - 1]:
                # exit decided at close of bar i-1 -> execute at open of i
                exit_px, reason = open_[i] * (1 - SLIPPAGE), 'signal'
            if exit_px is not None:
                close_position(i, exit_px, reason)

        elif i < end_idx and i + 1 < n and sig.entry[i] \
                and sig.confidence[i] >= min_confidence and sig.size[i] > 0:
            nxt = i + 1
            fill = None
            if sig.passive_offset > 0:
                # Passive maker entry: rest BELOW the market; fills only if
                # the next bar trades strictly through the limit. No taker
                # slippage (we are the resting side), fill never better
                # than the limit.
                limit = close[i] * (1 - sig.passive_offset)
                if low[nxt] <= limit * (1 - SLIPPAGE):
                    fill = limit
            else:
                limit = close[i] * (1 + LIMIT_TOLERANCE)
                if open_[nxt] <= limit:
                    fill = min(open_[nxt] * (1 + SLIPPAGE), limit)
                elif low[nxt] <= limit:
                    fill = limit               # resting limit taken out
            if fill is not None:
                equity_now = cash               # flat -> equity == cash
                notional = min(equity_now * sig.size[i],
                               equity_now * MAX_POSITION_SIZE)
                commission = notional * COMMISSION
                if notional >= 10.0 and notional + commission <= cash:
                    qty = notional / fill
                    cash -= notional + commission
                    fees_total += commission
                    entry_px, sl, tp = fill, sig.stop_loss[i], sig.take_profit[i]
                    entry_i = nxt

                    in_pos = True

        equity_curve[i] = cash + (qty * close[i] if in_pos else 0.0)

        if not in_pos and i >= end_idx:
            break

    # Force-close any position still open at the very end of the data
    if in_pos:
        close_position(n - 1, close[n - 1] * (1 - SLIPPAGE), 'window_end')
        equity_curve[n - 1] = cash

    final_equity = cash
    result = SimResult(trades=trades, final_equity=final_equity,
                       fees=fees_total)
    valid = ~np.isnan(equity_curve)
    if valid.any():
        result.equity = pd.Series(equity_curve[valid],
                                  index=df.index[valid])
    return result


# ── Walk-forward orchestration ──────────────────────────────────────────────

@dataclass
class WindowResult:
    window: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: Dict
    train_pf: float
    train_trades: int
    oos_return: float
    oos_pf: float
    oos_trades: int
    oos_fees: float
    equity_before: float
    equity_after: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None


def _score(sim: SimResult, min_trades: int = 15) -> float:
    """Train-window selection score: profit factor after fees, requiring a
    minimum sample. Untradeable combos score -inf (window sat out)."""
    if sim.n_trades < min_trades:
        return -np.inf
    pf = sim.profit_factor()
    return pf if np.isfinite(pf) else 10.0


def walk_forward(df: pd.DataFrame, strategy: str,
                 train_days: int = 60, embargo_days: int = 2,
                 test_days: int = 20, initial_equity: float = 10_000.0,
                 param_grid: Optional[List[Dict]] = None,
                 signal_fn=None, min_train_trades: int = 15
                 ) -> Tuple[List[WindowResult], float]:
    """Run the full walk-forward for one (strategy, symbol, timeframe) frame.

    Returns (per-window results, final compounded equity). Parameters are
    selected on each train window only; the following test window is touched
    once with the chosen parameters."""
    signal_fn = signal_fn or SIGNAL_GENERATORS[strategy]
    grid = param_grid if param_grid is not None else PARAM_GRIDS[strategy]

    start_ts = df.index[0]
    end_ts = df.index[-1]
    results: List[WindowResult] = []
    equity = initial_equity

    window_no = 0
    train_start = start_ts
    while True:
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        if test_end > end_ts:
            break
        window_no += 1

        train_df = df.loc[train_start:train_end]
        # Signals need trailing history: give the test slice its warmup but
        # forbid entries before test_start (start_idx below).
        warm_start = test_start - timedelta(days=5)
        test_df = df.loc[warm_start:min(test_end + timedelta(days=2), end_ts)]
        test_mask_start = int(test_df.index.searchsorted(test_start))
        test_mask_end = int(test_df.index.searchsorted(test_end))

        # ── Optimize on train only ──
        current_grid = grid.for_train(train_df) if isinstance(grid, _DynamicGrid) else grid
        best_params, best_score, best_train = None, -np.inf, None
        for params in current_grid:
            sig = signal_fn(train_df, params)
            sim = simulate(train_df, sig, start_idx=0, end_idx=len(train_df),
                           initial_equity=10_000.0)
            score = _score(sim, min_train_trades)
            if score > best_score:
                best_params, best_score, best_train = params, score, sim

        if best_params is None or not np.isfinite(best_score):
            # No tradeable configuration in-sample -> sit this window out
            results.append(WindowResult(
                window=window_no, train_start=train_start,
                test_start=test_start, test_end=test_end,
                params={}, train_pf=0.0, train_trades=0,
                oos_return=0.0, oos_pf=0.0, oos_trades=0, oos_fees=0.0,
                equity_before=equity, equity_after=equity))
            train_start = train_start + timedelta(days=test_days)
            continue

        # ── One OOS pass with the chosen parameters ──
        sig = signal_fn(test_df, best_params)
        sim = simulate(test_df, sig, start_idx=test_mask_start,
                       end_idx=test_mask_end, initial_equity=equity)

        results.append(WindowResult(
            window=window_no, train_start=train_start,
            test_start=test_start, test_end=test_end,
            params=best_params,
            train_pf=best_train.profit_factor() if best_train else 0.0,
            train_trades=best_train.n_trades if best_train else 0,
            oos_return=sim.final_equity / equity - 1,
            oos_pf=sim.profit_factor(),
            oos_trades=sim.n_trades,
            oos_fees=sim.fees,
            equity_before=equity,
            equity_after=sim.final_equity,
            trades=sim.trades,
            equity_curve=sim.equity))
        equity = sim.final_equity

        train_start = train_start + timedelta(days=test_days)

    return results, equity


def walk_forward_ensemble(df: pd.DataFrame, **kwargs):
    """Ensemble walk-forward: each train window first picks the best params
    for each sub-strategy (train data only), then the ensemble merges those
    winners. No extra free parameters are introduced."""
    def ensemble_grid_for_train(train_df):
        chosen = {}
        for name in ('ma_crossover', 'rsi_mean_reversion', 'breakout'):
            best_params, best_score = {}, -np.inf
            for params in PARAM_GRIDS[name]:
                sig = SIGNAL_GENERATORS[name](train_df, params)
                sim = simulate(train_df, sig, 0, len(train_df))
                score = _score(sim, min_trades=10)
                if score > best_score:
                    best_params, best_score = params, score
            chosen[name] = best_params
        return [chosen]

    # walk_forward with a per-window dynamic grid
    return walk_forward(df, 'ensemble',
                        param_grid=_DynamicGrid(ensemble_grid_for_train),
                        signal_fn=signals_ensemble, **kwargs)


class _DynamicGrid:
    """Sentinel list-like: walk_forward iterates it per train window."""
    def __init__(self, builder):
        self.builder = builder

    def for_train(self, train_df):
        return self.builder(train_df)

    def __iter__(self):
        raise TypeError("dynamic grid must be resolved per train window")
