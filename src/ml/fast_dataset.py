"""
Fast Lab dataset (1m horizon) — Part C of docs/FASTLAB_PLAN.md.

Anti-lookahead contract (tests/test_fast_dataset.py):
  * every feature/channel at bar T uses data <= T (rolling/shift only)
  * label at T = executable open(T+1) -> open(T+1+HORIZON) return, NET of
    the full modeled round trip (fees + slippage + per-symbol spread)
  * at this frequency most labels are honestly FLAT — that is the fee wall
    appearing inside the training target, by design

Two views of the same bars:
  * tabular features (trees/linear): TAB_COLUMNS
  * sequence channels (LSTM/CNN):    SEQ_CHANNELS, trailing SEQ_LEN bars
"""

import numpy as np
import pandas as pd

HORIZON = 30                 # minutes ahead
SEQ_LEN = 60                 # sequence length for deep models
STRIDE = 5                   # sample every 5th minute (overlap redundancy)
DEAD_ZONE = 0.001            # +/-0.1% net -> FLAT
BASE_ROUND_TRIP = 0.003      # 2 x (0.10% fee + 0.05% slip); spread added per symbol

LABEL_DOWN, LABEL_FLAT, LABEL_UP = 0, 1, 2

SEQ_CHANNELS = ['ret_1', 'ret_5', 'ret_15', 'ret_60',
                'vol_z60', 'volume_z60', 'range_pct', 'dist_sma60']
TAB_COLUMNS = SEQ_CHANNELS + ['rsi_14', 'atr_pct', 'ret_240',
                              'tod_sin', 'tod_cos']


def build_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar features, all trailing."""
    out = pd.DataFrame(index=df.index)
    close, volume = df['close'], df['volume']
    rets = close.pct_change()

    for n in (1, 5, 15, 60, 240):
        out[f'ret_{n}'] = close.pct_change(n)

    rv60 = rets.rolling(60).std()
    out['vol_z60'] = (rv60 - rv60.rolling(240).mean()) / \
        rv60.rolling(240).std().replace(0, np.nan)
    v_ma, v_sd = volume.rolling(60).mean(), volume.rolling(60).std()
    out['volume_z60'] = (volume - v_ma) / v_sd.replace(0, np.nan)
    out['range_pct'] = (df['high'] - df['low']) / close
    out['dist_sma60'] = close / close.rolling(60).mean() - 1

    from src.utils.indicators import RSI, ATR
    out['rsi_14'] = RSI(close, 14) / 100.0
    out['atr_pct'] = ATR(df['high'], df['low'], close, 14) / close

    minute_of_day = df.index.hour * 60 + df.index.minute
    out['tod_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    out['tod_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
    return out


def build_labels(df: pd.DataFrame, round_trip: float,
                 horizon: int = HORIZON,
                 dead_zone: float = DEAD_ZONE) -> pd.DataFrame:
    entry = df['open'].shift(-1)
    exit_ = df['open'].shift(-(1 + horizon))
    net = (exit_ / entry - 1) - round_trip
    label = pd.Series(LABEL_FLAT, index=df.index, dtype=int)
    label[net > dead_zone] = LABEL_UP
    label[net < -dead_zone] = LABEL_DOWN
    label[net.isna()] = -1
    return pd.DataFrame({'label': label, 'net_fwd_return': net})


def assemble_fast_panel(frames: dict, spreads: dict,
                        stride: int = STRIDE):
    """Pooled sampled panel across symbols.

    Returns (X_tab, X_seq, meta): X_tab (n, len(TAB_COLUMNS)) float32,
    X_seq (n, SEQ_LEN, len(SEQ_CHANNELS)) float32, meta with
    [date, symbol, label, net_fwd_return]."""
    tab_parts, seq_parts, meta_parts = [], [], []

    for symbol, df in frames.items():
        chan = build_channels(df)
        labels = build_labels(df, BASE_ROUND_TRIP + spreads.get(symbol, 0.0))

        chan_mat = chan[SEQ_CHANNELS].to_numpy(np.float32)
        tab_mat = chan[TAB_COLUMNS].to_numpy(np.float32)

        # valid sample indices: full seq history, finite features, labeled
        idx = np.arange(SEQ_LEN - 1, len(df), stride)
        finite = (np.isfinite(tab_mat[idx]).all(axis=1)
                  & (labels['label'].to_numpy()[idx] >= 0))
        # sequence windows must be fully finite too
        seq_ok = np.array([np.isfinite(chan_mat[i - SEQ_LEN + 1:i + 1]).all()
                           for i in idx])
        idx = idx[finite & seq_ok]

        seq = np.stack([chan_mat[i - SEQ_LEN + 1:i + 1] for i in idx])
        tab_parts.append(tab_mat[idx])
        seq_parts.append(seq)
        meta_parts.append(pd.DataFrame({
            'date': df.index[idx], 'symbol': symbol,
            'label': labels['label'].to_numpy()[idx],
            'net_fwd_return': labels['net_fwd_return'].to_numpy()[idx]}))

    X_tab = np.concatenate(tab_parts)
    X_seq = np.concatenate(seq_parts)
    meta = pd.concat(meta_parts, ignore_index=True)

    order = np.argsort(meta['date'].to_numpy(), kind='stable')
    return X_tab[order], X_seq[order], meta.iloc[order].reset_index(drop=True)
