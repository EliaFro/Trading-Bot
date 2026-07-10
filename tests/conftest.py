"""Shared test fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_db(tmp_path):
    """A DatabaseManager on a fresh canonical-schema database."""
    from scripts.init_db import init_db
    from src.utils.database import DatabaseManager
    db_path = str(tmp_path / 'test.db')
    init_db(db_path)
    db = DatabaseManager(db_path)
    yield db
    db.close()


@pytest.fixture
def trending_ohlcv():
    """Deterministic 500-bar OHLCV frame with a clear up-trend then pullback,
    enough structure for every strategy to fire at least once."""
    rng = np.random.default_rng(7)
    n = 500
    drift = np.concatenate([
        np.full(200, 0.0015),    # strong up-trend
        np.full(100, -0.002),    # pullback (RSI oversold territory)
        np.full(200, 0.001),     # recovery / breakout
    ])
    returns = drift + rng.normal(0, 0.004, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    volume = rng.uniform(500, 1500, n)
    volume[::17] *= 3  # periodic volume spikes for volume-confirmed setups

    index = pd.date_range('2025-01-01', periods=n, freq='15min')
    return pd.DataFrame({'open': open_, 'high': np.maximum(high, close),
                         'low': np.minimum(low, close), 'close': close,
                         'volume': volume}, index=index)
