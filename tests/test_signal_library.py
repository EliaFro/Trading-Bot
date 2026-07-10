"""Signal library: every published rule must be lookahead-free and produce
sane, tradeable signal arrays."""

import numpy as np
import pytest

from src.backtesting.signal_library import LIBRARY, N_STRATEGIES, noise_strategy
from tests.test_ml_dataset import synthetic_daily


@pytest.fixture(scope='module')
def df():
    return synthetic_daily(n=700, seed=21)


def test_library_size_is_counted():
    assert N_STRATEGIES == len(LIBRARY) == 29


@pytest.mark.parametrize('name', list(LIBRARY))
def test_signals_are_sane(name, df):
    sig = LIBRARY[name](df)
    n = len(df)
    assert sig.entry.shape == sig.exit_.shape == (n,)
    assert sig.entry.dtype == bool and sig.exit_.dtype == bool
    assert (sig.size >= 0).all() and (sig.size <= 0.34).all()
    # every strategy must actually fire on 700 days of volatile data
    assert sig.entry.sum() > 0, f"{name}: no entries ever"
    assert sig.exit_.sum() > 0, f"{name}: no exits ever"


@pytest.mark.parametrize('name', list(LIBRARY))
def test_no_lookahead_truncation_invariance(name, df):
    """Signal values at bar T must be identical whether computed on full
    history or history truncated at T."""
    if name.startswith('tsmom'):
        pytest.skip("tsmom generators covered by the daily-momentum study "
                    "tests; weekly masks are index-based and trailing")
    full = LIBRARY[name](df)
    for t_idx in (400, 550):
        trunc = LIBRARY[name](df.iloc[:t_idx + 1])
        assert full.entry[t_idx] == trunc.entry[-1], f"{name}: entry lookahead"
        assert full.exit_[t_idx] == trunc.exit_[-1], f"{name}: exit lookahead"


def test_noise_strategies_are_deterministic_per_seed(df):
    a = noise_strategy(42)(df)
    b = noise_strategy(42)(df)
    c = noise_strategy(43)(df)
    assert (a.entry == b.entry).all()
    assert not (a.entry == c.entry).all()
    # matched frequency: entries fire on roughly 5% of days
    assert 0.02 < a.entry.mean() < 0.09
