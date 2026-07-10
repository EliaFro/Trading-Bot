"""Ensemble merging logic and sentiment scoring (offline paths only)."""

import pytest

from src.models.ensemble import EnsembleModel
from src.utils.config import Config


@pytest.fixture
def ensemble():
    config = Config(
        strategies={'enabled': ['ma_crossover', 'rsi_mean_reversion', 'breakout'],
                    'min_confidence': 0.55,
                    'params': {}},
        models={'ensemble': {'weights': {'patterns': 0.1}}},
        sentiment={'size_modifier_max': 0.2},
    )
    return EnsembleModel(config)


def test_ensemble_builds_enabled_strategies(ensemble):
    assert set(ensemble.get_active_models()) == {
        'ma_crossover', 'rsi_mean_reversion', 'breakout'}


def test_merge_prefers_stronger_side(ensemble):
    merged = ensemble._merge('BTC/USDT', [
        {'action': 'BUY', 'confidence': 0.8, 'size': 0.05,
         'stop_loss': 49_000, 'take_profit': 52_000,
         'metadata': {'strategy': 'breakout'}},
        {'action': 'BUY', 'confidence': 0.6, 'size': 0.04,
         'metadata': {'strategy': 'ma_crossover'}},
        {'action': 'SELL', 'confidence': 0.5,
         'metadata': {'strategy': 'rsi_mean_reversion'}},
    ])
    assert merged['action'] == 'BUY'
    assert merged['symbol'] == 'BTC/USDT'
    assert merged['stop_loss'] == 49_000          # from strongest contributor
    assert set(merged['metadata']['contributors']) == {'breakout', 'ma_crossover'}


def test_merge_returns_none_on_tie(ensemble):
    assert ensemble._merge('BTC/USDT', [
        {'action': 'BUY', 'confidence': 0.6, 'metadata': {'strategy': 'a'}},
        {'action': 'SELL', 'confidence': 0.6, 'metadata': {'strategy': 'b'}},
    ]) is None


def test_sentiment_scales_size_not_direction(ensemble):
    base = [{'action': 'BUY', 'confidence': 0.8, 'size': 0.05,
             'metadata': {'strategy': 'breakout'}}]

    import pandas as pd, numpy as np
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.005, 120)))
    df = pd.DataFrame({'open': close, 'high': close * 1.01,
                       'low': close * 0.99, 'close': close,
                       'volume': np.full(120, 1000.0)},
                      index=pd.date_range('2025-01-01', periods=120, freq='15min'))

    # monkeypatch strategy outputs so the test controls the raw signals
    for strategy in ensemble._strategies.values():
        strategy.generate_signals = lambda *a, **k: []
    first = list(ensemble._strategies)
    ensemble._strategies[first[0]].generate_signals = \
        lambda *a, **k: [dict(base[0])]

    bullish = ensemble.generate_signals('BTC/USDT', df, {}, 10_000,
                                        sentiment={'sentiment': 1.0})
    bearish = ensemble.generate_signals('BTC/USDT', df, {}, 10_000,
                                        sentiment={'sentiment': -1.0})
    assert bullish[0]['action'] == bearish[0]['action'] == 'BUY'
    assert bullish[0]['size'] == pytest.approx(0.05 * 1.2)
    assert bearish[0]['size'] == pytest.approx(0.05 * 0.8)


def test_pattern_bonus_capped(ensemble):
    patterns = [{'symbol': 'BTC/USDT', 'pattern_type': 'double_bottom',
                 'confidence': 0.9}] * 10
    bonus = ensemble._pattern_bonus('BTC/USDT', 'BUY', patterns)
    assert bonus == pytest.approx(0.1)            # capped at pattern weight


def test_vader_sentiment_direction():
    from src.sentiment.analyzer import CryptoSentimentAnalyzer
    analyzer = CryptoSentimentAnalyzer(prefer_finbert=False)
    scores = analyzer.score_texts([
        "Bitcoin surges to record high as adoption accelerates, huge gains",
        "Crypto crash deepens: massive losses, fraud charges and panic selling",
    ])
    assert scores[0]['score'] > 0.2
    assert scores[1]['score'] < -0.2
