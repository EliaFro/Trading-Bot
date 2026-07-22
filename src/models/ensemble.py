"""
EnsembleModel — combines strategy signals into one decision per symbol.

Honest v1 design (see docs/ARCHITECTURE_REPORT.md §6.1): the ensemble is a
confidence-weighted vote over the classical strategies (MA crossover, RSI
mean-reversion, breakout), with two supporting inputs:

  * discovered patterns  — an aligned active pattern adds confidence
  * sentiment            — scales position size up/down, never creates trades

Deep models (transformer/TCN/RL) join this ensemble only after they pass
walk-forward validation; until then they are not pretended into existence.
The retrain() contract is kept so the orchestrator's retraining safeguard
(keep old models unless improvement > min_improvement) works end-to-end.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Signal combiner over the enabled strategies."""

    def __init__(self, config):
        # Accept either the full Config object or a plain models dict
        if hasattr(config, 'strategies'):
            self.strategies_cfg = config.strategies
            self.models_cfg = config.models
            self.sentiment_cfg = config.sentiment
        else:
            self.strategies_cfg = (config or {}).get('strategies', {})
            self.models_cfg = config or {}
            self.sentiment_cfg = (config or {}).get('sentiment', {})

        self.enabled = self.strategies_cfg.get(
            'enabled', ['ma_crossover', 'rsi_mean_reversion', 'breakout'])
        self.params = self.strategies_cfg.get('params', {})
        self.min_confidence = float(self.strategies_cfg.get('min_confidence', 0.55))
        self.pattern_weight = float(
            (self.models_cfg.get('ensemble', {}).get('weights', {}) or {})
            .get('patterns', 0.1))
        self.sentiment_size_max = float(
            self.sentiment_cfg.get('size_modifier_max', 0.2))

        self._strategies = self._build_strategies()
        self._last_retrain: Optional[datetime] = None

    def _build_strategies(self) -> Dict[str, Any]:
        from src.backtesting.backtest_module import StrategyFactory
        strategies = {}
        for name in self.enabled:
            try:
                strategies[name] = StrategyFactory.create_strategy(
                    name, dict(self.params.get(name, {})))
            except Exception as e:
                logger.error(f"Could not create strategy '{name}': {e}")
        logger.info(f"Ensemble strategies: {list(strategies)}")
        return strategies

    # ── Signal generation ────────────────────────────────────────────────

    def generate_signals(self, symbol: str, data: pd.DataFrame,
                         positions: Dict, portfolio_value: float,
                         sentiment: Optional[Dict] = None,
                         patterns: Optional[List[Dict]] = None) -> List[Dict]:
        """Run every strategy on `data` and merge their signals into at most
        one actionable signal for this symbol."""
        raw_signals: List[Dict] = []
        for name, strategy in self._strategies.items():
            try:
                out = strategy.generate_signals(
                    data.copy(), positions, portfolio_value, symbol=symbol)
                for s in out or []:
                    s.setdefault('metadata', {})['strategy'] = name
                    raw_signals.append(s)
            except Exception as e:
                logger.error(f"Strategy {name} failed on {symbol}: {e}")

        if not raw_signals:
            return []

        merged = self._merge(symbol, raw_signals)
        if merged is None:
            return []

        # Pattern confirmation: an active pattern pointing the same way
        # adds (weighted) confidence.
        bonus = self._pattern_bonus(symbol, merged['action'], patterns)
        if bonus:
            merged['confidence'] = min(0.99, merged['confidence'] + bonus)
            merged['metadata']['pattern_bonus'] = round(bonus, 4)

        # Sentiment: size modifier only (long-only: negative sentiment
        # shrinks buys, it never triggers sells).
        if merged['action'] == 'BUY' and sentiment and 'size' in merged:
            score = float(sentiment.get('sentiment', 0.0))
            modifier = 1.0 + max(-1.0, min(1.0, score)) * self.sentiment_size_max
            merged['size'] = merged['size'] * modifier
            merged['metadata']['sentiment_modifier'] = round(modifier, 3)

        return [merged]

    def _merge(self, symbol: str, signals: List[Dict]) -> Optional[Dict]:
        """Confidence-weighted vote. BUY and SELL cancel; strongest side wins."""
        buys = [s for s in signals if s.get('action') == 'BUY']
        sells = [s for s in signals if s.get('action') == 'SELL']

        buy_score = sum(float(s.get('confidence', 0.5) or 0.5) for s in buys)
        sell_score = sum(float(s.get('confidence', 0.5) or 0.5) for s in sells)

        if buy_score > sell_score and buys:
            best = max(buys, key=lambda s: float(s.get('confidence', 0) or 0))
            n = len(self._strategies)
            agreement = len(buys) / n if n else 1.0
            confidence = min(0.99, float(best.get('confidence', 0.5)) *
                             (0.75 + 0.25 * agreement * len(buys)))
            return {
                'symbol': symbol,
                'action': 'BUY',
                'size': float(best.get('size', 0.05) or 0.05),
                'confidence': confidence,
                'stop_loss': best.get('stop_loss'),
                'take_profit': best.get('take_profit'),
                'metadata': {
                    'strategy': 'ensemble',
                    'contributors': [s['metadata'].get('strategy') for s in buys],
                    'buy_score': round(buy_score, 3),
                    'sell_score': round(sell_score, 3),
                },
            }
        if sell_score > buy_score and sells:
            return {
                'symbol': symbol,
                'action': 'SELL',
                'confidence': min(0.99, sell_score / max(len(sells), 1)),
                'metadata': {
                    'strategy': 'ensemble',
                    'contributors': [s['metadata'].get('strategy') for s in sells],
                    'buy_score': round(buy_score, 3),
                    'sell_score': round(sell_score, 3),
                },
            }
        return None

    def _pattern_bonus(self, symbol: str, action: str,
                       patterns: Optional[List[Dict]]) -> float:
        if not patterns:
            return 0.0
        bullish = {'double_bottom', 'inverse_head_shoulders', 'flag_bullish',
                   'triangle_ascending', 'channel_up', 'cup_handle', 'breakout',
                   'reversal_bullish', 'continuation_bullish'}
        bearish = {'head_shoulders', 'double_top', 'flag_bearish',
                   'triangle_descending', 'channel_down', 'breakdown',
                   'reversal_bearish', 'continuation_bearish'}
        bonus = 0.0
        for p in patterns:
            if p.get('symbol') not in (None, symbol):
                continue
            ptype = p.get('pattern_type', '')
            confidence = float(p.get('confidence', 0.5) or 0.5)
            if action == 'BUY' and ptype in bullish:
                bonus += self.pattern_weight * confidence
            elif action == 'SELL' and ptype in bearish:
                bonus += self.pattern_weight * confidence
        return min(bonus, self.pattern_weight)  # cap at the pattern weight

    # ── Compatibility API ────────────────────────────────────────────────

    def predict(self, data: pd.DataFrame,
                symbol: str = 'BTC/USDT') -> Tuple[int, float]:
        """(direction, confidence) for MLEnsembleStrategy: 1 buy, -1 sell, 0 hold."""
        signals = self.generate_signals(symbol, data, {}, 10_000.0)
        if not signals:
            return 0, 0.0
        s = signals[0]
        direction = 1 if s['action'] == 'BUY' else -1 if s['action'] == 'SELL' else 0
        return direction, float(s.get('confidence', 0.0))

    def get_active_models(self) -> List[str]:
        return list(self._strategies.keys())

    async def retrain(self, training_data: Dict) -> Dict[str, Any]:
        """v1: no trainable deep models in the ensemble — report honestly.

        Phase 2 wires this to walk-forward strategy re-optimization; the
        orchestrator already keeps old parameters unless improvement exceeds
        models.min_improvement, so returning 0.0 is safe (never regresses).
        """
        self._last_retrain = datetime.now(timezone.utc)
        n = sum(len(df) for df in (training_data or {}).values()
                if df is not None and hasattr(df, '__len__'))
        logger.info(f"retrain() called with {n} rows — v1 ensemble has no "
                    f"trainable deep models; keeping current strategies")
        return {
            'improvement': 0.0,
            'noop': True,
            'model_name': 'ensemble_v1',
            'version': self._last_retrain.strftime('%Y%m%d%H%M%S'),
            'details': 'v1 classical ensemble: retraining is a no-op until '
                       'Phase 2 walk-forward optimization is wired in',
        }
