"""
PatternDiscoveryEngine — orchestrates the existing pattern pipeline:

    PatternDataLoader -> AdvancedFeatureExtractor -> AdvancedPatternDetector
                      -> AdvancedPatternPostprocessor

Driven by AITradingSystem._run_pattern_discovery(); discover() returns a list
of pattern dicts ready for DatabaseManager.store_pattern() and, if they pass
the mini-backtest gate in main.py, TradingEngine.add_pattern().

REAL DATA ONLY: if the database lacks sufficient history for a symbol, that
symbol is skipped loudly. Synthetic fallback data is never used here.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatternDiscoveryEngine:
    def __init__(self, config: Optional[Dict] = None, db=None):
        self.config = config or {}
        self.db = db
        self.timeframe = self.config.get('timeframe', '15m')
        self.min_history_bars = int(self.config.get('min_history_bars', 1000))
        self.symbols = self.config.get('symbols')  # None -> from caller/db

        self._loader = None
        self._extractor = None
        self._detector = None
        self._postprocessor = None
        self._initialized = False

    def attach_db(self, db):
        self.db = db

    def _lazy_init(self):
        """Import the heavy pipeline only when discovery actually runs."""
        if self._initialized:
            return
        from src.patterns.data_loader import PatternDataLoader
        from src.patterns.feature_extractor import AdvancedFeatureExtractor
        from src.patterns.detector import AdvancedPatternDetector
        from src.patterns.postprocessor import AdvancedPatternPostprocessor

        db_path = getattr(self.db, 'db_path', None) if self.db else None
        self._loader = PatternDataLoader(db_path=db_path)
        self._extractor = AdvancedFeatureExtractor()
        self._detector = AdvancedPatternDetector(enable_deep_learning=False)
        self._postprocessor = AdvancedPatternPostprocessor()
        self._initialized = True
        logger.info("Pattern pipeline initialized")

    async def discover(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Run one discovery pass. Returns pattern dicts:
        {pattern_id, pattern_type, symbol, timeframe, pattern_config,
         confidence, performance, discovery_date, status}"""
        symbols = symbols or self.symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        return await asyncio.to_thread(self._discover_sync, symbols)

    def _discover_sync(self, symbols: List[str]) -> List[Dict]:
        self._lazy_init()
        discovered: List[Dict] = []

        for symbol in symbols:
            try:
                df = self._load_real_history(symbol)
                if df is None:
                    continue
                discovered.extend(self._discover_for_symbol(symbol, df))
            except Exception as e:
                logger.exception(f"Pattern discovery failed for {symbol}: {e}")

        if discovered:
            logger.info(f"Discovery pass found {len(discovered)} candidate patterns")
        return discovered

    def _load_real_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load history from the database only — no synthetic fallback."""
        if self.db is None:
            logger.error("PatternDiscoveryEngine has no database attached")
            return None
        df = self.db.get_ohlcv_data(symbol, self.timeframe,
                                    limit=self.min_history_bars * 3)
        if df.empty or len(df) < self.min_history_bars:
            logger.warning(
                f"Skipping {symbol} {self.timeframe}: only {len(df)} bars "
                f"stored, need {self.min_history_bars}. Run scripts/backfill.py.")
            return None
        df = df.set_index('timestamp').sort_index()
        # Reuse the loader's preprocessing (indicators etc.) on REAL data
        return self._loader._preprocess_data(df)

    def _discover_for_symbol(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        window_size = 50
        windows = self._loader.prepare_pattern_windows(df, [window_size])
        if window_size not in windows or len(windows[window_size]) == 0:
            return []

        window_array = windows[window_size]
        # Score only the most recent windows each pass (discovery is periodic)
        recent = window_array[-200:]

        features = []
        for window in recent:
            f = self._extractor.extract_features(window, symbol=symbol)
            features.append(np.asarray(f.combined_features, dtype=float).ravel())

        # The extractor can emit shorter vectors when a feature group errors
        # on a degenerate window — keep only the modal length so the matrix
        # stays rectangular.
        if not features:
            return []
        lengths = [len(f) for f in features]
        modal = max(set(lengths), key=lengths.count)
        kept = [f for f in features if len(f) == modal]
        if len(kept) < len(features):
            logger.debug(f"{symbol}: dropped {len(features) - len(kept)} "
                         f"windows with inconsistent feature length")
        if len(kept) < 20:
            return []
        feature_matrix = np.vstack(kept)

        detections = self._detector.detect_patterns(
            feature_matrix, timestamps=None, symbol=symbol) or []

        results = []
        now = datetime.now(timezone.utc)
        for det in detections:
            confidence = float(getattr(det, 'confidence', 0.0) or 0.0)
            if confidence < 0.5:
                continue
            results.append({
                'pattern_id': str(uuid.uuid4()),
                'pattern_type': getattr(det, 'pattern_type', 'unknown'),
                'symbol': symbol,
                'timeframe': self.timeframe,
                'pattern_config': {
                    'window_size': window_size,
                    'market_regime': getattr(det, 'market_regime', 'unknown'),
                    'strength': float(getattr(det, 'pattern_strength', 0.0) or 0.0),
                    'risk_reward': float(getattr(det, 'risk_reward_ratio', 0.0) or 0.0),
                },
                'confidence': confidence,
                'performance': 0.0,        # set by the mini-backtest evaluator
                'discovery_date': now,
                'status': 'candidate',
            })
        return results

    def retrain_with_feedback(self, feedbacks: List[Dict]):
        """Feed realized trade outcomes back into the detector (Phase 5)."""
        self._lazy_init()
        try:
            self._detector.retrain_with_feedback(feedbacks)
        except Exception as e:
            logger.error(f"Pattern feedback retraining failed: {e}")
