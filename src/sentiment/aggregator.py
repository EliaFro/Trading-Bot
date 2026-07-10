"""
SentimentAggregator — merges scored social posts and news articles into
per-symbol SentimentScore objects for the standalone sentiment service
(src/sentiment/main.py). The in-process path (SentimentAnalyzer.analyze_batch)
does its own lighter aggregation.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)

DEFAULT_ASSETS = {
    'BTC': ['btc', 'bitcoin'],
    'ETH': ['eth', 'ethereum'],
    'SOL': ['sol', 'solana'],
}


@dataclass
class SentimentScore:
    symbol: str
    sentiment: float
    confidence: float
    volume: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)


class SentimentAggregator:
    def __init__(self, analyzer, asset_terms: Dict[str, List[str]] = None):
        self.analyzer = analyzer               # CryptoSentimentAnalyzer
        self.asset_terms = asset_terms or DEFAULT_ASSETS

    async def aggregate_sentiment(self, social_posts: List[Dict],
                                  news_articles: List[Dict],
                                  time_window: timedelta = timedelta(hours=24)
                                  ) -> Dict[str, SentimentScore]:
        cutoff = datetime.now(timezone.utc) - time_window
        items = [i for i in (social_posts + news_articles)
                 if i.get('timestamp') and i['timestamp'] >= cutoff]
        if not items:
            return {}

        texts = [i['text'] for i in items]
        scores = self.analyzer.score_texts(texts)

        results: Dict[str, SentimentScore] = {}
        for symbol, terms in self.asset_terms.items():
            matched = [(item, score) for item, score in zip(items, scores)
                       if any(t in item['text'].lower() for t in terms)]
            if not matched:
                continue
            values = [s['score'] for _, s in matched]
            mean = sum(values) / len(values)
            spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            confidence = min(0.95, (1 - min(spread, 1.0)) *
                             (1 - math.exp(-len(values) / 8)))
            sources: Dict[str, int] = {}
            for item, _ in matched:
                key = item.get('source', 'unknown').split('/')[0]
                sources[key] = sources.get(key, 0) + 1
            results[symbol] = SentimentScore(
                symbol=symbol, sentiment=round(mean, 4),
                confidence=round(confidence, 4), volume=len(matched),
                metadata={'sources': sources})
        return results
