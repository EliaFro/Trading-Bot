"""
Sentiment analysis for crypto assets.

CryptoSentimentAnalyzer — scores raw text in [-1, 1]:
  * FinBERT (models/finbert) when transformers+torch are available — the one
    real pre-trained model shipped with this project
  * VADER lexicon fallback otherwise (still keyless, still useful)

SentimentAnalyzer — the component main.py drives: analyze_batch(symbols)
collects fresh headlines and returns per-symbol aggregate scores in the shape
DatabaseManager.store_sentiment expects:
    {'sentiment': float, 'confidence': float, 'volume': int,
     'source': str, 'metadata': {...}}
"""

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

FINBERT_DIR = Path(os.getenv('MODEL_PATH', './models')) / 'finbert'


class CryptoSentimentAnalyzer:
    """Text -> sentiment score in [-1, 1] with confidence."""

    def __init__(self, prefer_finbert: bool = True, batch_size: int = 16):
        self.batch_size = batch_size
        self._finbert = None
        self._tokenizer = None
        self._vader = None
        self.backend = 'none'

        if prefer_finbert and FINBERT_DIR.exists():
            try:
                self._load_finbert()
                self.backend = 'finbert'
            except Exception as e:
                logger.warning(f"FinBERT unavailable ({e}); falling back to VADER")
        if self.backend == 'none':
            self._load_vader()
            self.backend = 'vader'
        logger.info(f"Sentiment backend: {self.backend}")

    def _load_finbert(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(FINBERT_DIR))
        self._finbert = AutoModelForSequenceClassification.from_pretrained(
            str(FINBERT_DIR))
        self._finbert.eval()
        self._torch = torch
        # ProsusAI/finbert label order: positive, negative, neutral
        id2label = getattr(self._finbert.config, 'id2label', {}) or {}
        self._labels = [str(id2label.get(i, '')).lower()
                        for i in range(self._finbert.config.num_labels)]

    def _load_vader(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()

    def score_texts(self, texts: List[str]) -> List[Dict]:
        """Score a list of texts. Returns [{'score': float, 'confidence': float}]."""
        if not texts:
            return []
        if self.backend == 'finbert':
            return self._score_finbert(texts)
        return self._score_vader(texts)

    def _score_finbert(self, texts: List[str]) -> List[Dict]:
        torch = self._torch
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [t[:512] for t in texts[i:i + self.batch_size]]
                inputs = self._tokenizer(batch, return_tensors='pt',
                                         truncation=True, padding=True,
                                         max_length=128)
                probs = torch.softmax(self._finbert(**inputs).logits, dim=-1)
                for row in probs:
                    scores = {label: float(p) for label, p
                              in zip(self._labels, row)}
                    positive = scores.get('positive', 0.0)
                    negative = scores.get('negative', 0.0)
                    results.append({
                        'score': positive - negative,
                        'confidence': max(scores.values()),
                    })
        return results

    def _score_vader(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            v = self._vader.polarity_scores(text[:1000])
            results.append({'score': v['compound'],
                            'confidence': 1.0 - v['neu'] * 0.5})
        return results


class SentimentAnalyzer:
    """Batch sentiment for trading symbols, driven by the orchestrator loop."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.asset_map: Dict[str, List[str]] = self.config.get('asset_map', {})
        self.analyzer = CryptoSentimentAnalyzer()

        from src.sentiment.collectors import NewsCollector
        self.news = NewsCollector(api_keys={
            'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
        })

    def _terms_for(self, symbol: str) -> List[str]:
        terms = self.asset_map.get(symbol)
        if terms:
            return [t.lower() for t in terms]
        base = symbol.split('/')[0]
        return [base.lower()]

    async def analyze_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Collect fresh headlines once and score them per symbol."""
        try:
            articles = await self.news.collect_news(limit=60)
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            articles = []

        if not articles:
            # Loud neutral: volume 0 and zero confidence — never fabricated
            return {s: {'sentiment': 0.0, 'confidence': 0.0, 'volume': 0,
                        'source': 'news', 'metadata': {'note': 'no data'}}
                    for s in symbols}

        texts = [a['text'] for a in articles]
        scores = await asyncio.to_thread(self.analyzer.score_texts, texts)

        results: Dict[str, Dict] = {}
        for symbol in symbols:
            terms = self._terms_for(symbol)
            matched = [
                (article, score) for article, score in zip(articles, scores)
                if any(term in article['text'].lower()
                       or term in str(article.get('categories', '')).lower()
                       for term in terms)
            ]
            if not matched:
                results[symbol] = {'sentiment': 0.0, 'confidence': 0.0,
                                   'volume': 0, 'source': 'news',
                                   'metadata': {'note': 'no matching articles'}}
                continue

            values = [s['score'] for _, s in matched]
            mean = sum(values) / len(values)
            # Confidence grows with sample size and falls with disagreement
            spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            confidence = min(0.95, (1 - min(spread, 1.0)) *
                             (1 - math.exp(-len(values) / 5)))
            sources: Dict[str, int] = {}
            for article, _ in matched:
                sources[article['source']] = sources.get(article['source'], 0) + 1

            results[symbol] = {
                'sentiment': round(mean, 4),
                'confidence': round(confidence, 4),
                'volume': len(matched),
                'source': 'news',
                'metadata': {
                    'backend': self.analyzer.backend,
                    'sources': sources,
                    'as_of': datetime.now(timezone.utc).isoformat(),
                },
            }
        return results
