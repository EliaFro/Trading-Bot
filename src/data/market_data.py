"""
Market data pipeline: live + historical OHLCV from Binance via ccxt.

Public endpoints only — no API keys required for market data. All methods are
synchronous (ccxt sync client with rate limiting); async callers wrap them in
asyncio.to_thread().
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '2h': 7200, '4h': 14400, '1d': 86400,
}


class MarketData:
    """Fetches OHLCV/tickers from Binance and persists them via DatabaseManager."""

    def __init__(self, db=None, exchange_id: str = 'binance',
                 max_retries: int = 3):
        self.db = db
        self.max_retries = max_retries
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,          # ccxt enforces exchange rate limits
            'options': {'defaultType': 'spot'},
        })
        self._markets_loaded = False

    def _ensure_markets(self):
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def _fetch_with_retry(self, fn, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(f"Network error ({e.__class__.__name__}), "
                               f"retry {attempt + 1}/{self.max_retries} in {wait}s")
                time.sleep(wait)
        raise last_error

    # ── Fetching ─────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    since_ms: Optional[int] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """One page of OHLCV as a DataFrame with a UTC datetime `timestamp`."""
        self._ensure_markets()
        raw = self._fetch_with_retry(
            self.exchange.fetch_ohlcv, symbol, timeframe,
            since=since_ms, limit=limit)
        if not raw:
            return pd.DataFrame(
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = pd.DataFrame(
            raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df

    def fetch_ticker(self, symbol: str) -> Dict:
        """Current ticker (last price, bid, ask)."""
        self._ensure_markets()
        t = self._fetch_with_retry(self.exchange.fetch_ticker, symbol)
        return {'symbol': symbol, 'last': t.get('last'),
                'bid': t.get('bid'), 'ask': t.get('ask'),
                'timestamp': t.get('timestamp')}

    def fetch_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        self._ensure_markets()
        out = {}
        try:
            all_t = self._fetch_with_retry(self.exchange.fetch_tickers, symbols)
            for symbol in symbols:
                t = all_t.get(symbol, {})
                out[symbol] = {'last': t.get('last'), 'bid': t.get('bid'),
                               'ask': t.get('ask')}
        except Exception as e:
            logger.error(f"fetch_tickers failed: {e}")
        return out

    # ── Persisting ───────────────────────────────────────────────────────

    def update_symbol(self, symbol: str, timeframe: str,
                      lookback_bars: int = 300) -> int:
        """Fetch bars since the last stored one (or `lookback_bars` back) and
        store them. Returns number of rows submitted. Drops the still-forming
        last candle so only closed bars are stored."""
        if self.db is None:
            raise RuntimeError("MarketData.update_symbol requires a DatabaseManager")

        tf_sec = TIMEFRAME_SECONDS[timeframe]
        latest = self.db.get_latest_ohlcv_timestamp(symbol, timeframe)
        if latest:
            since_ms = (latest + tf_sec) * 1000
        else:
            since_ms = int((time.time() - lookback_bars * tf_sec) * 1000)

        df = self.fetch_ohlcv(symbol, timeframe, since_ms=since_ms)
        if df.empty:
            return 0

        # Remove the candle still in progress (its close time is in the future)
        now = pd.Timestamp.now(tz=timezone.utc)
        df = df[df['timestamp'] + pd.Timedelta(seconds=tf_sec) <= now]
        if df.empty:
            return 0

        return self.db.store_ohlcv(symbol, timeframe, df)

    def repair_gaps(self, symbol: str, timeframe: str, months: int = 36,
                    max_gaps: int = 200) -> int:
        """Detect interior holes (bar spacing > 1.5x the timeframe) inside the
        target window and refill each one. Interrupted backfills leave holes
        that endpoint-based resume logic cannot see — this closes them."""
        from sqlalchemy import text
        tf_sec = TIMEFRAME_SECONDS[timeframe]
        target_start = int((datetime.now(timezone.utc)
                            - timedelta(days=months * 30.5)).timestamp())

        with self.db.engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT timestamp FROM ohlcv WHERE symbol=:s AND timeframe=:tf "
                "AND timestamp >= :t0 ORDER BY timestamp"),
                {'s': symbol, 'tf': timeframe, 't0': target_start}).fetchall()
        if len(rows) < 2:
            return 0

        stamps = [r[0] for r in rows]
        gaps = [(prev, nxt) for prev, nxt in zip(stamps, stamps[1:])
                if nxt - prev > tf_sec * 1.5]
        if not gaps:
            return 0

        logger.info(f"{symbol} {timeframe}: {len(gaps)} interior gap(s) — "
                    f"largest {max(n - p for p, n in gaps) / 86400:.1f} days")
        filled = 0
        for prev, nxt in gaps[:max_gaps]:
            since_ms = (prev + tf_sec) * 1000
            while since_ms < nxt * 1000:
                df = self.fetch_ohlcv(symbol, timeframe,
                                      since_ms=since_ms, limit=1000)
                if df.empty:
                    break
                filled += self.db.store_ohlcv(symbol, timeframe, df)
                last_ms = int(df['timestamp'].iloc[-1].timestamp() * 1000)
                nxt_since = last_ms + tf_sec * 1000
                if nxt_since <= since_ms:
                    break
                since_ms = nxt_since
        logger.info(f"{symbol} {timeframe}: gap repair stored {filled} bars")
        return filled

    def _earliest_ohlcv_timestamp(self, symbol: str, timeframe: str):
        from sqlalchemy import text
        try:
            with self.db.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT MIN(timestamp) FROM ohlcv "
                    "WHERE symbol = :s AND timeframe = :tf"),
                    {'s': symbol, 'tf': timeframe}).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def backfill(self, symbol: str, timeframe: str, months: int = 12,
                 progress: bool = True) -> int:
        """Download `months` of history in pages and persist. Resumable —
        starts from the latest stored bar when one exists."""
        if self.db is None:
            raise RuntimeError("MarketData.backfill requires a DatabaseManager")

        tf_sec = TIMEFRAME_SECONDS[timeframe]
        target_start = datetime.now(timezone.utc) - timedelta(days=months * 30.5)
        target_start_epoch = int(target_start.timestamp())

        # Resume from the newest stored bar ONLY if coverage already reaches
        # back to the target start; otherwise begin at target_start (duplicate
        # bars are ignored on insert, so overlap is harmless).
        latest = self.db.get_latest_ohlcv_timestamp(symbol, timeframe)
        earliest = self._earliest_ohlcv_timestamp(symbol, timeframe)
        if latest and earliest and earliest <= target_start_epoch + 86400:
            since_ms = (latest + tf_sec) * 1000
        else:
            since_ms = target_start_epoch * 1000

        total = 0
        pages = 0
        while True:
            df = self.fetch_ohlcv(symbol, timeframe, since_ms=since_ms, limit=1000)
            if df.empty:
                break
            # Drop the still-forming candle
            now = pd.Timestamp.now(tz=timezone.utc)
            closed = df[df['timestamp'] + pd.Timedelta(seconds=tf_sec) <= now]
            if closed.empty:
                break
            total += self.db.store_ohlcv(symbol, timeframe, closed)
            pages += 1
            last_ms = int(df['timestamp'].iloc[-1].timestamp() * 1000)
            next_since = last_ms + tf_sec * 1000
            if next_since <= since_ms:   # no forward progress -> stop
                break
            since_ms = next_since
            if progress and pages % 25 == 0:
                logger.info(f"  {symbol} {timeframe}: {total} bars "
                            f"(up to {df['timestamp'].iloc[-1]})")
            if len(df) < 2:              # reached the live edge
                break
        logger.info(f"Backfill complete: {symbol} {timeframe} -> {total} bars")
        return total
