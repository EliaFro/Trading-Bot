"""
Monitoring utilities: runtime metrics collection and component health checks.

MetricsCollector — in-memory time series + optional Prometheus gauges.
HealthChecker   — components report heartbeats; /health and /ready aggregate them.
"""

import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:  # metrics still collected in-memory
    PROMETHEUS_AVAILABLE = False

# Prometheus metrics live in a process-global registry, so create them once
# at module level (multiple MetricsCollector instances share them).
if PROMETHEUS_AVAILABLE:
    _PROM = {
        'equity': Gauge('trading_equity_usdt', 'Account equity in USDT'),
        'positions': Gauge('trading_open_positions', 'Open positions'),
        'sentiment': Gauge('trading_sentiment', 'Sentiment score', ['symbol']),
        'errors': Counter('trading_errors_total', 'Errors', ['component']),
        'trades': Counter('trading_trades_total', 'Trades executed',
                          ['symbol', 'side']),
        'cycle': Gauge('trading_cycle_seconds', 'Trading cycle duration'),
    }


class MetricsCollector:
    """Collects runtime metrics from all components.

    Keeps bounded in-memory series for the dashboard/health endpoints and
    mirrors key values to Prometheus gauges when prometheus_client is present.
    """

    MAX_POINTS = 10_000

    def __init__(self):
        self._series: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_POINTS))
        self._errors: deque = deque(maxlen=1000)
        self._counters: Dict[str, int] = defaultdict(int)

        if PROMETHEUS_AVAILABLE:
            self._prom_equity = _PROM['equity']
            self._prom_positions = _PROM['positions']
            self._prom_sentiment = _PROM['sentiment']
            self._prom_errors = _PROM['errors']
            self._prom_trades = _PROM['trades']
            self._prom_cycle = _PROM['cycle']

    # ── Recording ────────────────────────────────────────────────────────

    def record_metric(self, name: str, value: float,
                      timestamp: Optional[datetime] = None):
        self._series[name].append((timestamp or datetime.utcnow(), float(value)))

    def record_equity(self, equity: float, open_positions: int = 0):
        self.record_metric('equity', equity)
        self.record_metric('open_positions', open_positions)
        if PROMETHEUS_AVAILABLE:
            self._prom_equity.set(equity)
            self._prom_positions.set(open_positions)

    def record_trade(self, symbol: str, side: str, pnl: Optional[float] = None):
        self._counters['trades'] += 1
        if pnl is not None:
            self.record_metric('trade_pnl', pnl)
        if PROMETHEUS_AVAILABLE:
            self._prom_trades.labels(symbol=symbol, side=side).inc()

    def record_sentiment(self, symbol: str, sentiment: Dict):
        score = sentiment.get('sentiment', 0.0) if isinstance(sentiment, dict) else float(sentiment)
        self.record_metric(f'sentiment_{symbol}', score)
        if PROMETHEUS_AVAILABLE:
            self._prom_sentiment.labels(symbol=symbol).set(score)

    def record_error(self, component: str, message: str):
        self._errors.append((datetime.utcnow(), component, str(message)[:500]))
        self._counters[f'errors_{component}'] += 1
        if PROMETHEUS_AVAILABLE:
            self._prom_errors.labels(component=component).inc()

    def record_cycle_duration(self, seconds: float):
        self.record_metric('cycle_duration', seconds)
        if PROMETHEUS_AVAILABLE:
            self._prom_cycle.set(seconds)

    def update_system_metrics(self):
        """Periodic system-level gauges (called from the metrics exporter loop)."""
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            self.record_metric('memory_mb', rss_mb)
        except Exception:
            pass

    # ── Reading ──────────────────────────────────────────────────────────

    def get_latest(self, name: str) -> Optional[float]:
        series = self._series.get(name)
        return series[-1][1] if series else None

    def get_series(self, name: str, since: Optional[datetime] = None):
        points = list(self._series.get(name, []))
        if since:
            points = [p for p in points if p[0] >= since]
        return points

    def get_recent_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Sharpe/return over recent equity points. `sufficient_data` tells the
        caller whether the numbers mean anything (avoids retraining on noise)."""
        since = datetime.utcnow() - timedelta(hours=hours)
        equity = [v for t, v in self.get_series('equity', since)]

        result = {'sharpe_ratio': 0.0, 'total_return': 0.0,
                  'n_points': len(equity), 'sufficient_data': False}
        if len(equity) < 30:
            return result

        values = np.asarray(equity, dtype=float)
        returns = np.diff(values) / values[:-1]
        if returns.std() > 0:
            # cycles are ~1min; annualize per-cycle Sharpe to daily-equivalent
            result['sharpe_ratio'] = float(
                np.sqrt(365 * 24 * 60) * returns.mean() / returns.std())
        result['total_return'] = float(values[-1] / values[0] - 1)
        result['sufficient_data'] = True
        return result

    def get_recent_errors(self, limit: int = 20):
        return [{'time': t.isoformat(), 'component': c, 'message': m}
                for t, c, m in list(self._errors)[-limit:]]

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {'counters': dict(self._counters)}
        for name, series in self._series.items():
            if series:
                values = [v for _, v in series]
                summary[name] = {
                    'latest': values[-1],
                    'mean': float(np.mean(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                }
        summary['recent_errors'] = self.get_recent_errors(5)
        return summary


class HealthChecker:
    """Aggregates component heartbeats for /health and /ready endpoints.

    Components call heartbeat(name) each loop iteration; a component is
    unhealthy if it hasn't reported within its timeout.
    """

    def __init__(self, default_timeout: float = 300.0):
        self.default_timeout = default_timeout
        self._components: Dict[str, Dict[str, Any]] = {}
        self._started_at = time.time()
        self._ready = False

    def register(self, name: str, timeout: Optional[float] = None):
        self._components[name] = {
            'last_beat': None,
            'timeout': timeout or self.default_timeout,
            'detail': 'registered',
        }

    def heartbeat(self, name: str, detail: str = 'ok'):
        if name not in self._components:
            self.register(name)
        self._components[name]['last_beat'] = time.time()
        self._components[name]['detail'] = detail

    def set_ready(self, ready: bool = True):
        self._ready = ready

    async def check_all(self) -> Dict[str, Any]:
        now = time.time()
        components = {}
        healthy = True
        for name, state in self._components.items():
            last = state['last_beat']
            component_ok = last is not None and (now - last) < state['timeout']
            if not component_ok:
                healthy = False
            components[name] = {
                'healthy': component_ok,
                'seconds_since_heartbeat': round(now - last, 1) if last else None,
                'detail': state['detail'],
            }
        return {
            'healthy': healthy,
            'uptime_seconds': round(now - self._started_at, 1),
            'components': components,
            'timestamp': datetime.utcnow().isoformat(),
        }

    async def check_ready(self) -> Dict[str, Any]:
        return {'ready': self._ready,
                'timestamp': datetime.utcnow().isoformat()}
