#!/usr/bin/env python3
"""
Main entry point for the AI Crypto Trading System.

    python src/main.py --mode paper     # paper trading (default)
    python src/main.py --mode live      # live trading (Phase 4 gates apply)

Orchestrates: trading loop, sentiment loop, pattern-discovery loop,
health server (:8080) and Prometheus metrics exporter (:9100).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path so `src.` imports work when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create required directories BEFORE logging attaches a FileHandler
for _dir in ('logs', 'data', 'models', 'reports'):
    Path(_dir).mkdir(exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler('logs/trading.log',
                            maxBytes=50 * 1024 * 1024, backupCount=5),
    ],
)
logger = logging.getLogger(__name__)

from src.trading.engine import TradingEngine
from src.models.ensemble import EnsembleModel
from src.sentiment.analyzer import SentimentAnalyzer
from src.patterns.discovery import PatternDiscoveryEngine
from src.utils.database import DatabaseManager
from src.utils.monitoring import MetricsCollector, HealthChecker
from src.utils.notifier import Notifier
from src.utils.config import Config


class AITradingSystem:
    """Main trading system orchestrator."""

    def __init__(self, config_path: str = 'config/trading.yaml'):
        self.config = self._load_config(config_path)
        self.running = False
        self.components = {}
        self._shutdown_event = asyncio.Event()
        self._shutdown_done = False

        self._initialize_components()

    def _load_config(self, config_path: str) -> Config:
        try:
            config = Config.load(config_path)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _initialize_components(self):
        logger.info("Initializing AI Trading System components...")
        try:
            self.components['db'] = DatabaseManager(self.config.database)
            self.components['metrics'] = MetricsCollector()
            self.components['health'] = HealthChecker()
            self.components['notifier'] = Notifier(db=self.components['db'])
            self.components['models'] = EnsembleModel(self.config)

            if self.config.features.get('sentiment_analysis', True):
                self.components['sentiment'] = SentimentAnalyzer(self.config.sentiment)

            if self.config.features.get('pattern_discovery', True):
                self.components['patterns'] = PatternDiscoveryEngine(
                    self.config.patterns, db=self.components['db'])

            if self.config.features.get('ml_core', False):
                from src.ml.live import MLCore
                self.components['ml'] = MLCore(
                    self.components['db'],
                    min_improvement=self.config.models.get('min_improvement', 0.02))

            self.components['trading'] = TradingEngine(
                config=self.config,
                models=self.components['models'],
                db=self.components['db'],
                metrics=self.components['metrics'],
                health=self.components['health'],
                notifier=self.components['notifier'],
            )

            for name in ('trading_engine', 'sentiment', 'patterns'):
                self.components['health'].register(name, timeout=1800)
            self.components['health'].heartbeat('sentiment', 'starting')
            self.components['health'].heartbeat('patterns', 'starting')

            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        logger.info("Starting AI Trading System...")
        self.running = True

        # Register signal handlers on the running loop (a sync signal handler
        # calling asyncio.create_task is unreliable; an Event is not)
        loop = asyncio.get_running_loop()
        import signal as _signal
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_shutdown, sig)
            except NotImplementedError:
                _signal.signal(sig, lambda s, f: self._request_shutdown(s))

        tasks = [
            asyncio.create_task(self._run_health_server(), name='health'),
            asyncio.create_task(self._run_metrics_exporter(), name='metrics'),
            asyncio.create_task(self._run_trading_loop(), name='trading'),
        ]
        if 'sentiment' in self.components:
            tasks.append(asyncio.create_task(
                self._run_sentiment_analyzer(), name='sentiment'))
        if 'patterns' in self.components:
            tasks.append(asyncio.create_task(
                self._run_pattern_discovery(), name='patterns'))
        tasks.append(asyncio.create_task(
            self._run_daily_summary(), name='daily_summary'))
        if 'ml' in self.components:
            tasks.append(asyncio.create_task(
                self._run_ml_loop(), name='ml_core'))

        self.components['health'].set_ready(True)
        engine_status = self.components['trading'].get_status()
        self.components['notifier'].alert(
            'system', 'info', 'Trading system started',
            f"mode={engine_status['mode']}, equity=${engine_status['equity']:,.2f}, "
            f"open positions={engine_status['open_positions']}"
            + (f", HALTED: {engine_status['halted']}" if engine_status['halted'] else ''),
            dedupe_key=f'start:{datetime.now(timezone.utc):%Y%m%d%H%M}')

        # Run until a shutdown is requested, then cancel the loops
        await self._shutdown_event.wait()
        self.running = False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.shutdown()

    def _request_shutdown(self, signum=None):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_event.set()

    async def shutdown(self):
        """Close positions, persist state, release resources. Runs once."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        logger.info("Shutting down AI Trading System...")
        self.running = False

        try:
            if 'trading' in self.components:
                await self.components['trading'].close_all_positions('shutdown')
        except Exception as e:
            logger.error(f"Error closing positions on shutdown: {e}")

        try:
            state = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'active_models': self.components['models'].get_active_models(),
                'engine_status': self.components['trading'].get_status(),
                'performance': _json_safe(
                    self.components['metrics'].get_summary()),
            }
            with open('data/shutdown_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("State saved to data/shutdown_state.json")
        except Exception as e:
            logger.error(f"Error saving shutdown state: {e}")

        try:
            self.components['db'].close()
        except Exception:
            pass

        logger.info("Shutdown complete")

    # ── Loops ────────────────────────────────────────────────────────────

    async def _run_trading_loop(self):
        logger.info("Starting trading loop...")
        interval = self.config.trading.get('cycle_interval', 60)

        while self.running:
            try:
                if not self._is_trading_enabled():
                    self.components['health'].heartbeat(
                        'trading_engine', 'idle (trading disabled)')
                    await asyncio.sleep(60)
                    continue

                await self.components['trading'].run_cycle()

                if await self._should_retrain():
                    await self._retrain_models()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                self.components['metrics'].record_error('trading_loop', str(e))
                self.components['notifier'].alert(
                    'error', 'warning', 'Trading loop error', str(e)[:500],
                    dedupe_key='trading_loop_error')
                await asyncio.sleep(60)

    async def _run_sentiment_analyzer(self):
        logger.info("Starting sentiment analyzer...")
        interval = self.config.sentiment.get('update_interval', 300)

        while self.running:
            try:
                symbols = self.config.trading.get('symbols', [])
                results = await self.components['sentiment'].analyze_batch(symbols)

                for symbol, sentiment in results.items():
                    self.components['db'].store_sentiment(symbol, sentiment)
                    self.components['metrics'].record_sentiment(symbol, sentiment)

                self.components['trading'].update_sentiment(results)
                self.components['health'].heartbeat(
                    'sentiment',
                    f"ok ({sum(r['volume'] for r in results.values())} items)")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Error in sentiment analyzer: {e}")
                self.components['metrics'].record_error('sentiment', str(e))
                await asyncio.sleep(300)

    async def _run_pattern_discovery(self):
        logger.info("Starting pattern discovery engine...")
        interval = self.config.patterns.get('discovery_interval', 3600)
        min_performance = self.config.patterns.get('min_performance', 0.02)

        # Let the trading loop warm up (and data land) before heavy ML work
        await asyncio.sleep(120)

        while self.running:
            try:
                symbols = self.config.trading.get('symbols', [])
                discovered = await self.components['patterns'].discover(symbols)

                activated = 0
                for pattern in discovered or []:
                    performance = await self._evaluate_pattern(pattern)
                    pattern['performance'] = performance
                    if performance > min_performance:
                        pattern['status'] = 'active'
                        activated += 1
                        self.components['trading'].add_pattern(pattern)
                    self.components['db'].store_pattern(pattern)

                if discovered:
                    logger.info(f"Pattern pass: {len(discovered)} candidates, "
                                f"{activated} activated (>{min_performance:.1%})")
                self.components['health'].heartbeat(
                    'patterns', f"ok ({len(discovered or [])} candidates)")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Error in pattern discovery: {e}")
                self.components['metrics'].record_error('patterns', str(e))
                await asyncio.sleep(3600)

    async def _run_health_server(self):
        from aiohttp import web

        async def health_check(request):
            status = await self.components['health'].check_all()
            return web.json_response(status,
                                     status=200 if status['healthy'] else 503)

        async def ready_check(request):
            status = await self.components['health'].check_ready()
            return web.json_response(status,
                                     status=200 if status['ready'] else 503)

        async def engine_status(request):
            return web.json_response(
                _json_safe(self.components['trading'].get_status()))

        app = web.Application()
        app.router.add_get('/health', health_check)
        app.router.add_get('/ready', ready_check)
        app.router.add_get('/status', engine_status)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        logger.info("Health check server started on port 8080")

        try:
            while self.running:
                await asyncio.sleep(1)
        finally:
            await runner.cleanup()

    async def _run_metrics_exporter(self):
        try:
            from prometheus_client import start_http_server
            start_http_server(9100)
            logger.info("Metrics exporter started on port 9100")
        except Exception as e:
            logger.warning(f"Prometheus exporter unavailable: {e}")

        while self.running:
            self.components['metrics'].update_system_metrics()
            await asyncio.sleep(10)

    async def _run_ml_loop(self):
        """ML learning loop: one prediction per UTC day, one champion/
        challenger retrain per week, every decision logged and alerted."""
        from src.ml.live import build_daily_frames
        logger.info("Starting ML learning loop (daily predict, weekly retrain)")
        ml = self.components['ml']
        db = self.components['db']
        engine = self.components['trading']
        notifier = self.components['notifier']
        last_pred_day = None

        while self.running:
            try:
                self.components['health'].heartbeat('ml_core')
                today = datetime.now(timezone.utc).date()

                # ── Weekly retrain (also bootstraps the first champion) ──
                last_retrain = ml.last_retrain_time()
                needs_retrain = (ml.bundle is None or last_retrain is None
                                 or (datetime.now(timezone.utc)
                                     - last_retrain).days >= 7)
                if needs_retrain:
                    frames = await asyncio.to_thread(
                        build_daily_frames, db,
                        self.config.trading.get('symbols', []))
                    record = await asyncio.to_thread(ml.retrain, frames)
                    notifier.alert(
                        'ml', 'info',
                        f"ML retrain: {record['decision']}",
                        record['reason'],
                        dedupe_key=f"ml_retrain:{record['new_version']}")

                # ── Daily prediction + paper trades ──
                if ml.bundle is not None and last_pred_day != today:
                    # live prices must exist before signals can fill; if the
                    # exchange is unreachable, retry next hour (do NOT burn
                    # today's prediction slot on an unexecutable pass)
                    await engine._refresh_prices()
                    if not engine.latest_prices:
                        logger.warning("ML loop: no live prices yet — "
                                       "retrying next hour")
                        await asyncio.sleep(3600)
                        continue
                    frames = await asyncio.to_thread(
                        build_daily_frames, db,
                        self.config.trading.get('symbols', []))
                    predictions = await asyncio.to_thread(ml.predict, frames)
                    executed = {}
                    for symbol, p in predictions.items():
                        action = 'BUY' if p['pred'] == 'UP' else 'SELL'
                        engine._handle_signal({
                            'symbol': symbol, 'action': action,
                            'size': 0.10, 'confidence': 0.99,
                            'stop_loss': None, 'take_profit': None,
                            'metadata': {'strategy': 'ml_core',
                                         'p_up': p['p_up'],
                                         'p_down': p['p_down'],
                                         'model_version': p['model_version']},
                        })
                        executed[symbol] = (
                            (action == 'BUY' and symbol in engine.positions) or
                            (action == 'SELL' and symbol not in engine.positions))
                    ml.store_predictions(predictions, executed)
                    last_pred_day = today
                    logger.info("ML daily decisions: " + ", ".join(
                        f"{s.split('/')[0]}={p['pred']}"
                        for s, p in predictions.items()))

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"ML loop error: {e}")
                self.components['metrics'].record_error('ml_core', str(e))

            await asyncio.sleep(3600)   # hourly wake-up; acts once per day

    async def _run_daily_summary(self):
        """Send the daily P&L summary shortly after UTC midnight."""
        from datetime import timedelta
        started = datetime.now(timezone.utc)
        while self.running:
            now = datetime.now(timezone.utc)
            next_run = (now + timedelta(days=1)).replace(
                hour=0, minute=5, second=0, microsecond=0)
            try:
                await asyncio.sleep((next_run - now).total_seconds())
            except asyncio.CancelledError:
                raise
            if not self.running:
                break
            try:
                summary = self._build_daily_summary(started)
                self.components['notifier'].daily_summary(summary)
            except Exception as e:
                logger.error(f"Daily summary failed: {e}")

    def _build_daily_summary(self, started_at) -> dict:
        from datetime import timedelta
        db = self.components['db']
        engine = self.components['trading']
        day_ago = datetime.now(timezone.utc) - timedelta(days=1)

        recent = db.get_recent_trades(500)
        if not recent.empty:
            closed = recent[recent['status'] == 'CLOSED']
            day_trades = closed[closed['exit_time'] >=
                                day_ago.replace(tzinfo=None)]
        else:
            day_trades = recent

        equity_df = db.get_equity_curve(start_date=day_ago)
        status = engine.get_status()
        daily_return = 0.0
        if not equity_df.empty and len(equity_df) > 1:
            daily_return = float(equity_df['total_equity'].iloc[-1]
                                 / equity_df['total_equity'].iloc[0] - 1)
        drawdown = 0.0
        start_equity = engine._starting_equity()
        if start_equity:
            drawdown = status['equity'] / start_equity - 1

        has_trades = not day_trades.empty
        return {
            'equity': status['equity'],
            'daily_return': daily_return,
            'n_trades': len(day_trades),
            'wins': int((day_trades['pnl'] > 0).sum()) if has_trades else 0,
            'losses': int((day_trades['pnl'] <= 0).sum()) if has_trades else 0,
            'realized_pnl': float(day_trades['pnl'].sum()) if has_trades else 0.0,
            'fees': float(day_trades['commission'].sum()) if has_trades else 0.0,
            'open_positions': status['open_positions'],
            'drawdown': min(drawdown, 0.0),
            'mode': status['mode'],
            'uptime_hours': (datetime.now(timezone.utc)
                             - started_at).total_seconds() / 3600,
        }

    # ── Policy helpers ───────────────────────────────────────────────────

    def _is_trading_enabled(self) -> bool:
        """Paper OR live must be enabled (main() sets these from --mode)."""
        live = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
        paper = os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
        return live or paper

    async def _should_retrain(self) -> bool:
        db = self.components['db']
        last_retrain = db.get_last_retrain_time()
        if not last_retrain:
            return True

        hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
        if hours_since < self.config.models.get('retrain_interval_hours', 24):
            return False

        # Performance degradation triggers early retraining — but only when
        # there is enough data for the Sharpe estimate to mean anything.
        recent = self.components['metrics'].get_recent_performance()
        if recent['sufficient_data'] and \
                recent['sharpe_ratio'] < self.config.models.get('min_sharpe_ratio', 1.0):
            logger.info("Performance degradation detected, retraining needed")
            return True

        trade_count = db.get_trade_count_since(last_retrain)
        return trade_count >= self.config.models.get('min_trades_for_retrain', 100)

    async def _retrain_models(self):
        logger.info("Starting model retraining...")
        try:
            training_data = self.components['db'].get_training_data()
            results = await self.components['models'].retrain(training_data)

            min_improvement = self.config.models.get('min_improvement', 0.02)
            if results['improvement'] > min_improvement:
                logger.info(f"Model retraining successful, improvement: "
                            f"{results['improvement']:.2%}")
                self.components['db'].save_model_version(results)
            else:
                logger.info("Retraining did not improve performance "
                            f"(>{min_improvement:.0%} required), keeping current models")
                # Record the attempt so the retrain timer resets
                self.components['db'].save_model_version({**results,
                                                          'is_active': False})
        except Exception as e:
            logger.exception(f"Model retraining failed: {e}")
            self.components['metrics'].record_error('model_retraining', str(e))

    async def _evaluate_pattern(self, pattern: dict) -> float:
        """Mini-backtest: how did price move after this pattern historically?

        Scores the pattern's expected edge as the mean forward return over the
        next `horizon` bars each time the pattern's direction fired, on the
        symbol's real stored history. Bullish patterns score +mean, bearish
        score -mean (a profitable short signal is still an edge, used to
        veto/confirm longs)."""
        try:
            symbol = pattern.get('symbol', 'BTC/USDT')
            timeframe = pattern.get('timeframe',
                                    self.config.patterns.get('timeframe', '15m'))
            df = self.components['db'].get_ohlcv_data(symbol, timeframe, limit=2000)
            if df.empty or len(df) < 300:
                return 0.0

            closes = df.sort_values('timestamp')['close'].reset_index(drop=True)
            horizon = 12  # bars ahead (~3h on 15m)
            forward = closes.shift(-horizon) / closes - 1

            # Proxy for pattern occurrences: bars whose recent move matches the
            # pattern's direction and magnitude profile
            window = int(pattern.get('pattern_config', {}).get('window_size', 50))
            recent_move = closes / closes.shift(window) - 1

            bearish = pattern.get('pattern_type', '') in {
                'head_shoulders', 'double_top', 'flag_bearish', 'breakdown',
                'triangle_descending', 'channel_down', 'reversal_bearish',
                'continuation_bearish', 'wedge_rising'}

            mask = (recent_move < -0.01) if bearish else (recent_move > 0.01)
            sample = forward[mask].dropna()
            if len(sample) < 20:
                return 0.0

            edge = float(sample.mean())
            return -edge if bearish else edge
        except Exception as e:
            logger.error(f"Pattern evaluation failed: {e}")
            return 0.0


def _json_safe(obj):
    """Round-trip through JSON with str() fallback for datetimes/numpy."""
    return json.loads(json.dumps(obj, default=str))


def parse_arguments():
    parser = argparse.ArgumentParser(description='AI Crypto Trading System')
    parser.add_argument('--config', type=str, default='config/trading.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str,
                        choices=['live', 'paper', 'backtest'], default='paper',
                        help='Trading mode')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols to trade (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


async def main():
    args = parse_arguments()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mode == 'live':
        os.environ['ENABLE_LIVE_TRADING'] = 'true'
        os.environ['ENABLE_PAPER_TRADING'] = 'false'
    elif args.mode == 'paper':
        os.environ['ENABLE_LIVE_TRADING'] = 'false'
        os.environ['ENABLE_PAPER_TRADING'] = 'true'
    elif args.mode == 'backtest':
        print("For backtesting use: python scripts/run_backtest.py --help")
        return

    system = AITradingSystem(args.config)

    if args.symbols:
        system.config.trading['symbols'] = args.symbols
        system.components['trading'].symbols = args.symbols

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await system.shutdown()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        await system.shutdown()
        raise


if __name__ == "__main__":
    asyncio.run(main())
