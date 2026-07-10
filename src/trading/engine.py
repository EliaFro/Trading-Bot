"""
TradingEngine — paper and live execution for the AI Crypto Trading System.

Paper mode (default) simulates fills realistically:
  * LIMIT order semantics: orders rest until the market trades through the
    limit price; unfilled orders are cancelled after execution.order_timeout_seconds
  * commission (execution.commission_rate, default 0.1%) on every fill
  * slippage  (execution.slippage_rate, default 0.05%) against the fill price
  * cash and open positions persist in the database exactly like live mode,
    so a restart resumes where it left off

Live mode (Phase 4) submits real ccxt orders and is gated by safety checks
(withdrawals disabled on the API key, .env permissions, kill switch clear).

v1 is LONG-ONLY SPOT: SELL signals close positions, they never open shorts.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.data.market_data import MarketData
from src.trading import safety
from src.trading.safety import RiskLimits, SafetyError

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TradingEngine:
    """Runs the trade lifecycle each cycle: data → signals → orders → accounting."""

    DRAWDOWN_ALERT_LEVEL = 0.08   # notify at -8% from starting equity

    def __init__(self, config, models, db, metrics, health=None, notifier=None):
        self.config = config
        self.models = models
        self.db = db
        self.metrics = metrics
        self.health = health
        self.notifier = notifier
        self._dd_alert_active = False

        trading_cfg = config.trading
        self.symbols: List[str] = trading_cfg.get('symbols', ['BTC/USDT'])
        self.timeframes: List[str] = trading_cfg.get('timeframes', ['5m'])
        self.signal_timeframe: str = trading_cfg.get('signal_timeframe', '5m')
        self.lookback_bars: int = int(trading_cfg.get('lookback_bars', 200))
        self.initial_capital: float = float(trading_cfg.get('initial_capital', 10000.0))

        exec_cfg = config.execution
        self.commission_rate = float(exec_cfg.get('commission_rate', 0.001))
        self.slippage_rate = float(exec_cfg.get('slippage_rate', 0.0005))
        self.slippage_tolerance = float(exec_cfg.get('slippage_tolerance', 0.001))
        self.order_timeout = int(exec_cfg.get('order_timeout_seconds', 90))

        self.limits = RiskLimits.from_config(config.risk_management)
        self.min_confidence = float(config.strategies.get('min_confidence', 0.55))

        self.live_mode = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
        self.mode = 'live' if self.live_mode else 'paper'

        # Market data (public endpoints; live orders use a keyed exchange)
        self.market_data = MarketData(db=db)
        self.exchange = None  # keyed ccxt client, live mode only

        # Runtime state
        self.positions: Dict[str, Dict] = {}       # symbol -> open trade dict
        self.pending_orders: List[Dict] = []
        self.latest_prices: Dict[str, float] = {}
        self.sentiment: Dict[str, Dict] = {}
        self.active_patterns: List[Dict] = []
        self.cash: float = self.initial_capital
        self.halted: Optional[str] = None           # reason string when halted
        self._cycle_count = 0

        self._startup_checks()
        self._restore_state()

        logger.info(
            f"TradingEngine ready: mode={self.mode}, symbols={self.symbols}, "
            f"signal_tf={self.signal_timeframe}, cash={self.cash:.2f}, "
            f"open_positions={len(self.positions)}, "
            f"limits={self.limits}")

    # ── Startup ──────────────────────────────────────────────────────────

    def _startup_checks(self):
        """Safety gates. Live mode fails hard; paper mode logs warnings."""
        safety.check_env_file_permissions('.env', live_mode=self.live_mode)

        reason = safety.is_kill_switch_triggered(self.db)
        if reason:
            self.halted = f"kill switch: {reason}"
            logger.critical(
                f"Trading HALTED — {self.halted}. "
                f"Run scripts/clear_kill_switch.py after reviewing.")

        if self.live_mode:
            self._init_live_exchange()

    def _init_live_exchange(self):
        import ccxt
        keys = self.config.api_keys.get('binance', {})
        use_testnet = keys.get('use_testnet', False)
        api_key = keys.get('testnet_api_key') if use_testnet else keys.get('api_key')
        secret = keys.get('testnet_secret') if use_testnet else keys.get('secret')
        if not api_key or not secret:
            raise SafetyError("Live mode requires BINANCE_API_KEY/BINANCE_SECRET in .env")

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        if use_testnet:
            self.exchange.set_sandbox_mode(True)
            logger.warning("Live engine connected to Binance TESTNET")
        self.exchange.load_markets()

        # Hard refusal if the key can withdraw funds (skip on testnet — the
        # endpoint doesn't exist there and testnet funds are play money).
        if not use_testnet:
            safety.check_withdrawal_permissions(self.exchange, live_mode=True)

    def _restore_state(self):
        """Recover cash and open positions from the database after a restart."""
        # Cash
        try:
            with self.db.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT config_value FROM system_config WHERE config_key = :k"),
                    {'k': f'{self.mode}_cash'}).fetchone()
            if row:
                self.cash = float(row[0])
        except Exception as e:
            logger.warning(f"Could not restore cash state: {e}")

        # Open positions
        try:
            open_trades = self.db.get_active_positions()
            for _, row in open_trades.iterrows():
                self.positions[row['symbol']] = {
                    'id': row['id'],
                    'symbol': row['symbol'],
                    'side': row['side'],
                    'quantity': float(row['quantity']),
                    'entry_price': float(row['entry_price']),
                    'stop_loss': float(row['stop_loss']) if row['stop_loss'] else None,
                    'take_profit': float(row['take_profit']) if row['take_profit'] else None,
                    'strategy': row.get('strategy', 'unknown'),
                    'entry_time': row['entry_time'],
                }
            if self.positions:
                logger.info(f"Restored {len(self.positions)} open positions from DB")
        except Exception as e:
            logger.warning(f"Could not restore positions: {e}")

    def _persist_cash(self):
        try:
            with self.db.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO system_config (config_key, config_value,
                                               config_type, description, updated_at)
                    VALUES (:k, :v, 'number', 'Broker cash balance', :ts)
                    ON CONFLICT(config_key) DO UPDATE
                    SET config_value = :v, updated_at = :ts
                """), {'k': f'{self.mode}_cash', 'v': str(self.cash),
                       'ts': int(_utcnow().timestamp())})
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist cash: {e}")

    # ── Public API (called by AITradingSystem) ───────────────────────────

    def update_sentiment(self, sentiment_results: Dict[str, Dict]):
        self.sentiment.update(sentiment_results or {})

    def add_pattern(self, pattern: Dict):
        self.active_patterns.append(pattern)
        if len(self.active_patterns) > 50:
            self.active_patterns = self.active_patterns[-50:]

    async def run_cycle(self):
        """One full trading cycle. Never raises — errors are recorded."""
        started = _utcnow()
        self._cycle_count += 1
        try:
            if self.health:
                self.health.heartbeat('trading_engine')

            await self._update_market_data()
            await self._refresh_prices()

            if not self.latest_prices:
                logger.warning("No market prices available — skipping cycle")
                return

            self._process_pending_orders()
            self._check_stops()

            gate_reason = self._risk_gate()
            if gate_reason is None:
                await self._generate_and_execute_signals()
            elif self._cycle_count % 10 == 1:
                logger.warning(f"New trades blocked: {gate_reason}")

            self._record_equity()
            self._persist_cash()

        except Exception as e:
            logger.exception(f"Trading cycle error: {e}")
            if self.metrics:
                self.metrics.record_error('trading_engine', str(e))
        finally:
            elapsed = (_utcnow() - started).total_seconds()
            if self.metrics:
                self.metrics.record_cycle_duration(elapsed)

    async def close_all_positions(self, reason: str = 'shutdown'):
        """Close every open position at market. Used by shutdown and kill switch."""
        if not self.positions:
            return
        logger.info(f"Closing all {len(self.positions)} positions ({reason})")
        await self._refresh_prices()
        for symbol in list(self.positions.keys()):
            price = self.latest_prices.get(symbol)
            if price:
                self._close_position(symbol, price, reason)
            else:
                logger.error(f"No price for {symbol}; position left OPEN in DB "
                             f"for manual review")
        self._persist_cash()
        self._record_equity()

    def get_status(self) -> Dict[str, Any]:
        equity = self._total_equity()
        return {
            'mode': self.mode,
            'halted': self.halted,
            'cash': round(self.cash, 2),
            'equity': round(equity, 2),
            'open_positions': len(self.positions),
            'pending_orders': len(self.pending_orders),
            'cycle_count': self._cycle_count,
        }

    # ── Market data ──────────────────────────────────────────────────────

    async def _update_market_data(self):
        """Incrementally pull new closed bars for every symbol/timeframe."""
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    await asyncio.to_thread(
                        self.market_data.update_symbol, symbol, timeframe,
                        self.lookback_bars)
                except Exception as e:
                    logger.error(f"Data update failed {symbol} {timeframe}: {e}")
                    if self.metrics:
                        self.metrics.record_error('market_data', str(e))

    async def _refresh_prices(self):
        try:
            tickers = await asyncio.to_thread(
                self.market_data.fetch_tickers, self.symbols)
            for symbol, t in tickers.items():
                if t.get('last'):
                    self.latest_prices[symbol] = float(t['last'])
        except Exception as e:
            logger.error(f"Ticker refresh failed: {e}")

    # ── Risk gates ───────────────────────────────────────────────────────

    def _risk_gate(self) -> Optional[str]:
        """Return a reason string if NEW trades must be blocked, else None."""
        if self.halted:
            return self.halted

        equity = self._total_equity()

        # Kill switch: equity below (1 - max_drawdown) x starting balance
        start = self._starting_equity()
        if start and equity < start * (1 - self.limits.max_drawdown):
            reason = (f"equity {equity:.2f} below kill-switch level "
                      f"{start * (1 - self.limits.max_drawdown):.2f} "
                      f"(start {start:.2f}, max_dd {self.limits.max_drawdown:.0%})")
            safety.trigger_kill_switch(self.db, reason)
            self.halted = f"kill switch: {reason}"
            if self.notifier:
                self.notifier.alert('risk', 'critical', 'KILL SWITCH TRIGGERED',
                                    reason, dedupe_key='kill_switch')
            else:
                self.db.store_alert('risk', 'critical', 'KILL SWITCH', reason)
            asyncio.get_event_loop().create_task(
                self.close_all_positions('kill_switch'))
            return self.halted

        # Daily loss limit: no new trades after -daily_loss_limit today
        day_start_equity = self._day_start_equity()
        if day_start_equity and equity < day_start_equity * (1 - self.limits.daily_loss_limit):
            return (f"daily loss limit hit: equity {equity:.2f} vs day start "
                    f"{day_start_equity:.2f} (-{self.limits.daily_loss_limit:.0%})")

        # Max concurrent positions
        if len(self.positions) >= self.limits.max_concurrent_positions:
            return (f"max concurrent positions "
                    f"({self.limits.max_concurrent_positions}) reached")

        return None

    def _starting_equity(self) -> Optional[float]:
        try:
            with self.db.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT config_value FROM system_config WHERE config_key = :k"),
                    {'k': f'{self.mode}_starting_equity'}).fetchone()
            if row:
                return float(row[0])
            # First run: record current equity as the baseline
            equity = self._total_equity()
            with self.db.engine.connect() as conn:
                conn.execute(text("""
                    INSERT OR IGNORE INTO system_config
                    (config_key, config_value, config_type, description, updated_at)
                    VALUES (:k, :v, 'number', 'Kill-switch baseline equity', :ts)
                """), {'k': f'{self.mode}_starting_equity', 'v': str(equity),
                       'ts': int(_utcnow().timestamp())})
                conn.commit()
            return equity
        except Exception as e:
            logger.error(f"starting_equity lookup failed: {e}")
            return None

    def _day_start_equity(self) -> Optional[float]:
        try:
            day_start = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            with self.db.engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT total_equity FROM performance_tracking
                    WHERE timestamp >= :ts AND mode = :mode
                    ORDER BY timestamp ASC LIMIT 1
                """), {'ts': int(day_start.timestamp()), 'mode': self.mode}).fetchone()
            return float(row[0]) if row else None
        except Exception:
            return None

    # ── Signals ──────────────────────────────────────────────────────────

    async def _generate_and_execute_signals(self):
        for symbol in self.symbols:
            try:
                df = await asyncio.to_thread(
                    self.db.get_ohlcv_data, symbol, self.signal_timeframe,
                    None, None, self.lookback_bars)
                if df.empty or len(df) < 60:
                    logger.debug(f"Not enough history for {symbol} "
                                 f"({len(df)} bars) — need 60+")
                    continue

                signals = self.models.generate_signals(
                    symbol=symbol,
                    data=df,
                    positions=self.positions,
                    portfolio_value=self._total_equity(),
                    sentiment=self.sentiment.get(symbol),
                    patterns=self.active_patterns,
                )

                for signal in signals or []:
                    self._handle_signal(signal)

            except Exception as e:
                logger.exception(f"Signal generation failed for {symbol}: {e}")
                if self.metrics:
                    self.metrics.record_error('signals', str(e))

    def _handle_signal(self, signal: Dict):
        symbol = signal.get('symbol')
        action = signal.get('action')
        confidence = float(signal.get('confidence', 0) or 0)
        executed = False

        if action == 'BUY' and confidence >= self.min_confidence \
                and symbol not in self.positions:
            executed = self._open_position(signal)
        elif action == 'SELL' and symbol in self.positions:
            price = self.latest_prices.get(symbol)
            if price:
                self._close_position(symbol, price, 'signal')
                executed = True
        # SELL without a position is ignored: long-only spot.

        signal['executed'] = executed
        signal['timestamp'] = int(_utcnow().timestamp())
        self.db.store_signal(signal)

    # ── Order placement / fills (paper) ──────────────────────────────────

    def _open_position(self, signal: Dict) -> bool:
        symbol = signal['symbol']
        price = self.latest_prices.get(symbol)
        if not price:
            return False

        equity = self._total_equity()
        size_fraction = min(float(signal.get('size', self.limits.max_position_size)),
                            self.limits.max_position_size)
        notional = min(equity * size_fraction, self.cash / (1 + self.commission_rate))

        if notional < safety.MIN_ORDER_NOTIONAL_USDT:
            logger.debug(f"Order below min notional for {symbol}: {notional:.2f}")
            return False

        limit_price = price * (1 + self.slippage_tolerance)  # marketable limit
        quantity = notional / limit_price

        if self.live_mode:
            return self._submit_live_order(signal, 'BUY', quantity, limit_price)

        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'side': 'BUY',
            'order_type': 'LIMIT',
            'quantity': quantity,
            'price': limit_price,
            'created_at': _utcnow(),
            'signal': signal,
        }
        self.pending_orders.append(order)
        self._persist_order(order, 'PENDING')
        # Try to fill immediately against the current market
        self._try_fill_order(order)
        return True

    def _process_pending_orders(self):
        """Fill or expire resting LIMIT orders against current prices."""
        still_pending = []
        for order in self.pending_orders:
            if order.get('filled'):
                continue
            age = (_utcnow() - order['created_at']).total_seconds()
            if age > self.order_timeout:
                self._persist_order(order, 'CANCELLED')
                logger.info(f"LIMIT order expired unfilled: {order['symbol']} "
                            f"{order['side']} @ {order['price']:.2f}")
                continue
            if not self._try_fill_order(order):
                still_pending.append(order)
        self.pending_orders = still_pending

    def _try_fill_order(self, order: Dict) -> bool:
        """LIMIT semantics: BUY fills when market <= limit, SELL when >= limit.
        Fill price includes slippage, capped by the limit price."""
        market = self.latest_prices.get(order['symbol'])
        if not market:
            return False

        if order['side'] == 'BUY':
            if market > order['price']:
                return False
            fill_price = min(market * (1 + self.slippage_rate), order['price'])
        else:  # SELL
            if market < order['price']:
                return False
            fill_price = max(market * (1 - self.slippage_rate), order['price'])

        fill_value = order['quantity'] * fill_price
        commission = fill_value * self.commission_rate
        slippage_cost = abs(fill_price - market) * order['quantity']

        if order['side'] == 'BUY':
            total_cost = fill_value + commission
            if total_cost > self.cash:
                logger.debug(f"Insufficient cash for {order['symbol']} fill "
                             f"({total_cost:.2f} > {self.cash:.2f})")
                self._persist_order(order, 'REJECTED')
                order['filled'] = True   # drop from pending
                return True
            self.cash -= total_cost
            self._record_open_trade(order, fill_price, commission, slippage_cost)
        else:
            self.cash += fill_value - commission
            # SELL fills belong to closes; trade record updated by caller

        order['filled'] = True
        order['fill_price'] = fill_price
        order['commission'] = commission
        self._persist_order(order, 'FILLED', fill_price, commission, slippage_cost)
        return True

    def _record_open_trade(self, order: Dict, fill_price: float,
                           commission: float, slippage_cost: float):
        signal = order.get('signal', {})
        symbol = order['symbol']
        trade_id = str(uuid.uuid4())
        entry_time = _utcnow()

        position = {
            'id': trade_id,
            'symbol': symbol,
            'side': 'BUY',
            'quantity': order['quantity'],
            'entry_price': fill_price,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'strategy': (signal.get('metadata') or {}).get('strategy', 'ensemble'),
            'entry_time': entry_time,
        }
        self.positions[symbol] = position

        self.db.store_trade({
            'id': trade_id,
            'symbol': symbol,
            'side': 'BUY',
            'quantity': order['quantity'],
            'entry_price': fill_price,
            'exit_price': None,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'pnl': None,
            'pnl_percentage': None,
            'commission': commission,
            'slippage': slippage_cost,
            'strategy': position['strategy'],
            'features': (signal.get('metadata') or {}),
            'entry_time': entry_time,
            'exit_time': None,
            'status': 'OPEN',
        })
        if self.metrics:
            self.metrics.record_trade(symbol, 'BUY')
        logger.info(f"OPENED {symbol}: {order['quantity']:.6f} @ {fill_price:.2f} "
                    f"(SL {position['stop_loss']}, TP {position['take_profit']}, "
                    f"strategy {position['strategy']})")

    def _check_stops(self):
        """Close positions whose stop-loss or take-profit level was hit."""
        for symbol in list(self.positions.keys()):
            price = self.latest_prices.get(symbol)
            if not price:
                continue
            pos = self.positions[symbol]
            if pos.get('stop_loss') and price <= pos['stop_loss']:
                self._close_position(symbol, price, 'stop_loss')
            elif pos.get('take_profit') and price >= pos['take_profit']:
                self._close_position(symbol, price, 'take_profit')

    def _close_position(self, symbol: str, market_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        if self.live_mode:
            self._submit_live_close(pos, reason)
            return

        # Stops execute like market orders (slippage against us);
        # signal closes go out as marketable limits (same fill model).
        fill_price = market_price * (1 - self.slippage_rate)
        fill_value = pos['quantity'] * fill_price
        commission = fill_value * self.commission_rate
        slippage_cost = (market_price - fill_price) * pos['quantity']

        entry_value = pos['quantity'] * pos['entry_price']
        pnl = fill_value - commission - entry_value
        pnl_pct = pnl / entry_value if entry_value else 0.0

        self.cash += fill_value - commission

        self.db.update_trade(pos['id'], {
            'exit_price': fill_price,
            'exit_time': _utcnow(),
            'pnl': pnl,
            'pnl_percentage': pnl_pct,
            'status': 'CLOSED',
            'exit_reason': reason,
        })
        if self.metrics:
            self.metrics.record_trade(symbol, 'SELL', pnl=pnl)
        logger.info(f"CLOSED {symbol} ({reason}): {pos['quantity']:.6f} @ "
                    f"{fill_price:.2f}, PnL {pnl:+.2f} ({pnl_pct:+.2%})")

        if reason == 'stop_loss':
            self.db.store_alert('risk', 'warning', f'Stop-loss hit: {symbol}',
                                f'PnL {pnl:+.2f} ({pnl_pct:+.2%})', symbol=symbol)

    # ── Live order path (fully wired in Phase 4) ─────────────────────────

    def _submit_live_order(self, signal, side, quantity, limit_price) -> bool:
        """Live ccxt order with exchange filter compliance (LOT_SIZE,
        MIN_NOTIONAL, precision). Reconciliation loop lands in Phase 4."""
        symbol = signal['symbol']
        try:
            market = self.exchange.market(symbol)
            amount = float(self.exchange.amount_to_precision(symbol, quantity))
            price = float(self.exchange.price_to_precision(symbol, limit_price))

            min_amount = (market.get('limits', {}).get('amount', {}) or {}).get('min')
            min_cost = (market.get('limits', {}).get('cost', {}) or {}).get('min')
            if min_amount and amount < min_amount:
                logger.warning(f"Live order below LOT_SIZE for {symbol}: "
                               f"{amount} < {min_amount}")
                return False
            if min_cost and amount * price < min_cost:
                logger.warning(f"Live order below MIN_NOTIONAL for {symbol}: "
                               f"{amount * price:.2f} < {min_cost}")
                return False

            order = self.exchange.create_order(
                symbol, 'limit', side.lower(), amount, price,
                params={'timeInForce': 'GTC'})
            logger.info(f"LIVE order submitted: {side} {amount} {symbol} @ {price} "
                        f"(id {order.get('id')})")
            return True
        except Exception as e:
            logger.error(f"Live order failed for {symbol}: {e}")
            if self.metrics:
                self.metrics.record_error('live_order', str(e))
            return False

    def _submit_live_close(self, pos: Dict, reason: str):
        symbol = pos['symbol']
        try:
            amount = float(self.exchange.amount_to_precision(symbol, pos['quantity']))
            order = self.exchange.create_order(symbol, 'market', 'sell', amount)
            logger.info(f"LIVE close submitted: SELL {amount} {symbol} "
                        f"({reason}, id {order.get('id')})")
        except Exception as e:
            # Put the position back — it is still open on the exchange
            self.positions[symbol] = pos
            logger.error(f"Live close FAILED for {symbol}: {e}")
            self.db.store_alert('risk', 'critical',
                                f'Live close failed: {symbol}', str(e),
                                symbol=symbol)

    # ── Accounting ───────────────────────────────────────────────────────

    def _total_equity(self) -> float:
        positions_value = sum(
            pos['quantity'] * self.latest_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items())
        return self.cash + positions_value

    def _record_equity(self):
        equity = self._total_equity()
        positions_value = equity - self.cash
        start = self._starting_equity()
        drawdown = (equity / start - 1) if start else None
        btc_price = self.latest_prices.get('BTC/USDT')

        # Drawdown alert: fire once when crossing -8%, re-arm on recovery
        if drawdown is not None and self.notifier:
            if drawdown <= -self.DRAWDOWN_ALERT_LEVEL and not self._dd_alert_active:
                self._dd_alert_active = True
                self.notifier.alert(
                    'risk', 'critical',
                    f'Drawdown {drawdown:.1%} exceeds -{self.DRAWDOWN_ALERT_LEVEL:.0%}',
                    f'Equity ${equity:,.2f} vs starting ${start:,.2f}. '
                    f'Kill switch triggers at -{self.limits.max_drawdown:.0%}.',
                    dedupe_key='drawdown_8pct')
            elif drawdown > -self.DRAWDOWN_ALERT_LEVEL * 0.75:
                self._dd_alert_active = False

        self.db.record_equity(
            equity=equity, cash=self.cash, positions_value=positions_value,
            active_positions=len(self.positions),
            drawdown=min(drawdown, 0.0) if drawdown is not None else None,
            mode=self.mode, benchmark_price=btc_price)
        if self.metrics:
            self.metrics.record_equity(equity, len(self.positions))

    def _persist_order(self, order: Dict, status: str,
                       fill_price: float = None, commission: float = None,
                       slippage: float = None):
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO orders (id, symbol, side, order_type, quantity,
                                        price, status, fill_price, commission,
                                        slippage, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status = excluded.status,
                        fill_price = excluded.fill_price,
                        commission = excluded.commission,
                        slippage = excluded.slippage,
                        updated_at = excluded.updated_at
                """, (
                    order['id'], order['symbol'], order['side'],
                    order.get('order_type', 'LIMIT'), order['quantity'],
                    order.get('price'), status, fill_price, commission, slippage,
                    int(order['created_at'].timestamp()),
                    int(_utcnow().timestamp()),
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Order persist failed: {e}")
