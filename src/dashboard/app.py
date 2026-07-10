#!/usr/bin/env python3
"""
Streamlit Dashboard for AI Crypto Trading System.

REAL DATA ONLY: every number on this dashboard comes from the trading
database or live exchange tickers. When data is missing the dashboard says
so — it never fabricates demo values.

Run:  streamlit run src/dashboard/app.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.database import DatabaseManager

st.set_page_config(
    page_title="AI Crypto Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.getenv('DB_PATH', './data/trading_system.db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h']


# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    if not os.path.exists(DB_PATH):
        return None
    return DatabaseManager(DB_PATH)


@st.cache_resource
def get_redis():
    try:
        import redis
        client = redis.from_url(REDIS_URL, decode_responses=True,
                                socket_connect_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


@st.cache_data(ttl=20)
def live_prices(symbols: tuple) -> dict:
    """Live last prices from Binance public API (20s cache)."""
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = exchange.fetch_tickers(list(symbols))
        return {s: t.get('last') for s, t in tickers.items() if t.get('last')}
    except Exception:
        return {}


@st.cache_data(ttl=30)
def load_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    db = get_db()
    if db is None:
        return pd.DataFrame()
    return db.get_ohlcv_data(symbol, timeframe, limit=limit)


class TradingDashboard:
    """Main dashboard application."""

    def __init__(self):
        self.db = get_db()
        self.redis_client = get_redis()

    # ── Layout ───────────────────────────────────────────────────────────

    def run(self):
        st.title("🤖 AI Cryptocurrency Trading System")

        if self.db is None:
            st.error(f"Database not found at `{DB_PATH}`. "
                     "Run `python scripts/init_db.py` and start the bot first.")
            return

        self._render_sidebar()

        tabs = st.tabs([
            "📈 Live Trading", "🧪 ML Lab", "📊 Performance",
            "📋 Trade History", "💭 Sentiment", "🎯 Patterns",
        ])
        with tabs[0]:
            self._render_live_trading()
        with tabs[1]:
            self._render_ml_lab()
        with tabs[2]:
            self._render_performance()
        with tabs[3]:
            self._render_trade_history()
        with tabs[4]:
            self._render_sentiment()
        with tabs[5]:
            self._render_patterns()

        if st.session_state.get('auto_refresh', True):
            import time
            time.sleep(st.session_state.get('refresh_interval', 15))
            st.rerun()

    def _render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ System")

            # Engine reachability (health endpoint)
            engine = self._engine_status()
            if engine:
                mode = engine.get('mode', '?')
                halted = engine.get('halted')
                if halted:
                    st.error(f"🛑 HALTED ({mode}): {halted}")
                else:
                    st.success(f"🟢 Engine running — {mode} mode")
                st.caption(f"equity ${engine.get('equity', 0):,.2f} · "
                           f"{engine.get('open_positions', 0)} open · "
                           f"cycle {engine.get('cycle_count', 0)}")
            else:
                st.warning("🔴 Engine not reachable on :8080")

            st.divider()
            st.subheader("Manual actions")
            if st.button("🛑 Close all positions"):
                if self._send_command('close_all_positions'):
                    st.warning("Close-all command sent")
                else:
                    st.error("Redis not connected — stop the bot with Ctrl+C "
                             "or `make stop` instead")
            if st.button("🔄 Force model retrain"):
                if self._send_command('retrain_models'):
                    st.info("Retrain command sent")
                else:
                    st.error("Redis not connected")

            st.divider()
            st.subheader("Display")
            st.session_state.auto_refresh = st.checkbox("Auto refresh", value=True)
            st.session_state.refresh_interval = st.slider(
                "Refresh interval (s)", 5, 60, 15)

    # ── Tabs ─────────────────────────────────────────────────────────────

    def _render_live_trading(self):
        prices = live_prices(tuple(SYMBOLS))
        metrics = self.db.get_performance_metrics() or {}
        equity_df = self.db.get_equity_curve(
            start_date=datetime.now() - timedelta(days=1))

        col1, col2, col3, col4 = st.columns(4)
        equity_now = equity_df['total_equity'].iloc[-1] if not equity_df.empty else None
        day_change = None
        if not equity_df.empty and len(equity_df) > 1:
            day_change = equity_now / equity_df['total_equity'].iloc[0] - 1

        col1.metric("Equity",
                    f"${equity_now:,.2f}" if equity_now else "—",
                    f"{day_change:+.2%} today" if day_change is not None else None)
        col2.metric("Closed trades", f"{metrics.get('total_trades', 0):,}")
        col3.metric("Win rate", f"{metrics.get('win_rate', 0):.1%}"
                    if metrics.get('total_trades') else "—")
        col4.metric("Profit factor", f"{metrics.get('profit_factor', 0):.2f}"
                    if metrics.get('total_trades') else "—")

        st.divider()

        # Active positions with REAL current P&L from live prices
        st.subheader("🔥 Active positions")
        positions = self.db.get_active_positions()
        if positions.empty:
            st.info("No open positions")
        else:
            def current_pnl(row):
                price = prices.get(row['symbol'])
                if price is None:
                    return np.nan
                return (price - row['entry_price']) * row['quantity']

            positions['current_price'] = positions['symbol'].map(prices)
            positions['unrealized_pnl'] = positions.apply(current_pnl, axis=1)
            positions['unrealized_pct'] = (
                positions['current_price'] / positions['entry_price'] - 1)

            display = positions[['symbol', 'side', 'quantity', 'entry_price',
                                 'current_price', 'unrealized_pnl',
                                 'unrealized_pct', 'stop_loss', 'take_profit',
                                 'strategy', 'entry_time']]
            st.dataframe(
                display.style.map(
                    lambda v: f"color: {'#00c853' if v > 0 else '#ff5252'}"
                    if isinstance(v, (int, float)) and not pd.isna(v) else '',
                    subset=['unrealized_pnl', 'unrealized_pct']),
                use_container_width=True, hide_index=True)

        # Live price chart from stored candles
        st.subheader("📊 Price chart")
        chart_col, ctl_col = st.columns([4, 1])
        with ctl_col:
            symbol = st.selectbox("Symbol", SYMBOLS, key="chart_symbol")
            timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=2,
                                     key="chart_timeframe")
            indicators = st.multiselect("Indicators", ["SMA 20", "SMA 50", "RSI"],
                                        default=["SMA 20", "RSI"])
        with chart_col:
            df = load_ohlcv(symbol, timeframe, 400)
            if df.empty:
                st.warning(f"No stored candles for {symbol} {timeframe} — "
                           "run `python scripts/backfill.py`")
            else:
                st.plotly_chart(self._price_figure(df, symbol, indicators),
                                use_container_width=True)

        # Recent signals — from the signals table, not Redis guesswork
        st.subheader("🎯 Recent signals")
        signals = self.db.get_recent_signals(15)
        if signals.empty:
            st.info("No signals recorded yet")
        else:
            st.dataframe(
                signals.style.apply(
                    lambda row: ['background-color: rgba(0,200,83,0.15)'] * len(row)
                    if row['action'] == 'BUY'
                    else ['background-color: rgba(255,82,82,0.15)'] * len(row)
                    if row['action'] == 'SELL' else [''] * len(row), axis=1),
                use_container_width=True, hide_index=True)

    def _render_performance(self):
        st.header("📊 Performance")

        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", datetime.now() - timedelta(days=30))
        end_date = col2.date_input("End", datetime.now())
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        equity = self.db.get_equity_curve(start_dt, end_dt)
        if equity.empty:
            st.info("No equity history yet — the curve builds while the bot runs.")
            return

        # Real equity curve vs holding BTC (from stored benchmark prices)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity['timestamp'], y=equity['total_equity'],
            name='Portfolio', line=dict(color='#00c853', width=2)))
        bench = equity.dropna(subset=['benchmark_price'])
        if not bench.empty:
            btc_norm = (bench['benchmark_price'] / bench['benchmark_price'].iloc[0]
                        * equity['total_equity'].iloc[0])
            fig.add_trace(go.Scatter(
                x=bench['timestamp'], y=btc_norm, name='Hold BTC',
                line=dict(color='#ff9800', width=1, dash='dash')))
        fig.update_layout(template='plotly_dark', height=400,
                          title='Equity vs buy-and-hold BTC',
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Real metrics from trades + equity curve
        from src.utils.metrics import MetricsCalculator
        calc = MetricsCalculator()
        trades = self._closed_trades(start_dt, end_dt)
        equity_series = pd.Series(equity['total_equity'].values,
                                  index=pd.to_datetime(equity['timestamp']))
        returns = equity_series.pct_change().dropna()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Returns")
            total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1
            st.metric("Period return", f"{total_ret:.2%}")
            dd = calc.calculate_drawdown_statistics(equity_series)
            st.metric("Max drawdown", f"{dd['max_drawdown']:.2%}")
            st.metric("Current drawdown", f"{dd['current_drawdown']:.2%}")
        with col2:
            st.subheader("Risk")
            st.metric("Sharpe (period)", f"{calc.calculate_sharpe_ratio(returns):.2f}")
            st.metric("Sortino", f"{calc.calculate_sortino_ratio(returns):.2f}")
            st.metric("VaR 95%", f"{calc.calculate_var(returns):.3%}")
        with col3:
            st.subheader("Trades")
            stats = calc.calculate_trade_statistics(trades)
            st.metric("Closed trades", stats['total_trades'])
            st.metric("Win rate", f"{stats['win_rate']:.1%}")
            st.metric("Profit factor",
                      f"{stats['profit_factor']:.2f}"
                      if np.isfinite(stats['profit_factor']) else "∞")
            if 'commission' in trades.columns and not trades.empty:
                st.metric("Fees paid", f"${trades['commission'].sum():,.2f}")

        # Per-strategy breakdown from real trades
        st.subheader("Strategy breakdown")
        strategy_stats = calc.calculate_strategy_metrics(trades)
        if strategy_stats.empty:
            st.info("No closed trades in this period yet")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.pie(strategy_stats, values='total_trades',
                                       names='strategy',
                                       title='Trades by strategy'),
                                use_container_width=True)
            with col2:
                st.dataframe(strategy_stats, use_container_width=True,
                             hide_index=True)

    def _render_trade_history(self):
        st.header("📋 Trade history")

        col1, col2, col3 = st.columns(3)
        symbol = col1.selectbox("Symbol", ["All"] + SYMBOLS)
        status = col2.selectbox("Status", ["All", "OPEN", "CLOSED"])
        days = col3.slider("Days back", 1, 90, 30)

        trades = self._query_trades(
            symbol=None if symbol == "All" else symbol,
            status=None if status == "All" else status,
            since=datetime.now() - timedelta(days=days))

        if trades.empty:
            st.info("No trades match these filters")
            return

        closed = trades[trades['status'] == 'CLOSED']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trades", len(trades))
        if not closed.empty:
            wins = (closed['pnl'] > 0).sum()
            col2.metric("Wins", f"{wins} ({wins / len(closed):.0%})")
            col3.metric("Total P&L", f"${closed['pnl'].sum():,.2f}")
            col4.metric("Fees", f"${closed['commission'].sum():,.2f}")

        show_cols = ['entry_time', 'exit_time', 'symbol', 'side', 'quantity',
                     'entry_price', 'exit_price', 'pnl', 'pnl_percentage',
                     'commission', 'status', 'strategy', 'exit_reason']
        show_cols = [c for c in show_cols if c in trades.columns]
        st.dataframe(trades[show_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download CSV", trades.to_csv(index=False),
            file_name=f"trades_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv")

    def _render_sentiment(self):
        st.header("💭 Sentiment (news-based)")
        df = self._query(
            """SELECT symbol, timestamp, sentiment_score, confidence, volume
               FROM sentiment_scores
               WHERE timestamp >= :since ORDER BY timestamp""",
            {'since': int((datetime.now() - timedelta(days=7)).timestamp())})
        if df is None or df.empty:
            st.info("No sentiment data yet — it accrues every 5 minutes "
                    "while the bot runs.")
            return
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        latest = df.sort_values('timestamp').groupby('symbol').tail(1)
        cols = st.columns(max(len(latest), 1))
        for col, (_, row) in zip(cols, latest.iterrows()):
            col.metric(row['symbol'],
                       f"{row['sentiment_score']:+.2f}",
                       f"conf {row['confidence']:.0%} · {int(row['volume'])} items")

        fig = px.line(df, x='timestamp', y='sentiment_score', color='symbol',
                      title='7-day sentiment')
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)

    def _render_patterns(self):
        st.header("🎯 Discovered patterns")
        df = self._query("""SELECT pattern_type, symbol, timeframe, confidence,
                                   performance, status, discovery_date
                            FROM discovered_patterns
                            ORDER BY discovery_date DESC LIMIT 100""")
        if df is None or df.empty:
            st.info("No patterns discovered yet — discovery runs hourly.")
            return
        df['discovery_date'] = pd.to_datetime(df['discovery_date'], unit='s')

        active = df[df['status'] == 'active']
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates (100 latest)", len(df))
        col2.metric("Active", len(active))
        col3.metric("Avg performance (active)",
                    f"{active['performance'].mean():.2%}" if not active.empty else "—")

        st.dataframe(df, use_container_width=True, hide_index=True)

        counts = df['pattern_type'].value_counts().reset_index()
        counts.columns = ['pattern_type', 'count']
        st.plotly_chart(px.bar(counts, x='pattern_type', y='count',
                               title='Pattern types'),
                        use_container_width=True)

    def _render_ml_lab(self):
        st.header("🧪 ML Lab — watching the learning, honestly")

        # ── The banner (never celebrates; states what the data says) ─────
        st.warning(
            "**ML has not demonstrated an edge over simple momentum.** "
            "Stage 1 walk-forward (36 months, after fees): ML −2.8% vs "
            "TSMOM-60d +0.4% on the identical calendar — see "
            "docs/ML_RESULTS.md. Status: **learning** — this page watches "
            "whether that verdict ever changes, under the same rules that "
            "produced it.")

        retrains = self._query(
            "SELECT * FROM ml_retrain_log ORDER BY timestamp")
        if retrains is None or retrains.empty:
            st.info("No retrains logged yet — the first champion trains when "
                    "the bot starts.")
            return
        retrains['time'] = pd.to_datetime(retrains['timestamp'], unit='s')
        latest = retrains.iloc[-1]

        # ── Overfitting gauge — front and center ─────────────────────────
        st.subheader("Overfitting gauge (latest champion candidate)")
        gap = latest['new_is_bal_acc'] - latest['new_val_bal_acc']
        if gap > 0.15:
            verdict, color = "🚨 this model has memorized noise", "red"
        elif gap > 0.05:
            verdict, color = "⚠️ watch — meaningful memorization", "orange"
        else:
            verdict, color = "✅ healthy — small IS/validation gap", "green"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("In-sample balanced acc", f"{latest['new_is_bal_acc']:.1%}",
                    help="How well it fits data it trained on. High numbers "
                         "here are NOT good news.")
        col2.metric("Validation balanced acc", f"{latest['new_val_bal_acc']:.1%}",
                    help="Held-out recent data. 3-class chance = 33.3%.")
        col3.metric("IS − OOS gap", f"{gap:+.1%}")
        col4.markdown(f"### :{color}[{verdict}]")

        # ── Learning curve across retrains ────────────────────────────────
        st.subheader("Learning curve (every retrain, live)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=retrains['time'], y=retrains['new_is_bal_acc'],
                                 name='in-sample', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=retrains['time'], y=retrains['new_val_bal_acc'],
                                 name='validation (honest)',
                                 line=dict(color='#00c853', width=2)))
        fig.add_hline(y=1 / 3, line_dash='dot', line_color='white',
                      annotation_text='3-class chance (33.3%)')
        fig.update_layout(template='plotly_dark', height=320,
                          yaxis_title='balanced accuracy')
        st.plotly_chart(fig, use_container_width=True)

        # ── Feature importance with EVIDENCE TIERS ────────────────────────
        st.subheader("What the model leans on — and how much evidence backs it")
        st.caption(
            "**Stable is not the same as meaningful.** A feature can rank "
            "high in every retrain and still rest on a handful of "
            "independent observations (e.g. `month`: 3 years of data = "
            "~3 samples per month). Importance says what the model uses; "
            "the evidence column says whether the pattern deserves belief.")
        try:
            import json as _json
            from src.ml.dataset import evidence_count, evidence_tier
            imp = _json.loads(latest['feature_importance'])
            span_days = 1090   # 36 months of daily data
            rows = []
            for feat, weight in list(imp.items())[:15]:
                n = evidence_count(feat, span_days)
                tier = evidence_tier(feat, span_days)
                icon = ('🟢' if tier == 'well-supported'
                        else '🟠' if tier == 'moderate' else '🔴')
                rows.append({'feature': feat, 'importance': round(weight, 4),
                             'independent obs (~)': n,
                             'evidence': f"{icon} {tier}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
            thin = [r['feature'] for r in rows if '🔴' in r['evidence']]
            if thin:
                st.error(f"🔴 Thin-evidence features the model is using: "
                         f"**{', '.join(thin)}** — treat their contribution "
                         f"as noise until years more data exist.")
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")

        # ── Equity vs benchmarks ─────────────────────────────────────────
        st.subheader("Paper equity: ML decisions vs doing nothing clever")
        equity = self.db.get_equity_curve()
        if not equity.empty and len(equity) > 2:
            fig = go.Figure()
            eq0 = equity['total_equity'].iloc[0]
            fig.add_trace(go.Scatter(
                x=equity['timestamp'], y=equity['total_equity'] / eq0,
                name='ML paper account', line=dict(color='#00c853', width=2)))
            bench = equity.dropna(subset=['benchmark_price'])
            if not bench.empty:
                fig.add_trace(go.Scatter(
                    x=bench['timestamp'],
                    y=bench['benchmark_price'] / bench['benchmark_price'].iloc[0],
                    name='Hold BTC', line=dict(color='orange', dash='dash', width=1)))
            fig.add_hline(y=1.0, line_color='gray', line_width=1,
                          annotation_text='cash')
            fig.update_layout(template='plotly_dark', height=350,
                              yaxis_title='equity multiple')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Equity history builds while the bot runs.")

        # ── Today's predictions ───────────────────────────────────────────
        preds = self._query(
            "SELECT timestamp, symbol, pred, p_up, p_down, model_version, "
            "executed FROM ml_predictions ORDER BY timestamp DESC LIMIT 9")
        if preds is not None and not preds.empty:
            st.subheader("Latest predictions")
            preds['time'] = pd.to_datetime(preds['timestamp'], unit='s')
            st.dataframe(preds[['time', 'symbol', 'pred', 'p_up', 'p_down',
                                'model_version', 'executed']],
                         use_container_width=True, hide_index=True)

        # ── Retrain log: watch the guard work ─────────────────────────────
        st.subheader("Retrain log — the keep-old-unless-better guard, audited")
        log = retrains.sort_values('timestamp', ascending=False)[
            ['time', 'decision', 'old_val_f1', 'new_val_f1', 'n_train',
             'reason']]
        st.dataframe(
            log.style.apply(
                lambda row: ['background-color: rgba(0,200,83,0.12)'] * len(row)
                if row['decision'] == 'REPLACED'
                else ['background-color: rgba(158,158,158,0.10)'] * len(row)
                if row['decision'] == 'KEPT_OLD' else [''] * len(row), axis=1),
            use_container_width=True, hide_index=True)

    # ── Data helpers (parameterized SQL only) ────────────────────────────

    def _query(self, sql: str, params: dict = None):
        from sqlalchemy import text
        try:
            return pd.read_sql_query(text(sql), self.db.engine,
                                     params=params or {})
        except Exception as e:
            st.warning(f"Query failed: {e}")
            return None

    def _query_trades(self, symbol=None, status=None, since=None) -> pd.DataFrame:
        sql = "SELECT * FROM trades WHERE 1=1"
        params = {}
        if symbol:
            sql += " AND symbol = :symbol"
            params['symbol'] = symbol
        if status:
            sql += " AND status = :status"
            params['status'] = status
        if since:
            sql += " AND entry_time >= :since"
            params['since'] = int(since.timestamp())
        sql += " ORDER BY entry_time DESC LIMIT 1000"

        df = self._query(sql, params)
        if df is None or df.empty:
            return pd.DataFrame()
        for col in ('entry_time', 'exit_time'):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit='s')
        return df

    def _closed_trades(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = self._query_trades(status='CLOSED', since=start)
        if df.empty:
            return df
        return df[df['exit_time'] <= end]

    def _engine_status(self):
        try:
            import requests
            r = requests.get('http://localhost:8080/status', timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def _send_command(self, command: str) -> bool:
        if not self.redis_client:
            return False
        try:
            self.redis_client.publish('system:commands', json.dumps({
                'command': command,
                'timestamp': datetime.now().isoformat(),
            }))
            return True
        except Exception:
            return False

    # ── Charts ───────────────────────────────────────────────────────────

    def _price_figure(self, df: pd.DataFrame, symbol: str,
                      indicators: list) -> go.Figure:
        rows = 3 if "RSI" in indicators else 2
        heights = [0.6, 0.2, 0.2][:rows]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=heights)

        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=symbol), row=1, col=1)

        if "SMA 20" in indicators:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['close'].rolling(20).mean(),
                name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        if "SMA 50" in indicators:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['close'].rolling(50).mean(),
                name='SMA 50', line=dict(color='cyan', width=1)), row=1, col=1)

        colors = np.where(df['close'] >= df['open'], '#00c853', '#ff5252')
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                             marker_color=colors, name='Volume'), row=2, col=1)

        if "RSI" in indicators:
            from src.utils.indicators import RSI
            rsi = RSI(df['close'], 14)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=rsi, name='RSI',
                                     line=dict(color='purple', width=1)),
                          row=3, col=1)
            fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
            fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)

        fig.update_layout(template='plotly_dark', height=650, showlegend=False,
                          xaxis_rangeslider_visible=False)
        return fig


def main():
    TradingDashboard().run()


if __name__ == "__main__":
    main()
