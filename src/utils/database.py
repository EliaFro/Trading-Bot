"""
Database management utilities for AI Crypto Trading System
Handles all database operations including SQLite and PostgreSQL support
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# psycopg2 is only required for PostgreSQL mode; SQLite (v1 default) must not
# depend on it being installed.
try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    commission: float
    slippage: float
    strategy: str
    features: Dict
    entry_time: datetime
    exit_time: Optional[datetime]
    status: str

@dataclass
class Pattern:
    """Pattern data structure"""
    id: int
    pattern_type: str
    pattern_config: Dict
    performance: float
    discovery_date: datetime
    status: str

class DatabaseManager:
    """Manages all database operations"""
    
    def __init__(self, config: Union[str, Dict]):
        """
        Initialize database manager
        
        Args:
            config: Database path (SQLite) or configuration dict (PostgreSQL)
        """
        self.config = config
        self.engine = None
        self.pool = None
        self.db_type = 'sqlite'  # Default

        if isinstance(config, str):
            # SQLite path given directly
            self.db_path = config
            self._init_sqlite()
        elif isinstance(config, dict) and config.get('path') and not config.get('postgres_url'):
            # Dict from Config: {'path': ..., 'postgres_url': None} -> SQLite
            self.db_path = config['path']
            self._init_sqlite()
        elif isinstance(config, dict):
            # PostgreSQL configuration (explicit postgres_url or host/user keys)
            if not PSYCOPG2_AVAILABLE:
                raise RuntimeError(
                    "PostgreSQL configuration provided but psycopg2 is not "
                    "installed. Install psycopg2-binary or use SQLite (DB_PATH).")
            self.db_type = 'postgresql'
            self._init_postgresql(config)
        else:
            raise ValueError(f"Unsupported database config: {type(config)}")
    
    def _init_sqlite(self):
        """Initialize SQLite connection"""
        try:
            # Create directory if it doesn't exist
            parent = os.path.dirname(self.db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                f'sqlite:///{self.db_path}',
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            logger.info(f"Initialized SQLite database at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    def _init_postgresql(self, config: Dict):
        """Initialize PostgreSQL connection pool"""
        try:
            # Create connection string
            conn_string = (
                f"postgresql://{config['user']}:{config['password']}"
                f"@{config['host']}:{config['port']}/{config['database']}"
            )
            
            # Create SQLAlchemy engine with connection pool
            self.engine = create_engine(
                conn_string,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Also create psycopg2 pool for direct queries
            self.pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
            
            logger.info(f"Initialized PostgreSQL connection to {config['host']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        else:
            conn = self.pool.getconn()
            try:
                yield conn
            finally:
                self.pool.putconn(conn)
    
    # --- OHLCV Data Operations ---
    
    def store_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> int:
        """Store OHLCV data (idempotent: duplicate bars are ignored)."""
        if data is None or data.empty:
            return 0
        try:
            data = data.copy()

            # Normalize timestamp column to epoch seconds (UTC)
            ts = data['timestamp']
            if pd.api.types.is_datetime64_any_dtype(ts):
                epochs = (ts.astype('int64') // 10**9).tolist()
            else:
                epochs = ts.astype('int64').tolist()
                # Millisecond epochs (from ccxt) -> seconds
                epochs = [int(t // 1000) if t > 10**11 else int(t) for t in epochs]

            rows = [
                (symbol, timeframe, epoch,
                 float(r.open), float(r.high), float(r.low),
                 float(r.close), float(r.volume))
                for epoch, r in zip(epochs, data.itertuples(index=False))
            ]

            if self.db_type == 'sqlite':
                insert_sql = """
                    INSERT OR IGNORE INTO ohlcv
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            else:
                insert_sql = """
                    INSERT INTO ohlcv
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(insert_sql, rows)
                inserted = conn.total_changes if self.db_type == 'sqlite' else cursor.rowcount
                conn.commit()

            logger.debug(f"Stored OHLCV for {symbol} {timeframe}: "
                         f"{len(rows)} rows submitted")
            return len(rows)

        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            return 0

    def get_latest_ohlcv_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Latest stored bar time (epoch seconds) — used to resume backfills."""
        try:
            query = text("""
                SELECT MAX(timestamp) FROM ohlcv
                WHERE symbol = :symbol AND timeframe = :timeframe
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query, {'symbol': symbol,
                                              'timeframe': timeframe}).fetchone()
            return int(result[0]) if result and result[0] is not None else None
        except Exception as e:
            logger.error(f"Error getting latest OHLCV timestamp: {e}")
            return None
    
    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieve OHLCV data"""
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = :symbol AND timeframe = :timeframe
            """
            
            params = {'symbol': symbol, 'timeframe': timeframe}
            
            if start_date:
                if self.db_type == 'sqlite':
                    query += " AND timestamp >= :start_date"
                    params['start_date'] = int(start_date.timestamp())
                else:
                    query += " AND timestamp >= :start_date"
                    params['start_date'] = start_date
            
            if end_date:
                if self.db_type == 'sqlite':
                    query += " AND timestamp <= :end_date"
                    params['end_date'] = int(end_date.timestamp())
                else:
                    query += " AND timestamp <= :end_date"
                    params['end_date'] = end_date
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            df = pd.read_sql_query(query, self.engine, params=params)
            
            # Convert timestamp back to datetime
            if not df.empty:
                if self.db_type == 'sqlite':
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            return pd.DataFrame()
    
    # --- Trade Operations ---
    
    def store_trade(self, trade: Union[Trade, Dict]) -> bool:
        """Store a trade record"""
        try:
            if isinstance(trade, Trade):
                trade_dict = asdict(trade)
            else:
                trade_dict = trade
            
            # Convert datetime to timestamp for SQLite
            if self.db_type == 'sqlite':
                trade_dict['entry_time'] = int(trade_dict['entry_time'].timestamp())
                if trade_dict.get('exit_time'):
                    trade_dict['exit_time'] = int(trade_dict['exit_time'].timestamp())
            
            # Serialize features to JSON
            trade_dict['features'] = json.dumps(trade_dict.get('features', {}))
            
            # Insert trade
            with self.get_connection() as conn:
                if self.db_type == 'sqlite':
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO trades (
                            id, symbol, side, quantity, entry_price, exit_price,
                            stop_loss, take_profit, pnl, pnl_percentage,
                            commission, slippage, strategy, features,
                            entry_time, exit_time, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_dict['id'], trade_dict['symbol'], trade_dict['side'],
                        trade_dict['quantity'], trade_dict['entry_price'],
                        trade_dict.get('exit_price'), trade_dict['stop_loss'],
                        trade_dict['take_profit'], trade_dict.get('pnl'),
                        trade_dict.get('pnl_percentage'), trade_dict.get('commission', 0),
                        trade_dict.get('slippage', 0), trade_dict['strategy'],
                        trade_dict['features'], trade_dict['entry_time'],
                        trade_dict.get('exit_time'), trade_dict['status']
                    ))
                    conn.commit()
                else:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO trades (
                            id, symbol, side, quantity, entry_price, exit_price,
                            stop_loss, take_profit, pnl, pnl_percentage,
                            commission, slippage, strategy, features,
                            entry_time, exit_time, status
                        ) VALUES (
                            %(id)s, %(symbol)s, %(side)s, %(quantity)s, %(entry_price)s,
                            %(exit_price)s, %(stop_loss)s, %(take_profit)s, %(pnl)s,
                            %(pnl_percentage)s, %(commission)s, %(slippage)s,
                            %(strategy)s, %(features)s, %(entry_time)s,
                            %(exit_time)s, %(status)s
                        )
                    """, trade_dict)
                    conn.commit()
            
            logger.info(f"Stored trade {trade_dict['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
            return False
    
    def update_trade(self, trade_id: str, updates: Dict) -> bool:
        """Update a trade record"""
        try:
            # Handle datetime conversions
            if self.db_type == 'sqlite':
                if 'exit_time' in updates and updates['exit_time']:
                    updates['exit_time'] = int(updates['exit_time'].timestamp())
            
            # Build update query
            set_clause = ', '.join([f"{k} = :{k}" for k in updates.keys()])
            query = f"UPDATE trades SET {set_clause} WHERE id = :trade_id"
            updates['trade_id'] = trade_id
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), updates)
                conn.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            return False
    
    def get_active_positions(self) -> pd.DataFrame:
        """Get all active trading positions"""
        try:
            query = """
                SELECT id, symbol, side, quantity, entry_price, stop_loss,
                       take_profit, strategy, entry_time, commission
                FROM trades
                WHERE status = 'OPEN'
                ORDER BY entry_time DESC
            """
            
            df = pd.read_sql_query(query, self.engine)
            
            if not df.empty and self.db_type == 'sqlite':
                df['entry_time'] = pd.to_datetime(df['entry_time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return pd.DataFrame()
    
    def get_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        """Get recent trades"""
        try:
            query = f"""
                SELECT * FROM trades
                ORDER BY entry_time DESC
                LIMIT {limit}
            """
            
            df = pd.read_sql_query(query, self.engine)
            
            if not df.empty:
                # Convert timestamps
                if self.db_type == 'sqlite':
                    df['entry_time'] = pd.to_datetime(df['entry_time'], unit='s')
                    df['exit_time'] = pd.to_datetime(df['exit_time'], unit='s')
                
                # Parse features JSON
                df['features'] = df['features'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return pd.DataFrame()
    
    # --- Performance Metrics ---
    
    def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Calculate performance metrics"""
        try:
            # Get closed trades
            query = "SELECT * FROM trades WHERE status = 'CLOSED'"
            params = {}
            
            if start_date:
                if self.db_type == 'sqlite':
                    query += " AND exit_time >= :start_date"
                    params['start_date'] = int(start_date.timestamp())
                else:
                    query += " AND exit_time >= :start_date"
                    params['start_date'] = start_date
            
            if end_date:
                if self.db_type == 'sqlite':
                    query += " AND exit_time <= :end_date"
                    params['end_date'] = int(end_date.timestamp())
                else:
                    query += " AND exit_time <= :end_date"
                    params['end_date'] = end_date
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            if df.empty:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profit_factor': 0
                }
            
            # Calculate metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Returns
            total_return = df['pnl_percentage'].sum() / 100 if 'pnl_percentage' in df else 0
            
            # Sharpe ratio (simplified)
            if 'pnl_percentage' in df and len(df) > 1:
                returns = df['pnl_percentage'] / 100
                sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            if 'pnl' in df:
                cumulative_pnl = df['pnl'].cumsum()
                running_max = cumulative_pnl.expanding().max()
                drawdown = (cumulative_pnl - running_max) / running_max
                max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0
            
            # Profit factor
            gross_profit = df[df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'avg_win': df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
                'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        return self.get_performance_metrics()
    
    # --- Model Management ---
    
    def save_model_version(self, model_data: Dict) -> bool:
        """Save model version information"""
        try:
            with self.get_connection() as conn:
                if self.db_type == 'sqlite':
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO model_versions (
                            model_name, version, parameters, performance_metrics,
                            created_at, is_active
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        model_data['model_name'],
                        model_data['version'],
                        json.dumps(model_data.get('parameters', {})),
                        json.dumps(model_data.get('performance_metrics', {})),
                        int(datetime.now().timestamp()),
                        model_data.get('is_active', False)
                    ))
                    conn.commit()
                else:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO model_versions (
                            model_name, version, parameters, performance_metrics, is_active
                        ) VALUES (%(model_name)s, %(version)s, %(parameters)s, 
                                %(performance_metrics)s, %(is_active)s)
                    """, {
                        'model_name': model_data['model_name'],
                        'version': model_data['version'],
                        'parameters': json.dumps(model_data.get('parameters', {})),
                        'performance_metrics': json.dumps(model_data.get('performance_metrics', {})),
                        'is_active': model_data.get('is_active', False)
                    })
                    conn.commit()
            
            logger.info(f"Saved model version {model_data['model_name']} v{model_data['version']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            return False
    
    def get_last_retrain_time(self) -> Optional[datetime]:
        """Get the timestamp of the last model retrain"""
        try:
            query = """
                SELECT MAX(created_at) as last_retrain
                FROM model_versions
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
            
            if result and result[0]:
                if self.db_type == 'sqlite':
                    return datetime.fromtimestamp(result[0])
                else:
                    return result[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last retrain time: {e}")
            return None
    
    # --- Pattern Discovery ---
    
    def store_pattern(self, pattern: Union[Pattern, Dict]) -> bool:
        """Store a discovered pattern"""
        try:
            if isinstance(pattern, Pattern):
                pattern_dict = asdict(pattern)
            else:
                pattern_dict = pattern
            
            # Convert datetime for SQLite
            if self.db_type == 'sqlite' and 'discovery_date' in pattern_dict:
                pattern_dict['discovery_date'] = int(pattern_dict['discovery_date'].timestamp())
            
            # Serialize config
            pattern_dict['pattern_config'] = json.dumps(pattern_dict.get('pattern_config', {}))
            
            with self.get_connection() as conn:
                if self.db_type == 'sqlite':
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO discovered_patterns (
                            pattern_type, pattern_config, performance,
                            discovery_date, status
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        pattern_dict['pattern_type'],
                        pattern_dict['pattern_config'],
                        pattern_dict['performance'],
                        pattern_dict['discovery_date'],
                        pattern_dict.get('status', 'active')
                    ))
                    conn.commit()
                else:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO discovered_patterns (
                            pattern_type, pattern_config, performance, status
                        ) VALUES (%(pattern_type)s, %(pattern_config)s, 
                                %(performance)s, %(status)s)
                    """, pattern_dict)
                    conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return False
    
    def get_active_patterns(self) -> List[Dict]:
        """Get all active patterns"""
        try:
            query = """
                SELECT * FROM discovered_patterns
                WHERE status = 'active'
                ORDER BY performance DESC
            """
            
            df = pd.read_sql_query(query, self.engine)
            
            if not df.empty:
                # Parse JSON config
                df['pattern_config'] = df['pattern_config'].apply(
                    lambda x: json.loads(x) if x else {}
                )
                
                # Convert timestamp
                if self.db_type == 'sqlite':
                    df['discovery_date'] = pd.to_datetime(df['discovery_date'], unit='s')
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting active patterns: {e}")
            return []
    
    # --- Sentiment Data ---
    
    def store_sentiment(self, symbol: str, sentiment_data: Dict) -> bool:
        """Store sentiment analysis results"""
        try:
            with self.get_connection() as conn:
                if self.db_type == 'sqlite':
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO sentiment_scores (
                            symbol, source, timestamp, sentiment_score,
                            confidence, volume, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        sentiment_data.get('source', 'aggregate'),
                        int(datetime.now().timestamp()),
                        sentiment_data['sentiment'],
                        sentiment_data['confidence'],
                        sentiment_data.get('volume', 0),
                        json.dumps(sentiment_data.get('metadata', {}))
                    ))
                    conn.commit()
                else:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO sentiment_scores (
                            symbol, source, sentiment_score, confidence, volume, metadata
                        ) VALUES (%(symbol)s, %(source)s, %(sentiment_score)s,
                                %(confidence)s, %(volume)s, %(metadata)s)
                    """, {
                        'symbol': symbol,
                        'source': sentiment_data.get('source', 'aggregate'),
                        'sentiment_score': sentiment_data['sentiment'],
                        'confidence': sentiment_data['confidence'],
                        'volume': sentiment_data.get('volume', 0),
                        'metadata': json.dumps(sentiment_data.get('metadata', {}))
                    })
                    conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing sentiment: {e}")
            return False
    
    def get_sentiment_history(
        self,
        symbol: str,
        hours: int = 24
    ) -> pd.DataFrame:
        """Get sentiment history for a symbol"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = """
                SELECT timestamp, sentiment_score, confidence, volume
                FROM sentiment_scores
                WHERE symbol = :symbol AND timestamp > :cutoff
                ORDER BY timestamp
            """
            
            params = {'symbol': symbol}
            
            if self.db_type == 'sqlite':
                params['cutoff'] = int(cutoff_time.timestamp())
            else:
                params['cutoff'] = cutoff_time
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            if not df.empty and self.db_type == 'sqlite':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment history: {e}")
            return pd.DataFrame()
    
    # --- Signals ---

    def store_signal(self, signal: Dict) -> bool:
        """Store a strategy signal (executed or not) for the dashboard/audit."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholder = '?' if self.db_type == 'sqlite' else '%s'
                cursor.execute(f"""
                    INSERT INTO signals
                    (timestamp, symbol, action, confidence, size,
                     stop_loss, take_profit, strategy, executed, metadata)
                    VALUES ({', '.join([placeholder] * 10)})
                """, (
                    int(signal.get('timestamp', datetime.now().timestamp())),
                    signal.get('symbol'),
                    signal.get('action'),
                    signal.get('confidence'),
                    signal.get('size'),
                    signal.get('stop_loss'),
                    signal.get('take_profit'),
                    (signal.get('metadata') or {}).get('strategy', signal.get('strategy')),
                    1 if signal.get('executed') else 0,
                    json.dumps(signal.get('metadata', {}), default=str),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            return False

    def get_recent_signals(self, limit: int = 20) -> pd.DataFrame:
        """Most recent strategy signals."""
        try:
            query = text("""
                SELECT timestamp, symbol, action, confidence, size, stop_loss,
                       take_profit, strategy, executed
                FROM signals ORDER BY timestamp DESC LIMIT :limit
            """)
            df = pd.read_sql_query(query, self.engine, params={'limit': limit})
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return pd.DataFrame()

    # --- Equity curve ---

    def record_equity(self, equity: float, cash: float = None,
                      positions_value: float = None, active_positions: int = 0,
                      drawdown: float = None, mode: str = 'paper',
                      benchmark_price: float = None) -> bool:
        """Append a point to the equity curve (performance_tracking)."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholder = '?' if self.db_type == 'sqlite' else '%s'
                cursor.execute(f"""
                    INSERT INTO performance_tracking
                    (timestamp, total_equity, cash, positions_value,
                     drawdown, active_positions, mode, benchmark_price)
                    VALUES ({', '.join([placeholder] * 8)})
                """, (
                    int(datetime.now().timestamp()), float(equity),
                    cash, positions_value, drawdown,
                    int(active_positions), mode, benchmark_price,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error recording equity: {e}")
            return False

    def get_equity_curve(self, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         mode: Optional[str] = None) -> pd.DataFrame:
        """Equity curve from performance_tracking as a DataFrame."""
        try:
            query = ("SELECT timestamp, total_equity, cash, positions_value, "
                     "drawdown, active_positions, benchmark_price "
                     "FROM performance_tracking WHERE 1=1")
            params = {}
            if start_date:
                query += " AND timestamp >= :start"
                params['start'] = int(start_date.timestamp())
            if end_date:
                query += " AND timestamp <= :end"
                params['end'] = int(end_date.timestamp())
            if mode:
                query += " AND mode = :mode"
                params['mode'] = mode
            query += " ORDER BY timestamp"

            df = pd.read_sql_query(text(query), self.engine, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Error getting equity curve: {e}")
            return pd.DataFrame()

    # --- Alerts ---

    def store_alert(self, alert_type: str, severity: str, title: str,
                    message: str, symbol: str = None, data: Dict = None) -> bool:
        """Persist an alert (also delivered via notifier channels)."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholder = '?' if self.db_type == 'sqlite' else '%s'
                cursor.execute(f"""
                    INSERT INTO alerts
                    (alert_type, severity, title, message, symbol, data, timestamp)
                    VALUES ({', '.join([placeholder] * 7)})
                """, (
                    alert_type, severity, title, message, symbol,
                    json.dumps(data or {}, default=str),
                    int(datetime.now().timestamp()),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            return False

    # --- Utility Methods ---

    def get_trade_count_since(self, since_date: datetime) -> int:
        """Get number of trades since a given date"""
        try:
            query = "SELECT COUNT(*) FROM trades WHERE entry_time > :since_date"
            
            params = {}
            if self.db_type == 'sqlite':
                params['since_date'] = int(since_date.timestamp())
            else:
                params['since_date'] = since_date
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params).fetchone()
            
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting trade count: {e}")
            return 0
    
    def get_training_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get data formatted for model training"""
        try:
            if not symbols:
                # Get all traded symbols
                query = "SELECT DISTINCT symbol FROM trades"
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    symbols = [row[0] for row in result]
            
            training_data = {}
            
            for symbol in symbols:
                # Get OHLCV data
                df = self.get_ohlcv_data(
                    symbol,
                    '1h',  # Default timeframe
                    start_date,
                    end_date
                )
                
                if not df.empty:
                    # Get trades for labeling
                    trades = self.get_trades_for_symbol(symbol, start_date, end_date)
                    
                    # Merge and create labels
                    df = self._create_training_labels(df, trades)
                    
                    training_data[symbol] = df
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return {}
    
    def get_trades_for_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get trades for a specific symbol"""
        try:
            query = "SELECT * FROM trades WHERE symbol = :symbol"
            params = {'symbol': symbol}
            
            if start_date:
                if self.db_type == 'sqlite':
                    query += " AND entry_time >= :start_date"
                    params['start_date'] = int(start_date.timestamp())
                else:
                    query += " AND entry_time >= :start_date"
                    params['start_date'] = start_date
            
            if end_date:
                if self.db_type == 'sqlite':
                    query += " AND entry_time <= :end_date"
                    params['end_date'] = int(end_date.timestamp())
                else:
                    query += " AND entry_time <= :end_date"
                    params['end_date'] = end_date
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            if not df.empty and self.db_type == 'sqlite':
                df['entry_time'] = pd.to_datetime(df['entry_time'], unit='s')
                df['exit_time'] = pd.to_datetime(df['exit_time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting trades for symbol: {e}")
            return pd.DataFrame()
    
    def _create_training_labels(
        self,
        ohlcv_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create training labels based on trades"""
        # This is a simplified version - in practice, you'd have more sophisticated labeling
        ohlcv_df['label'] = 0  # Default: HOLD
        
        for _, trade in trades_df.iterrows():
            # Find the entry time in OHLCV data
            mask = (ohlcv_df['timestamp'] >= trade['entry_time'] - timedelta(hours=1)) & \
                   (ohlcv_df['timestamp'] <= trade['entry_time'] + timedelta(hours=1))
            
            if trade['side'] == 'BUY':
                ohlcv_df.loc[mask, 'label'] = 1
            elif trade['side'] == 'SELL':
                ohlcv_df.loc[mask, 'label'] = -1
        
        return ohlcv_df
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to save space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            if self.db_type == 'sqlite':
                cutoff_timestamp = int(cutoff_date.timestamp())
                
                # Clean old OHLCV data (keep only daily for old data)
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM ohlcv 
                        WHERE timestamp < ? AND timeframe != '1d'
                    """, (cutoff_timestamp,))
                    
                    # Clean old sentiment data
                    cursor.execute("""
                        DELETE FROM sentiment_scores
                        WHERE timestamp < ?
                    """, (cutoff_timestamp,))
                    
                    conn.commit()
            else:
                with self.engine.connect() as conn:
                    conn.execute(
                        text("DELETE FROM ohlcv WHERE timestamp < :cutoff AND timeframe != '1d'"),
                        {'cutoff': cutoff_date}
                    )
                    conn.execute(
                        text("DELETE FROM sentiment_scores WHERE timestamp < :cutoff"),
                        {'cutoff': cutoff_date}
                    )
                    conn.commit()
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
        
        if self.pool:
            self.pool.closeall()
        
        logger.info("Database connections closed")
