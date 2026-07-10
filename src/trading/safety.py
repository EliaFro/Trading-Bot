"""
Hard, non-overridable safety rails for the trading engine.

These constants are CEILINGS: config can tighten them but never loosen them.
The kill switch persists in the database — once triggered, trading stays
halted across restarts until manually cleared with scripts/clear_kill_switch.py.

Live-mode startup checks (activate in live mode):
  * refuse to run if the Binance API key has withdrawal permission
  * refuse to run if .env is readable by anyone but the owner
"""

import logging
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.config import mask_secret

logger = logging.getLogger(__name__)

# ── Hard ceilings (cannot be raised via config) ─────────────────────────────
HARD_MAX_POSITION_SIZE = 0.10        # max 10% of equity per position
HARD_MAX_CONCURRENT_POSITIONS = 5
HARD_MAX_DRAWDOWN = 0.15             # kill switch at -15% from starting equity
HARD_DAILY_LOSS_LIMIT = 0.03         # no new trades after -3% in a day
MIN_ORDER_NOTIONAL_USDT = 10.0       # Binance spot MIN_NOTIONAL floor

KILL_SWITCH_KEY = 'kill_switch_triggered'


class SafetyError(RuntimeError):
    """Raised when a safety precondition fails. The engine must not trade."""


@dataclass(frozen=True)
class RiskLimits:
    """Effective limits: config values clamped to the hard ceilings."""
    max_position_size: float
    max_concurrent_positions: int
    max_drawdown: float
    daily_loss_limit: float
    stop_loss_pct: float
    take_profit_pct: float

    @classmethod
    def from_config(cls, risk_cfg: dict) -> 'RiskLimits':
        return cls(
            max_position_size=min(
                float(risk_cfg.get('max_position_size', HARD_MAX_POSITION_SIZE)),
                HARD_MAX_POSITION_SIZE),
            max_concurrent_positions=min(
                int(risk_cfg.get('max_concurrent_positions',
                                 HARD_MAX_CONCURRENT_POSITIONS)),
                HARD_MAX_CONCURRENT_POSITIONS),
            max_drawdown=min(
                float(risk_cfg.get('max_drawdown', HARD_MAX_DRAWDOWN)),
                HARD_MAX_DRAWDOWN),
            daily_loss_limit=min(
                float(risk_cfg.get('daily_loss_limit', HARD_DAILY_LOSS_LIMIT)),
                HARD_DAILY_LOSS_LIMIT),
            stop_loss_pct=float(risk_cfg.get('stop_loss_pct', 0.02)),
            take_profit_pct=float(risk_cfg.get('take_profit_pct', 0.04)),
        )


# ── Startup checks ──────────────────────────────────────────────────────────

def check_env_file_permissions(env_path: str = '.env',
                               live_mode: bool = False) -> None:
    """Refuse live mode if .env is group/world readable. chmod 600 fixes it."""
    path = Path(env_path)
    if not path.exists():
        return
    mode = path.stat().st_mode
    exposed = bool(mode & (stat.S_IRGRP | stat.S_IWGRP |
                           stat.S_IROTH | stat.S_IWOTH))
    if exposed:
        message = (f"{env_path} is readable by other users "
                   f"(mode {stat.filemode(mode)}). Run: chmod 600 {env_path}")
        if live_mode:
            raise SafetyError(message)
        logger.warning(message)


def check_withdrawal_permissions(exchange, live_mode: bool = False) -> None:
    """Refuse live mode if the API key can withdraw funds.

    Uses Binance's /sapi/v1/account/apiRestrictions endpoint via ccxt.
    In paper mode this only logs; in live mode a withdrawal-enabled key is
    a hard refusal.
    """
    try:
        restrictions = exchange.sapi_get_account_apirestrictions()
    except AttributeError:
        try:
            restrictions = exchange.sapiGetAccountApiRestrictions()
        except Exception as e:
            _handle_restriction_check_failure(e, live_mode)
            return
    except Exception as e:
        _handle_restriction_check_failure(e, live_mode)
        return

    withdrawals = str(restrictions.get('enableWithdrawals', '')).lower() == 'true'
    trading = str(restrictions.get('enableSpotAndMarginTrading', '')).lower() == 'true'

    key_id = mask_secret(getattr(exchange, 'apiKey', None))
    if withdrawals:
        message = (f"API key {key_id} has WITHDRAWALS ENABLED. "
                   f"Create a key with withdrawals disabled before live trading.")
        if live_mode:
            raise SafetyError(message)
        logger.warning(message)
    if live_mode and not trading:
        raise SafetyError(f"API key {key_id} does not have spot trading enabled.")
    if not withdrawals:
        logger.info(f"API key {key_id}: withdrawals disabled ✓")


def _handle_restriction_check_failure(error: Exception, live_mode: bool) -> None:
    message = (f"Could not verify API key restrictions: {error}. "
               f"Live trading requires this check to pass.")
    if live_mode:
        raise SafetyError(message)
    logger.warning(message)


# ── Kill switch (persisted in system_config) ────────────────────────────────

def is_kill_switch_triggered(db) -> Optional[str]:
    """Return the trigger reason if the kill switch is set, else None."""
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            row = conn.execute(
                text("SELECT config_value FROM system_config WHERE config_key = :k"),
                {'k': KILL_SWITCH_KEY}).fetchone()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"Kill switch state unreadable: {e} — failing safe (halted)")
        return f"kill switch state unreadable: {e}"


def trigger_kill_switch(db, reason: str) -> None:
    """Set the persistent kill switch. Trading halts until manually cleared."""
    from sqlalchemy import text
    stamp = f"{datetime.now(timezone.utc).isoformat()} — {reason}"
    with db.engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO system_config (config_key, config_value, config_type,
                                       description, updated_at)
            VALUES (:k, :v, 'string', 'Kill switch trigger record',
                    :ts)
            ON CONFLICT(config_key) DO UPDATE
            SET config_value = :v, updated_at = :ts
        """), {'k': KILL_SWITCH_KEY, 'v': stamp,
               'ts': int(datetime.now(timezone.utc).timestamp())})
        conn.commit()
    logger.critical(f"KILL SWITCH TRIGGERED: {stamp}")


def clear_kill_switch(db) -> bool:
    from sqlalchemy import text
    with db.engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM system_config WHERE config_key = :k"),
            {'k': KILL_SWITCH_KEY})
        conn.commit()
    return result.rowcount > 0
