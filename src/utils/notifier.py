"""
Alerting for the trading system: Telegram (primary) and email (optional).

Alert policy (Phase 3):
  * drawdown > 8% from starting equity     -> critical, once per crossing
  * any trading-loop error                 -> warning (rate-limited)
  * system start/restart                   -> info
  * daily P&L summary at UTC midnight      -> info

Configuration comes from .env: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID enable
Telegram; ALERT_EMAIL (+ SMTP_* vars) enables email. With neither set, alerts
still land in the database `alerts` table and the log — nothing is lost.

Secrets are never logged (mask_secret on any identifier that could leak).
"""

import logging
import os
import smtplib
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, Optional

import requests

from src.utils.config import mask_secret

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}


class Notifier:
    """Delivers alerts to Telegram/email, persists them to the DB, and
    rate-limits repeats so an error loop can't flood the channel."""

    def __init__(self, db=None, min_interval_per_key: float = 900.0):
        self.db = db
        self.min_interval = min_interval_per_key
        self._last_sent: Dict[str, float] = {}

        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN') or None
        self.telegram_chat = os.getenv('TELEGRAM_CHAT_ID') or None
        self.email_to = os.getenv('ALERT_EMAIL') or None
        self.smtp_host = os.getenv('SMTP_HOST') or None

        channels = []
        if self.telegram_token and self.telegram_chat:
            channels.append(f"telegram(chat {mask_secret(self.telegram_chat)})")
        if self.email_to and self.smtp_host:
            channels.append(f"email({mask_secret(self.email_to)})")
        logger.info(f"Notifier channels: {channels or ['db+log only']}")

    # ── Public API ───────────────────────────────────────────────────────

    def alert(self, alert_type: str, severity: str, title: str,
              message: str, symbol: Optional[str] = None,
              dedupe_key: Optional[str] = None) -> bool:
        """Send an alert. Returns True if delivered to at least one external
        channel. Repeats with the same dedupe_key inside min_interval are
        persisted but not re-sent."""
        if self.db is not None:
            try:
                self.db.store_alert(alert_type, severity, title, message,
                                    symbol=symbol)
            except Exception as e:
                logger.error(f"Alert DB persist failed: {e}")

        key = dedupe_key or f"{alert_type}:{title}"
        now = time.time()
        if now - self._last_sent.get(key, 0) < self.min_interval:
            logger.debug(f"Alert '{key}' rate-limited")
            return False
        self._last_sent[key] = now

        emoji = SEVERITY_EMOJI.get(severity, '')
        text = (f"{emoji} *{title}*\n{message}\n"
                f"_{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_")

        delivered = False
        if self.telegram_token and self.telegram_chat:
            delivered |= self._send_telegram(text)
        if self.email_to and self.smtp_host:
            delivered |= self._send_email(f"[trading-bot] {title}",
                                          f"{title}\n\n{message}")
        if not delivered:
            logger.log(logging.CRITICAL if severity == 'critical' else logging.WARNING,
                       f"ALERT (no external channel): {title} — {message}")
        return delivered

    def daily_summary(self, summary: Dict) -> bool:
        """Daily P&L summary — bypasses dedupe (one per day by design)."""
        message = (
            f"Equity: ${summary.get('equity', 0):,.2f} "
            f"({summary.get('daily_return', 0):+.2%} today)\n"
            f"Trades: {summary.get('n_trades', 0)} "
            f"(W {summary.get('wins', 0)} / L {summary.get('losses', 0)})\n"
            f"Realized P&L: ${summary.get('realized_pnl', 0):+,.2f}\n"
            f"Fees: ${summary.get('fees', 0):,.2f}\n"
            f"Open positions: {summary.get('open_positions', 0)}\n"
            f"Drawdown from start: {summary.get('drawdown', 0):+.2%}\n"
            f"Mode: {summary.get('mode', 'paper')} · "
            f"Uptime: {summary.get('uptime_hours', 0):.1f}h")
        self._last_sent.pop('daily:summary', None)
        return self.alert('daily_summary', 'info', 'Daily P&L summary',
                          message, dedupe_key='daily:summary')

    # ── Channels ─────────────────────────────────────────────────────────

    def _send_telegram(self, text: str) -> bool:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={'chat_id': self.telegram_chat, 'text': text,
                      'parse_mode': 'Markdown'},
                timeout=10)
            if resp.status_code == 200:
                return True
            logger.error(f"Telegram send failed: HTTP {resp.status_code} "
                         f"{resp.text[:200]}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
        return False

    def _send_email(self, subject: str, body: str) -> bool:
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = os.getenv('SMTP_FROM', 'trading-bot@localhost')
            msg['To'] = self.email_to
            port = int(os.getenv('SMTP_PORT', '587'))
            with smtplib.SMTP(self.smtp_host, port, timeout=15) as server:
                if os.getenv('SMTP_USER'):
                    server.starttls()
                    server.login(os.getenv('SMTP_USER'),
                                 os.getenv('SMTP_PASSWORD', ''))
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
        return False
