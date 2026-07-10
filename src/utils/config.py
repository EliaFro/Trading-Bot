"""
Configuration management for AI Crypto Trading System.

Config wraps the merged YAML + environment configuration and exposes each
top-level section as an attribute (config.trading, config.risk_management,
config.execution, config.strategies, config.features, config.sentiment,
config.patterns, config.models, config.database, config.redis, config.api_keys).

Secrets are only ever read from environment variables (.env), never from YAML.
Use mask_secret() whenever a key/token could reach a log, report, or screen.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Sections main.py and components expect to exist even if absent from YAML
_KNOWN_SECTIONS = (
    'trading', 'execution', 'risk_management', 'strategies', 'features',
    'sentiment', 'patterns', 'models', 'database', 'redis', 'api_keys',
)


def mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask a secret for display: show only the last `visible` characters."""
    if not value:
        return '(not set)'
    value = str(value)
    if len(value) <= visible:
        return '*' * len(value)
    return '*' * (len(value) - visible) + value[-visible:]


class Config:
    """Attribute-access wrapper over the merged configuration dict."""

    def __init__(self, **sections: Any):
        self._data: Dict[str, Any] = dict(sections)
        for name in _KNOWN_SECTIONS:
            self._data.setdefault(name, {})

    def __getattr__(self, name: str) -> Any:
        # Called only when normal attribute lookup fails
        data = object.__getattribute__(self, '_data')
        if name in data:
            return data[name]
        raise AttributeError(f"Config has no section '{name}'")

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """Full config as a plain dict. api_keys are masked — safe to log."""
        out = {}
        for key, value in self._data.items():
            if key == 'api_keys':
                out[key] = _mask_tree(value)
            else:
                out[key] = value
        return out

    @classmethod
    def load(cls, config_path: str = 'config/trading.yaml',
             models_path: str = 'config/models.yaml') -> 'Config':
        """Load YAML config, merge models.yaml, apply env overrides."""
        data: Dict[str, Any] = {}

        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")

        # models.yaml holds model hyperparameters; trading.yaml's `models:`
        # section holds retraining policy. Merge both under `models`.
        mpath = Path(models_path)
        if mpath.exists():
            with open(mpath) as f:
                model_params = yaml.safe_load(f) or {}
            merged = dict(model_params)
            merged.update(data.get('models', {}))
            data['models'] = merged

        data = _apply_env_overrides(data)
        return cls(**data)


def _mask_tree(value: Any) -> Any:
    """Recursively mask every leaf string in a dict of secrets."""
    if isinstance(value, dict):
        return {k: _mask_tree(v) for k, v in value.items()}
    return mask_secret(value)


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fold environment variables (from .env) into the config dict."""
    data['api_keys'] = {
        'binance': {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'testnet_api_key': os.getenv('BINANCE_TESTNET_API_KEY'),
            'testnet_secret': os.getenv('BINANCE_TESTNET_SECRET'),
            'use_testnet': os.getenv('USE_TESTNET', 'false').lower() == 'true',
        },
        'telegram': {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        },
    }
    data['database'] = {
        'path': os.getenv('DB_PATH', './data/trading_system.db'),
        'postgres_url': os.getenv('DATABASE_URL'),
    }
    data['redis'] = {
        'url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
    }

    trading = data.setdefault('trading', {})
    if os.getenv('INITIAL_CAPITAL'):
        trading['initial_capital'] = float(os.getenv('INITIAL_CAPITAL'))
    trading.setdefault('initial_capital', 10000.0)

    risk = data.setdefault('risk_management', {})
    for env_var, key in (
        ('MAX_POSITION_SIZE', 'max_position_size'),
        ('MAX_DRAWDOWN_LIMIT', 'max_drawdown'),
        ('DAILY_LOSS_LIMIT', 'daily_loss_limit'),
    ):
        if os.getenv(env_var):
            try:
                risk[key] = float(os.getenv(env_var))
            except ValueError:
                logger.warning(f"Ignoring non-numeric env var {env_var}")
    if os.getenv('MAX_CONCURRENT_POSITIONS'):
        try:
            risk['max_concurrent_positions'] = int(os.getenv('MAX_CONCURRENT_POSITIONS'))
        except ValueError:
            pass

    return data
