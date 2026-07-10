"""Safety rails, kill switch persistence, config loading, secret masking."""

import os
import stat

import pytest

from src.trading import safety
from src.trading.safety import RiskLimits, SafetyError
from src.utils.config import Config, mask_secret


# ── RiskLimits: config can tighten, never loosen ────────────────────────────

def test_risk_limits_clamp_to_hard_ceilings():
    limits = RiskLimits.from_config({
        'max_position_size': 0.5,          # tries to loosen -> clamped to 0.10
        'max_concurrent_positions': 20,    # -> clamped to 5
        'max_drawdown': 0.9,               # -> clamped to 0.15
        'daily_loss_limit': 0.5,           # -> clamped to 0.03
    })
    assert limits.max_position_size == pytest.approx(0.10)
    assert limits.max_concurrent_positions == 5
    assert limits.max_drawdown == pytest.approx(0.15)
    assert limits.daily_loss_limit == pytest.approx(0.03)


def test_risk_limits_allow_tightening():
    limits = RiskLimits.from_config({'max_position_size': 0.02,
                                     'max_concurrent_positions': 2})
    assert limits.max_position_size == pytest.approx(0.02)
    assert limits.max_concurrent_positions == 2


# ── .env permission check ───────────────────────────────────────────────────

def test_env_permissions_block_live_mode(tmp_path):
    env = tmp_path / '.env'
    env.write_text('SECRET=x\n')
    env.chmod(0o644)                      # world-readable
    with pytest.raises(SafetyError):
        safety.check_env_file_permissions(str(env), live_mode=True)
    # Paper mode: warning only
    safety.check_env_file_permissions(str(env), live_mode=False)

    env.chmod(0o600)
    safety.check_env_file_permissions(str(env), live_mode=True)  # passes


# ── Withdrawal-permission check ─────────────────────────────────────────────

class FakeExchange:
    apiKey = 'abcdefghijklmnop'

    def __init__(self, withdrawals, trading=True, fail=False):
        self._resp = {'enableWithdrawals': str(withdrawals).lower(),
                      'enableSpotAndMarginTrading': str(trading).lower()}
        self._fail = fail

    def sapi_get_account_apirestrictions(self):
        if self._fail:
            raise RuntimeError('endpoint unavailable')
        return self._resp


def test_withdrawal_enabled_blocks_live():
    with pytest.raises(SafetyError, match='WITHDRAWALS ENABLED'):
        safety.check_withdrawal_permissions(FakeExchange(True), live_mode=True)


def test_withdrawal_disabled_passes_live():
    safety.check_withdrawal_permissions(FakeExchange(False), live_mode=True)


def test_unverifiable_key_blocks_live():
    with pytest.raises(SafetyError, match='Could not verify'):
        safety.check_withdrawal_permissions(FakeExchange(False, fail=True),
                                            live_mode=True)


def test_trading_disabled_blocks_live():
    with pytest.raises(SafetyError, match='spot trading'):
        safety.check_withdrawal_permissions(
            FakeExchange(False, trading=False), live_mode=True)


# ── Kill switch persistence ─────────────────────────────────────────────────

def test_kill_switch_persists_and_clears(tmp_db):
    assert safety.is_kill_switch_triggered(tmp_db) is None
    safety.trigger_kill_switch(tmp_db, 'equity below -15%')
    reason = safety.is_kill_switch_triggered(tmp_db)
    assert reason and 'equity below -15%' in reason
    # Survives a "restart" (fresh check on same db)
    assert safety.is_kill_switch_triggered(tmp_db)
    assert safety.clear_kill_switch(tmp_db)
    assert safety.is_kill_switch_triggered(tmp_db) is None


# ── Config & secret masking ─────────────────────────────────────────────────

def test_mask_secret():
    assert mask_secret('supersecretkey1234') == '**************1234'
    assert mask_secret('ab') == '**'
    assert mask_secret(None) == '(not set)'
    assert mask_secret('') == '(not set)'


def test_config_load_masks_api_keys_in_dict(monkeypatch):
    monkeypatch.setenv('BINANCE_API_KEY', 'verysecretapikey9999')
    config = Config.load('config/trading.yaml')
    dumped = str(config.to_dict())
    assert 'verysecretapikey9999' not in dumped
    assert '9999' in dumped               # last 4 visible for identification
    # But the real value is still available to code that needs it
    assert config.api_keys['binance']['api_key'] == 'verysecretapikey9999'


def test_config_sections_exist():
    config = Config.load('config/trading.yaml')
    assert 'BTC/USDT' in config.trading['symbols']
    assert config.execution['commission_rate'] == pytest.approx(0.001)
    assert config.execution['slippage_rate'] == pytest.approx(0.0005)
    assert config.risk_management['allow_shorting'] is False
    assert config.models['retrain_interval_hours'] == 24
    # models.yaml hyperparameters merged in
    assert config.models['ensemble']['weights']['transformer'] == pytest.approx(0.4)
