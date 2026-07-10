"""Self-executing kill rule: the deadline must engage mechanically,
permanently, and survive restarts — without anyone remembering."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.trading import kill_rule as kr


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Sentinel redirected to a temp dir; offline artifacts stubbed to the
    real failing verdicts."""
    monkeypatch.setattr(kr, 'SENTINEL', tmp_path / 'SEARCH_CLOSED')
    art_b = tmp_path / 'partB.json'
    art_c = tmp_path / 'partC.json'
    art_b.write_text(json.dumps({'bonferroni_significant': [],
                                 'reality_check': {'p': 1.0}}))
    art_c.write_text(json.dumps({'bonferroni_significant': []}))
    monkeypatch.setattr(kr, 'OFFLINE_ARTIFACTS', (str(art_b), str(art_c)))
    return tmp_path


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def alert(self, *args, **kwargs):
        self.alerts.append((args, kwargs))
        return True


BEFORE = kr.KILL_DATE - timedelta(days=10)
AFTER = kr.KILL_DATE + timedelta(hours=1)


def test_nothing_happens_before_deadline(isolated, tmp_db):
    assert kr.check_and_engage(tmp_db, now=BEFORE) is None
    assert not kr.is_closed(tmp_db)
    kr.assert_search_allowed(tmp_db)          # does not raise


def test_deadline_engages_lockout(isolated, tmp_db):
    notifier = FakeNotifier()
    record = kr.check_and_engage(tmp_db, notifier, now=AFTER)
    assert record is not None
    assert kr.is_closed(tmp_db)
    assert len(notifier.alerts) == 1
    # evidence captured mechanically
    assert record['evidence']['anything_passed'] is False
    assert 'live' in record['evidence']


def test_lockout_survives_restart_via_sentinel_alone(isolated, tmp_db):
    kr.check_and_engage(tmp_db, now=AFTER)
    # "restart": fresh DB (simulates db reset losing the lock row)
    from scripts.init_db import init_db
    from src.utils.database import DatabaseManager
    fresh = str(isolated / 'fresh.db')
    init_db(fresh)
    fresh_db = DatabaseManager(fresh)
    assert kr.is_closed(fresh_db), "sentinel alone must hold the lock"
    with pytest.raises(kr.SearchClosedError):
        kr.assert_search_allowed(fresh_db)


def test_lockout_survives_sentinel_loss_via_db(isolated, tmp_db):
    kr.check_and_engage(tmp_db, now=AFTER)
    kr.SENTINEL.unlink()                      # simulate file loss
    assert kr.is_closed(tmp_db), "DB record alone must hold the lock"


def test_closure_alert_fires_exactly_once(isolated, tmp_db):
    notifier = FakeNotifier()
    assert kr.check_and_engage(tmp_db, notifier, now=AFTER) is not None
    assert kr.check_and_engage(tmp_db, notifier, now=AFTER) is None
    assert kr.check_and_engage(tmp_db, notifier,
                               now=AFTER + timedelta(days=30)) is None
    assert len(notifier.alerts) == 1


def test_search_entry_points_refuse_when_locked(isolated, tmp_db):
    kr.check_and_engage(tmp_db, now=AFTER)
    with pytest.raises(kr.SearchClosedError, match='PERMANENTLY CLOSED'):
        kr.assert_search_allowed()            # sentinel path, no db needed


def test_no_bypass_parameter_exists():
    """The lockout API must not accept any override argument."""
    import inspect
    for fn in (kr.assert_search_allowed, kr.is_closed):
        params = set(inspect.signature(fn).parameters)
        assert params <= {'lab_db'}, f"{fn.__name__} grew a bypass parameter"


def test_genuine_pass_blocks_closure_and_alerts(isolated, tmp_db, monkeypatch):
    """Safety valve: if something DID clear the bar, the rule must NOT lock
    — it must escalate instead."""
    art = isolated / 'partB.json'
    art.write_text(json.dumps({'bonferroni_significant': ['some_winner'],
                               'reality_check': {'p': 0.001}}))
    notifier = FakeNotifier()
    record = kr.check_and_engage(tmp_db, notifier, now=AFTER)
    assert record is None
    assert not kr.is_closed(tmp_db)
    assert len(notifier.alerts) == 1
    assert 'PASSED' in notifier.alerts[0][0][2]
