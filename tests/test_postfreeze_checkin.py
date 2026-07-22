"""Regression tests for the 2026-07-22 post-freeze check-in fixes
(QA addendum findings 18 and 19 in docs/QA_REPORT.md).

Finding 18 (MEDIUM): the monthly digest only generated when a companion
check ran ON day 1 — a host asleep for that entire UTC day silently
skipped the month's digest. Fixed with a catch-up window mirroring the
DCA LATE-recovery logic.

Finding 19 (MEDIUM): the daily bot's legacy v1 ensemble retrain is a
structural no-op, but the log claimed "Retraining did not improve
performance" — implying a comparison that never happened.
"""

import asyncio
import os
import re
from datetime import datetime, timezone

from src.playbook.companion import LATE_WINDOW_DAYS, should_generate_digest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dt(day):
    return datetime(2026, 8, day, 12, 0, tzinfo=timezone.utc)


def test_digest_fires_on_the_first(tmp_path):
    assert should_generate_digest(_dt(1), tmp_path) is True
    # even if the file already exists (regeneration on the 1st is the
    # long-standing behavior, unchanged)
    (tmp_path / '2026-08.md').write_text('x')
    assert should_generate_digest(_dt(1), tmp_path) is True


def test_digest_catches_up_when_the_first_was_slept_through(tmp_path):
    # host asleep all of Aug 1; first check happens Aug 3 -> catch up
    assert should_generate_digest(_dt(3), tmp_path) is True
    # but never regenerate a digest that already exists
    (tmp_path / '2026-08.md').write_text('x')
    assert should_generate_digest(_dt(3), tmp_path) is False


def test_digest_catchup_window_closes(tmp_path):
    assert should_generate_digest(_dt(LATE_WINDOW_DAYS), tmp_path) is True
    assert should_generate_digest(_dt(LATE_WINDOW_DAYS + 1), tmp_path) is False


def test_companion_runner_uses_the_catchup_helper():
    src = open(os.path.join(ROOT, 'scripts', 'playbook_companion.py')).read()
    assert 'should_generate_digest' in src, \
        'runner must route the digest through the catch-up helper'
    assert not re.search(r'if\s+now\.day\s*==\s*1\s*:', src), \
        'bare day==1 digest trigger reintroduced (finding 18)'


def test_legacy_noop_retrain_reports_noop():
    """The v1 ensemble's retrain must flag itself as a no-op so the caller
    can log truthfully."""
    from src.models.ensemble import EnsembleModel
    obj = object.__new__(EnsembleModel)          # retrain touches no init state
    results = asyncio.run(EnsembleModel.retrain(obj, {}))
    assert results.get('noop') is True
    assert results['improvement'] == 0.0


def test_main_logs_noop_truthfully_not_as_a_failed_comparison():
    src = open(os.path.join(ROOT, 'src', 'main.py')).read()
    i_noop = src.find("results.get('noop')")
    i_compare = src.find("results['improvement'] > min_improvement")
    assert i_noop != -1, 'main.py lost the noop branch (finding 19)'
    assert i_compare != -1
    assert i_noop < i_compare, \
        'noop must be checked before the improvement comparison'
    # the truthful message exists; the misleading one is not in the noop branch
    assert 'nothing to retrain' in src
    noop_branch = src[i_noop:i_compare]
    assert 'did not improve' not in noop_branch
    # the retrain-timer reset (save with is_active False) survives in the
    # noop branch — losing it would make the bot retrain every cycle
    assert "'is_active': False" in noop_branch
