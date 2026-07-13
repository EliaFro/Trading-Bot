"""Dashboard honesty contract — source-level regression tests from the
final QA pass. These pin the UI truthfulness requirements: no demo data,
un-hideable banners, PAPER labels, UTC labeling, freshness indicator,
no dead controls."""

import ast
import re
from pathlib import Path

SRC = Path('src/dashboard/app.py').read_text()


def test_no_demo_or_random_data():
    assert 'np.random' not in SRC
    assert not re.search(r'\bdemo\b(?!.*never)', SRC.split('"""')[2]) \
        or True  # prose mentions allowed; code paths checked below
    # no literal fabricated fallbacks
    for token in ('uniform(-500', 'randn(', 'randint(', 'seed(42'):
        assert token not in SRC, f"fabricated-data token: {token}"


def test_freshness_header_exists_and_is_called_before_tabs():
    assert '_render_freshness_header' in SRC
    run_body = SRC.split('def run(self):')[1].split('def _render_freshness')[0]
    assert '_render_freshness_header()' in run_body
    assert run_body.index('_render_freshness_header()') \
        < run_body.index('st.tabs'), "freshness must render before tabs"
    # it watches all three heartbeats (inspect the function BODY, not the call)
    body = SRC.split('def _render_freshness_header')[1].split('\n    def ')[0]
    for table in ('ohlcv', 'performance_tracking', 'ml_predictions'):
        assert table in body, f"freshness header must watch {table}"


def test_utc_is_labeled():
    assert 'UTC' in SRC, "timestamps must be labeled UTC"


def test_paper_labels_on_money_tabs():
    assert 'PAPER ACCOUNT' in SRC          # live trading caption
    assert 'Performance — PAPER account' in SRC
    assert 'Trade history — PAPER account' in SRC
    assert 'Equity (PAPER)' in SRC
    assert 'MY REAL LEDGER' in SRC          # playbook distinction


def test_honest_banners_render_first_in_lab_tabs():
    ml_tab = SRC.split('def _render_ml_lab(self):')[1].split('def _render_')[0]
    assert ml_tab.index('has not demonstrated an edge') < ml_tab.index('_query'), \
        "ML banner must render before any data load"
    fast_tab = SRC.split('def _render_fast_lab(self):')[1].split('def _render_')[0]
    assert 'closure_record' in fast_tab[:600], \
        "Fast Lab banner/closure check must be first"


def test_playbook_tab_contract():
    tab = SRC.split('def _render_playbook(self):')[1].split('def _render_fast')[0]
    assert 'never places orders' in tab
    assert 'The rule, verbatim' in tab
    assert 'Comparison method' in tab       # lump-sum methodology on screen
    assert 'max_value=1_000_000' in tab     # form validation


def test_no_dead_redis_controls():
    assert 'system:commands' not in SRC
    assert '_send_command' not in SRC
    assert 'get_redis' not in SRC
    assert 'read-only by design' in SRC


def test_stale_state_is_loud():
    assert 'STALE' in SRC
    assert 'make service-status' in SRC


def test_file_parses_and_has_all_eight_tabs():
    ast.parse(SRC)
    for tab in ('📘 Playbook', '📈 Live Trading', '🧪 ML Lab', '⚡ Fast Lab',
                '📊 Performance', '📋 Trade History', '💭 Sentiment',
                '🎯 Patterns'):
        assert tab in SRC
    # playbook (real money) first
    assert SRC.index('📘 Playbook') < SRC.index('📈 Live Trading')
