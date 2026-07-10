# Phase 0 — Architecture Report & Gap Analysis

**Project:** AI Crypto Trading System (`Trading_Bot_2`)
**Date:** 2026-07-09
**Scope:** Every project file read (excluding 10 stray/broken virtualenv folders). This report is the reference document for all later phases.

---

## 1. Executive summary (read this first)

The codebase is roughly **40% real, 60% scaffold**. The good news: the backtesting engine, the metrics library, the database layer, and the pattern-ML pipeline are genuinely substantial, well-structured code worth keeping. The bad news: **the system has never run end-to-end and cannot run today** — the orchestrator (`src/main.py`) imports seven modules that do not exist anywhere in the project, the two database init scripts disagree with each other *and* with the database layer, and the dashboard papers over every failure with random demo data.

There is **no trading engine in this project.** The file `engine.py` at the repo root is an unrelated *stock options* dashboard mock (SPY/QQQ iron condors, mock data) — it is not the crypto `TradingEngine` that `main.py` imports from `src/trading/engine.py`, which simply doesn't exist. This is the single largest gap.

**Verdict:** The architecture as designed (async orchestrator → engine/sentiment/patterns loops → SQLite/Redis → Streamlit dashboard) is sound and we will keep it. Phase 1 is primarily *filling holes to the existing interfaces*, not rewriting.

---

## 2. How the system runs end-to-end (as designed vs. as-is)

### As designed (traced from `src/main.py`)

```
python src/main.py --mode paper
 └─ parse args → set ENABLE_LIVE_TRADING / ENABLE_PAPER_TRADING env vars
 └─ AITradingSystem(config_path='config/trading.yaml')
     ├─ _load_config: YAML + env overrides → Config(**data)          [Config class MISSING]
     ├─ _initialize_components:
     │    ├─ DatabaseManager(config.database)                        [exists, but called wrong — see §5.3]
     │    ├─ MetricsCollector, HealthChecker (src.utils.monitoring)  [module MISSING]
     │    ├─ EnsembleModel(config.models) (src.models.ensemble)      [module MISSING]
     │    ├─ SentimentAnalyzer(config.sentiment)                     [module MISSING]
     │    ├─ PatternDiscoveryEngine(config.patterns)                 [module MISSING]
     │    └─ TradingEngine(config, models, db, metrics)              [module MISSING]
     └─ start() → asyncio.gather of 5 loops:
          ├─ _run_trading_loop        every cycle_interval (60s): trading.run_cycle(),
          │                           retrain check (24h or Sharpe < 1.0, keep old model
          │                           unless improvement > min_improvement)
          ├─ _run_sentiment_analyzer  every 300s: analyze_batch(symbols) → db + engine
          ├─ _run_pattern_discovery   every 3600s: discover() → _evaluate_pattern()
          │                           [placeholder: returns hardcoded 0.05]
          ├─ _run_health_server       aiohttp on :8080 (/health, /ready)
          └─ _run_metrics_exporter    prometheus_client on :9100
 └─ SIGINT/SIGTERM → shutdown(): close all positions, write data/shutdown_state.json
```

### As-is

`python src/main.py` dies at **line 23** (`from src.trading.engine import TradingEngine` → `ModuleNotFoundError`). Even before that, if `logs/` doesn't exist, the module-level `logging.basicConfig(FileHandler('logs/…'))` at line 35–42 raises `FileNotFoundError` — the directory is only created in the `__main__` block, which runs *after* imports. Even `import src` alone fails, because `src/__init__.py` line 13–14 does `from . import trading` / `from . import models` (neither package exists), and **every subpackage `__init__.py` is a byte-identical copy of that same broken file** (`src/backtesting/__init__.py`, `src/dashboard/__init__.py`, `src/patterns/__init__.py`, `src/sentiment/__init__.py`). `src/utils/` has **no** `__init__.py` at all.

The separate sentiment service (`src/sentiment/main.py`) similarly dies importing `src.sentiment.collectors`, `.analyzer`, `.aggregator` — none exist.

---

## 3. Component inventory and state

| Component | File | Lines | State |
|---|---|---|---|
| Orchestrator `AITradingSystem` | `src/main.py` | 453 | **Structurally complete, cannot start.** 7 missing imports, config-shape mismatch, log-dir crash, signal handler creates a task that's never awaited, `shutdown()` runs twice (signal + `finally`). Retraining logic and safeguards are written but call methods that don't exist yet. |
| Trading engine `TradingEngine` | `src/trading/engine.py` | — | **Does not exist.** Must be built (paper + live execution, sentiment injection via `update_sentiment()`, `add_pattern()`, `run_cycle()`, `close_all_positions()`). |
| Root `engine.py` (`OptionEngine`) | `engine.py` | 369 | **Unrelated dead code.** A stock-*options* Streamlit mock (SPY/QQQ/IWM, iron condors, `np.random` P&L). Nothing imports it. Archive it. |
| Backtester `AdvancedBacktester` | `src/backtesting/engine.py` | 853 | **Best code in the repo. ~90% real.** Orders, limit/stop fills, slippage, commissions, margin calls, full metric suite (Sharpe, Sortino, Calmar, Omega, VaR, CVaR). Two real bugs: short-sale cash double-count (§5.4) and single-symbol price fallback. No look-ahead in fills, but strategy wrapper breaks it (next row). |
| Strategies + runner + optimizer | `src/backtesting/backtest_module.py` | 679 | **Strategies real, harness broken.** MA-crossover / RSI mean-reversion / Breakout are complete implementations with Kelly-fraction sizing (`calculate_position_size`, conservative 0.25×) and total-exposure caps. **Fatal flaw:** `BacktestRunner.strategy_func` hands each strategy a **one-row** DataFrame while every strategy computes 20–50-bar rolling windows and `iloc[-2]` → every backtest produces zero trades or crashes. Symbols hardcoded to `'BTC/USDT'`. `MLEnsembleStrategy._get_model_prediction` returns dummy `(1, 0.75)`; `_get_sentiment_score` returns `0.6`. `StrategyOptimizer` grid-searches **in-sample only**. Imports `talib` + two missing modules. |
| Pattern pipeline | `src/patterns/` (6 files) | 6,512 | **~80% real, self-contained, heavy.** DataLoader → FeatureExtractor (geometric/statistical/fractal features, Hurst, entropy) → Detector (RF/GB/XGB-style ensemble + optional TF deep models, regime detection, `retrain_with_feedback`) → Trainer (optuna tuning, walk-forward-ish CV) → Postprocessor (validation scoring). Issues: queries table `ohlcv_data` (doesn't exist — DB layer uses `ohlcv`); **silently falls back to seeded `np.random` synthetic data** (would "discover" patterns in noise); hard-imports `talib` and `optuna` (not installed); TF optional. |
| `EnsembleModel` | `src/models/ensemble.py` | — | **Does not exist.** `config/models.yaml` weights (transformer .4, tcn .3, rl .2, patterns .1) reference model files that are 32 KB stubs (see Models row). |
| Sentiment service | `src/sentiment/main.py` | 353 | **Shell only.** Redis plumbing, health loop, and aggregation flow are written, but the three modules doing actual work (collectors / analyzer / aggregator) are missing. `_store_in_database` is a stub. Symbols are `BTC,ETH,SOL` (no `/USDT` mapping). |
| Database layer | `src/utils/database.py` | 865 | **~85% real.** SQLite + PostgreSQL, OHLCV/trades/patterns/sentiment/model-version ops, training-data labeling. Bugs: treats any dict config as *PostgreSQL* (main.py passes a dict for SQLite → instant `KeyError: 'user'`); `store_ohlcv` blind-appends (duplicate rows / UNIQUE violations); expects trades columns `entry_time`/`exit_time`/`pnl_percentage`/`commission`/`slippage` which the *live DB doesn't have* (§5.2). |
| Metrics library | `src/utils/metrics.py` | 749 | **Complete and solid.** `MetricsCalculator` (all ratios, drawdown periods, rolling metrics, alpha/beta vs benchmark), `MetricsCollector`, `PerformanceTracker`. But note: main.py imports `MetricsCollector` from missing `src.utils.monitoring`, and calls methods (`record_error`, `record_sentiment`, `update_system_metrics`, `get_recent_performance`) that this class doesn't have. |
| Dashboard | `src/dashboard/app.py` (not `dashboard/`) | 1,500 | **UI complete, data fake.** `_calculate_current_pnl` → `np.random.uniform(-500, 1000)`. Equity curve, benchmark, returns/risk/trade metrics, strategy breakdown, model analytics, feature importance, sentiment timeline: all hardcoded or random demo data. Queries `ohlcv_data` table and `timestamp` trades column (both wrong vs. DB layer). `_generate_report()` is `pass`. Pattern tab commented out. SQL built via f-string interpolation (injection-prone). |
| DB init scripts | `init_database_script.py` / `init_db_script.py` | 308/349 | **Two conflicting schemas.** Script A (`init_database_script.py`): trades with `entry_time`/`exit_time`/commission/slippage + 9 tables incl. `performance_tracking`, `alerts`, `system_config` — matches DatabaseManager. Script B (`init_db_script.py`): trades with single `timestamp`, no commission/slippage, `model_performance` instead of `model_versions`. **The existing `data/trading_system.db` was created by script B** (the wrong one) — fortunately it's empty (0 rows), so we can rebuild from a single canonical schema. |
| CLI | `cli_tool.py` | 417 | Real (click-based start/stop/status/positions/performance/backtest), but shells out to missing `src/backtest.py` and imports `tabulate` (not in requirements). |
| Test harness | `test_system_script.py` | 532 | Integration self-check; imports the same missing modules, so it can't run. No `tests/` directory; `make test` targets it. |
| Model downloader | `download_models_script.py` | 368 | Downloads from `https://example.com/models/…` — placeholder URLs. FinBERT part is real. |
| Models on disk | `models/` | — | `transformer_btc/eth.pt`, `tcn_btc/eth.pt`, `rl_agent.pt` are **~32 KB each = untrained stubs** (no SOL variants at all). `models/finbert/` is a real 836 MB FinBERT — usable for sentiment. |
| Configs | `config/trading.yaml`, `config/models.yaml` | 18/24 | Valid YAML but **shape mismatch**: main.py reads `config.trading`, `config.features`, `config.sentiment`, `config.patterns`, `config.database`, `config.models` — none of these top-level keys exist in `trading.yaml` (it has `symbols`, `timeframes`, `execution`, `risk_management` at top level). No commission rate in config (only `.env` `BACKTEST_COMMISSION=0.001`). |
| Docker | `Dockerfile.{trading,sentiment,dashboard}`, `docker-compose.yml` | — | Well-intentioned, currently unbuildable: `COPY scripts/` (dir missing), `requirements-dashboard.txt` / `requirements-sentiment.txt` missing, compose mounts `./scripts/init.sql`, `config/prometheus/`, `config/grafana/`, `config/nginx/` — all missing. Dockerfile.trading installs torch 2.0.1 **over** requirements' torch 1.13.1 (torchvision 0.14.1 then incompatible). 9 services incl. Grafana/Prometheus/Jupyter/Nginx — overkill for v1. |
| Makefile | `makefile.txt` | 190 | Good command set, but named `.txt` (make won't find it) and references 6 missing files (`src/backtest.py`, `src/train.py`, `cli.py`, `scripts/init_db.py`, `scripts/download_models.py`, `requirements-dev.txt`). |
| Env files | `.env`, `env_file.sh`, `env_template.sh` | — | Rich, well-organized template. **See security note §7 — real keys are sitting in these files.** |
| Data | `data/` | — | `sample_data.csv` (8,737 hourly rows of synthetic BTC), `pattern_definitions.json` (3 pattern configs), empty DB. No real historical data yet. |
| Junk | `#/`, `a/`, `called/`, `create/`, `folder/`, `from/`, `it/`, `recreate/`, `scratch/`, `".venv"/` (smart quotes) | — | Ten broken Python-3.9 virtualenvs whose names spell out a sentence — clearly a shell quoting accident (`python -m venv` called with a sentence's words as arguments). Safe to delete. Real `.venv/` is Python 3.9.1. |

---

## 4. Complete gap list

### 4.1 Missing modules (imported by existing code)

| Missing file | Imported by | Needed symbol(s) |
|---|---|---|
| `src/trading/engine.py` | `src/main.py:23` | `TradingEngine` (`run_cycle`, `update_sentiment`, `add_pattern`, `close_all_positions`) |
| `src/models/ensemble.py` | `src/main.py:24`, `backtest_module.py:18` | `EnsembleModel` (`retrain`, `get_active_models`) |
| `src/sentiment/analyzer.py` | `src/main.py:25`, `sentiment/main.py:90` | `SentimentAnalyzer` (`analyze_batch`), `CryptoSentimentAnalyzer` |
| `src/sentiment/collectors.py` | `sentiment/main.py:89` | `RedditCollector`, `TwitterCollector`, `NewsCollector` |
| `src/sentiment/aggregator.py` | `sentiment/main.py:91` | `SentimentAggregator` |
| `src/patterns/discovery.py` | `src/main.py:26` | `PatternDiscoveryEngine` (`discover`) |
| `src/utils/monitoring.py` | `src/main.py:28` | `MetricsCollector` (w/ `record_error`, `record_sentiment`, `update_system_metrics`, `get_recent_performance`, `get_summary`), `HealthChecker` (`check_all`, `check_ready`) |
| `src/utils/config.py` | `src/main.py:29` | `Config` (attribute access: `.database .models .features .sentiment .patterns .trading`) |
| `src/utils/indicators.py` | `backtest_module.py:17` | `TechnicalIndicators` |
| `src/utils/__init__.py` | package import | (empty is fine) |
| `src/backtest.py`, `src/train.py`, `cli.py`, `scripts/init_db.py`, `scripts/download_models.py`, `requirements-dev.txt` | makefile.txt, cli_tool.py | CLI entry points |
| `requirements-dashboard.txt`, `requirements-sentiment.txt`, `scripts/init.sql`, `config/prometheus/…`, `config/grafana/…`, `config/nginx/…` | Dockerfiles, docker-compose | build/deploy assets |
| `tests/` | `make test` | actual test suite |

### 4.2 Placeholder / random / demo implementations to replace

1. `src/main.py:349` `_evaluate_pattern` → hardcoded `0.05` (must run a real mini-backtest).
2. `backtest_module.py:446` `MLEnsembleStrategy._get_model_prediction` → dummy `(1, 0.75)`; `:452` `_get_sentiment_score` → `0.6`.
3. Dashboard: `_calculate_current_pnl` (random), `_create_performance_chart` (random equity + random BTC benchmark), `_calculate_returns_metrics` / `_calculate_risk_metrics` / `_calculate_trade_statistics` / `_get_strategy_performance` / `_get_model_performance_history` / `_get_model_comparison` / `_get_feature_importance` / `_get_pattern_type_distribution` (all hardcoded), `_get_trade_history` fallback (seeded random 100 trades), `_create_sentiment_timeline` (random), `_generate_report` (`pass`).
4. `patterns/data_loader.py` `_generate_sample_data` fallback — synthetic data must **never** silently substitute for real data in a money system; make it an explicit test-only mode.
5. `sentiment/main.py:_store_in_database` — logs instead of storing.
6. `download_models_script.py` — example.com URLs.
7. Root `engine.py` — entire file is mock.
8. `models/*.pt` — 32 KB untrained stubs.

### 4.3 Schema mismatches (single worst source of dashboard bugs)

| Concept | database.py expects | init_database_script.py | init_db_script.py (＝ current DB) | dashboard queries |
|---|---|---|---|---|
| OHLCV table | `ohlcv` | `ohlcv` ✓ | `ohlcv` ✓ | `ohlcv_data` ✗ (also patterns/data_loader.py) |
| Trade open time | `entry_time` | `entry_time` ✓ | `timestamp` ✗ | `timestamp` ✗ |
| Trade close time | `exit_time` | `exit_time` ✓ | — ✗ | — |
| `pnl_percentage`, `commission`, `slippage` | required | present ✓ | missing ✗ | expected |
| Model tracking | `model_versions` | `model_versions` ✓ | `model_performance` ✗ | — |
| `sentiment_scores`, `discovered_patterns` | ✓ | ✓ (richer) | ✓ (basic) | — |

**Resolution (Phase 1):** one canonical schema based on `init_database_script.py` (the richer one that matches `DatabaseManager`), one init/migration script, delete the other; fix `ohlcv_data` → `ohlcv` in dashboard + data_loader; kill every `timestamp`-column fallback query. Current DB is empty → drop and recreate.

### 4.4 Logic bugs in existing code

1. **`DatabaseManager.__init__` dict handling** — `main.py` passes `{'path': …, 'postgres_url': …}`; any dict is routed to `_init_postgresql(config)` → `KeyError: 'user'`. Must accept `{'path': …}` as SQLite.
2. **`BacktestRunner.strategy_func` single-row DataFrame** — strategies need rolling history; as written no strategy can ever fire. Rewrite the runner to feed a growing/sliding window per symbol.
3. **Short-sale cash double-count** (`AdvancedBacktester`): open short adds full sale proceeds to cash *and* close adds `gross_pnl` again (proceeds − buyback already equals gross_pnl) → shorts inflate equity by the position value. `allow_shorting` defaults to `True`. Fix the accounting *and* default to long-only for our spot-only v1.
4. **`main.py` logging before dirs exist** → crash on fresh checkout.
5. **Signal handler** calls `asyncio.create_task(self.shutdown())` from a sync handler and the task is never awaited; combined with `finally: await system.shutdown()` shutdown can run twice (double position-close attempts).
6. **Strategies hardcode `'BTC/USDT'`** in every signal regardless of the data's symbol.
7. **`StrategyOptimizer` optimizes in-sample** on the full dataset — guaranteed overfit; Phase 2 replaces with walk-forward.
8. `store_ohlcv` blind `to_sql(append)` — duplicates on re-fetch; with the canonical UNIQUE index, whole batches will fail. Needs upsert/ignore.
9. Dashboard SQL assembled with f-strings (injection-prone; also breaks on `'` in inputs).
10. `_align_data` multi-symbol handling: base symbol unprefixed, others prefixed, generic `close` fallback — brittle; standardize on per-symbol prefixed columns.
11. Sentiment symbols `BTC,ETH,SOL` vs trading symbols `BTC/USDT,…` — needs a mapping.

---

## 5. Dependency audit

**Environment found on this Mac:** default `python3` = 3.9.1 (`/usr/local/bin`); Homebrew provides **3.10.18** and **3.13.5**; Docker 28.3.0 installed and working. Existing `.venv` is Python 3.9.1 (below the code's own declared minimum of 3.10 in `src/__init__.py`).

**Pinned stack is mid-2023 era:** numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, ccxt 4.0.3, torch 1.13.1(+torchvision 0.14.1), streamlit 1.24.1, transformers 4.26.1, scikit-learn 1.2.2.

Issues found:

| # | Issue | Severity | Resolution plan |
|---|---|---|---|
| 1 | **TA-Lib**: imported unconditionally in `backtest_module.py`, `patterns/data_loader.py`, `patterns/feature_extractor.py`, but commented out of requirements ("requires C++ deps"). | Blocker | Make TA-Lib optional: vendor a small pure-pandas indicator module (`src/utils/indicators.py` — which is *already* an expected missing module) implementing the ~12 indicators actually used (SMA/EMA/RSI/MACD/ATR/BBANDS/STOCH/ADX/OBV/AROON/candle patterns), with `talib` used when importable. The `ta` package (in requirements ✓) covers most as fallback reference. |
| 2 | **`optuna`** imported by `patterns/model_training.py`, not in requirements. | Blocker | Add to requirements (or guard import). |
| 3 | **`prometheus_client`** used by `main.py`, commented out in requirements. | Blocker | Add. |
| 4 | **`tabulate`** used by `cli_tool.py`, not in requirements. `click` arrives only as a transitive Streamlit dep. | Minor | Add both explicitly. |
| 5 | **`pandas-ta 0.3.14b0`** is abandoned; uses `np.NaN` (removed in numpy ≥ 2) — breaks on modern numpy. | Medium | Drop it; rely on our vendored indicators. |
| 6 | **`gym 0.26.2` + `stable-baselines3 2.0.0`** conflict — SB3 2.x switched to `gymnasium`. | Medium | RL agent is a stub anyway → remove both from v1 requirements. |
| 7 | **Dockerfile torch conflict**: `pip install torch==2.0.1` after requirements installed torch 1.13.1 + torchvision 0.14.1 → broken pair. | Blocker (docker) | Align on one torch version; drop torchvision (unused). |
| 8 | **python-binance + ccxt** both pinned; only ccxt used. `yfinance` unused. | Minor | Remove. |
| 9 | **psycopg2-binary/SQLAlchemy** duplicated intent (commented block + appended lines). `database.py` imports psycopg2 unconditionally even for SQLite. | Minor | Keep SQLAlchemy; make psycopg2 import lazy/optional (SQLite is our v1 DB). |
| 10 | **macOS/Python-version fit**: the 2023 pins have wheels for py3.10 (incl. Apple Silicon) but *not* 3.13. `src/__init__.py` requires ≥ 3.10. | Decision | **Target Python 3.11** in a fresh venv with moderately modernized pins (numpy 1.26.x, pandas 2.2.x, ccxt current, streamlit current, torch 2.x). We are already fixing the code that pandas 2.x deprecations touch (`fillna(method=…)`, `resample('M')`, epoch conversions). Rationale: 4.0.3-era ccxt is 2 years stale against live Binance endpoints/filters — for a money system, the exchange client must be current. |

Deprecation hotspots to fix while touching files anyway: `fillna(method='ffill')` (backtester `_align_data`, data_loader), `resample('M')` → `'ME'` (metrics), `datetime.now()` naive timestamps everywhere (standardize on UTC), `applymap` → `map` (dashboard styling).

---

## 6. Fundamental-flaw assessment (honest opinions, with evidence)

1. **The ML ensemble is aspirational, not real.** No `EnsembleModel` module, 32 KB untrained weight stubs, dummy predictions hardwired in `MLEnsembleStrategy`, no training pipeline for transformer/TCN/RL (only the *pattern* classifier trainer exists). **Minimal change:** ship v1 gated on the three classical strategies (MA crossover, RSI mean-reversion, breakout) + regime filter + the existing Kelly sizing; implement `EnsembleModel` as a clean interface that starts with patterns+classical signals and can absorb deep models later *if they ever earn their weight in walk-forward tests*. Pretending the 0.4/0.3/0.2/0.1 ensemble exists would be exactly the kind of self-deception Phase 2 is designed to catch.
2. **The backtest harness invalidates every historical claim.** Because strategies receive one row, any past "backtest results" from this repo were structurally impossible. Nothing needs re-litigating: Phase 2 starts from zero with the fixed harness and walk-forward validation.
3. **Silent synthetic-data fallbacks are dangerous** (`PatternDataLoader._generate_sample_data`, dashboard demo data, `OptionEngine` mocks). In a money system, "no data" must be a loud failure, never fabricated data. Phase 1 policy: demo/synthetic paths only behind an explicit `--synthetic` flag used by unit tests.
4. **Two init scripts, three schemas** — resolved per §4.3.
5. **Risk controls exist only as config values.** `max_drawdown: 0.15`, `max_position_size: 0.1` are read by nobody today. Phase 4 hard-codes enforcement in the engine as specified (kill switch, daily loss limit, position caps, exchange filters).

---

## 7. Security notes (act on this independent of the code)

- **Live API keys are present in plaintext** in `.env` *and* in `env_file.sh` (Binance main + testnet keys, Reddit, Twitter bearer token, CryptoCompare, NewsAPI, Telegram bot token, AWS access keys, SMTP-ish alert email). This folder has been shared/moved (it's in Downloads). **Recommendation: rotate the Binance keys and AWS keys now**, keep only `.env` (gitignored) and a sanitized `env_template.sh`. When creating new Binance keys for Phase 4: enable *reading* + *spot trading* only, **disable withdrawals**, and IP-restrict them. The Phase 4 engine will additionally refuse to start if the key has withdrawal permission.
- Dashboard builds SQL with f-strings — fix during Phase 1 rewiring.
- `docker-compose` exposes Redis :6379 and Postgres :5432 on the host with default/blank passwords — will be locked down in Phase 3 deployment.

---

## 8. What stays, what goes, what gets built

**Keep as-is (foundation):** `AdvancedBacktester` (with 2 bug fixes), `MetricsCalculator`/`PerformanceTracker`, `DatabaseManager` (with dict-config + upsert fixes), strategy implementations' signal logic, pattern pipeline (deferred integration), dashboard UI structure, Docker/compose skeleton, config layout, `AITradingSystem` orchestration flow.

**Archive (dead/wrong):** root `engine.py` (options mock), `init_db_script.py` (wrong schema), 10 junk venv dirs, stub `.pt` files (regenerate when real training lands).

**Build in Phase 1 (to existing interfaces):**
1. `src/utils/config.py` — `Config` matching main.py's attribute access, normalizing `trading.yaml` + env.
2. `src/utils/monitoring.py` — `MetricsCollector` (with the methods main.py calls, Prometheus-backed), `HealthChecker`.
3. `src/utils/indicators.py` — `TechnicalIndicators` pure-pandas, talib-compatible where used.
4. `src/trading/engine.py` — `TradingEngine`: ccxt market data, paper fill simulation (0.1% commission, 0.05% slippage, LIMIT semantics), position/equity persistence via DatabaseManager, strategy dispatch, `run_cycle`/`update_sentiment`/`add_pattern`/`close_all_positions`.
5. `src/models/ensemble.py` — honest `EnsembleModel` (see §6.1).
6. `src/sentiment/analyzer.py` (+ minimal collectors/aggregator) — VADER/FinBERT-based, no-API-keys-required baseline (news via free endpoints), keyed APIs optional.
7. `src/patterns/discovery.py` — `PatternDiscoveryEngine` wrapping the existing loader→extractor→detector→postprocessor chain, real-data-only.
8. `scripts/init_db.py` — canonical schema; `scripts/backfill.py` — ≥ 12 months ccxt OHLCV backfill for 3 symbols × 4 timeframes.
9. Fixed `__init__.py` files, `Makefile` (renamed), requirements split, real `_evaluate_pattern` mini-backtest, dashboard rewired to real queries, graceful shutdown hardening.

---

## 9. Assumptions stated

- **Python 3.11 + modernized pins** (per §5.10) rather than resurrecting the 2023 stack on 3.9. If you prefer zero dependency movement, say so and I'll pin to 3.10 with the original versions — but ccxt must still be upgraded for live-trading correctness.
- **SQLite is the v1 database** (PostgreSQL path kept but untested until needed). Single-process bot + dashboard reader is well within SQLite's envelope.
- **v1 trades long-only spot on Binance** (per your constraints); backtester keeps shorting capability but our configs disable it.
- **Sentiment v1 runs keyless** (public news feeds + FinBERT/VADER); Reddit/Twitter collectors activate only if you provide working keys (rotate first).
- The five 32 KB `.pt` stubs contain nothing worth preserving.

---

*End of Phase 0 report. Phase 1 begins by making `python src/main.py --mode paper` start cleanly against this exact plan.*
