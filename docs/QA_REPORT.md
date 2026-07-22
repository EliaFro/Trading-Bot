# Final Quality Pass — QA Report

**Date:** 2026-07-10 · **Scope:** QA, polish, and hardening of what exists. Zero strategy work, zero new features, zero new studies.
**Outcome:** all findings fixed, every non-cosmetic fix covered by a regression test, full Definition of Done ticked. The system is **frozen** as of the final line of this report.

---

## 1. Findings table

Every issue found in the pass, worst first. "Regression test" names the test that now fails if the bug ever returns.

| # | Severity | Finding | Root cause | Fix | Regression test |
|---|----------|---------|------------|-----|-----------------|
| 1 | **CRITICAL** | Paper equity did **not** reconcile against the trade ledger: daily lab off by **$8.00** (8 closed trades), fast lab by **$2.00** (2 closed trades) — exactly $1.00 per closed trade. | The live engine's `_close_position` omitted the **entry-leg commission** from recorded `pnl` (and the `commission` column held only the exit leg). The backtester had been fixed in Phase 2; the live engine was missed — and `test_round_trip_pnl_math` had *encoded the buggy expectation*, so it passed. | Engine now carries `entry_commission` in the position dict (open, restore-from-DB, and graceful-close paths) and records `pnl` and `commission` with **both legs**. The 12 affected historical rows were repaired with a guarded SQL update (guard: only rows whose commission < both-legs minimum). Both labs now tie out to **$0.0000**. | `test_paper_engine.py::test_round_trip_pnl_math` (corrected to assert the true math, both commission legs, and the reconciliation identity `cash == initial + Σ closed pnl`) |
| 2 | HIGH | Pattern detector in an hourly error loop: `X has 220 features, but StandardScaler is expecting 223`. | Scaler/PCA fitted at one feature width; the extractor's width drifted after a retrain; transform then raised every cycle, forever. | `_preprocess_features` checks `n_features_in_` against the incoming width and **refits** (with a log line) on drift instead of raising. | `test_qa_hardening.py::test_detector_survives_feature_width_drift` |
| 3 | HIGH | No staleness indication anywhere: a stalled pipeline or dead bot would display yesterday's data as if it were current. | Dashboard never compared data timestamps to now. | Freshness header on every tab: last candle / last bot cycle / last ML decision from `ohlcv`, `performance_tracking`, `ml_predictions`, 🟢/🔴 against thresholds (15 min / 10 min / 26 h) plus a **STALE** warning banner; all times labeled **UTC**. | `test_dashboard_contract.py` (header exists, renders before any tab, watches all three tables, UTC labeled) |
| 4 | HIGH | Simulated money not consistently labeled: equity, performance, and trade-history panels showed dollar figures with no PAPER marking. | — | "🧪 PAPER ACCOUNT" captions on every simulated money figure; the Playbook ledger is explicitly "MY REAL LEDGER". A stranger cannot mistake simulation for money on any screen. | `test_dashboard_contract.py` (PAPER labels, ledger label) |
| 5 | HIGH | Dead sidebar controls: Start/Stop buttons published commands to a Redis channel **nothing subscribes to** — a silent no-op dressed as control. | Leftover from the inherited codebase's abandoned command bus. | Controls removed; replaced with a "read-only by design" caption pointing at the real `make` commands. | `test_dashboard_contract.py` (no publish-to-nowhere controls in source) |
| 6 | HIGH | Dashboard crashed at startup with `NameError: name 'Path' is not defined` (found when run outside the service environment). | A refactor dropped `from pathlib import Path`. | Import restored (commit `55d355f`). | Contract tests parse/execute the module source; full suite imports the app |
| 7 | MEDIUM | Playbook tab: the buy form accepted absurd inputs, the rule text paraphrased PLAYBOOK.md, and the lump-sum comparison had no methodology note. | — | Form validation caps (max $1,000,000, no negatives), rule text now **verbatim** from PLAYBOOK.md, comparison methodology caption, days-in-regime shown. | `test_dashboard_contract.py` (playbook contract: never-places-orders text, verbatim rule, validation, methodology) |
| 8 | MEDIUM | Panels showed silent blanks before first data; DB-unreachable produced a stack trace. | No empty/error states. | Honest empty states with expectations ("the ML decides once per day, just after 00:00 UTC"); DB-unreachable message says what to do (`make service-status`). | `test_dashboard_contract.py` (banners/states render before data loads) |
| 9 | MEDIUM | `launchd_dashboard.err.log` unbounded (6.9 MB and growing from Streamlit chatter); file-watcher churn. | Streamlit defaults in the launchd plist. | Plist now passes `--logger.level error --server.fileWatcherType none`; log trimmed. App logs were already rotation-capped. | `test_qa_hardening.py::test_log_rotation_configured` (50/20/5 MB caps) |
| 10 | MEDIUM | `.env.example` drift: 7 documented variables nothing reads; 3 variables the code reads were missing (`DATABASE_URL`, `PLAYBOOK_CHECK_HOUR_UTC`, `SENTIMENT_UPDATE_INTERVAL`). | Doc rot across phases. | Template reconciled in both directions. | `test_docs_contract.py::test_env_example_covers_every_var_the_code_reads` (scans code for every env read, incl. config.py's mapping-tuple idiom) |
| 11 | MEDIUM | OPERATING.md §4 routed operators through a Docker log command; the supported deployment is launchd. | Doc rot from the Docker era. | Replaced with `tail -n 200 logs/trading.log`. | `test_docs_contract.py::test_operating_doc_does_not_route_users_through_docker` |
| 12 | MEDIUM | `cli_tool.py` orphaned (nothing referenced it; its imports were stale). | Inherited dead code. | Moved to `_quarantine/` per the standing quarantine rule. | `test_docs_contract.py` link/command checks would catch dangling references |
| 13 | MEDIUM | Host disk hit 97% full (418 MiB free) mid-pass — an operational risk to the SQLite WAL and log writes, though **not caused by this project** (project total incl. data+models ≈ 1.6 GB). | Machine state: pip cache 387 MB; two stale 1.6 GB stray venvs sitting in `_quarantine/stray_venvs` since Jul/Sep 2025; `Docker.raw` ~2.5 GB actual; app-updater caches ~2.6 GB. | Cleaned pip cache; deleted the quarantined **stray venvs** (pure pip artifacts, zero project code — package inventory preserved in `_quarantine/stray_venvs/INVENTORY.md` before deletion). Now 4.1 GB free. **User note:** `Docker.raw` (~2.5 GB) and app-updater caches (~2.6 GB) are yours to reclaim if you want more headroom. | Environmental — no test. (The WAL-under-pressure behavior itself is tested: finding 15.) |
| 14 | MEDIUM | Fresh-environment install was unproven (and first attempt died on the full disk). | — | Fresh Python 3.11 venv + `pip install --no-cache-dir -r requirements.txt` + import smoke test of every core module (engine, kill_rule, companion, detector, database, config, dashboard, torch, streamlit, ccxt, sklearn, pandas): **ALL PASS**, exit 0. Venv deleted after. | One-shot verification (environmental); requirements.txt itself unchanged |
| 15 | MEDIUM | Concurrent readers + writer on the shared SQLite DBs unproven under load. | — | WAL mode confirmed on all three DBs; hammer drill made permanent: 300 writes against 3 concurrent readers, zero dropped writes. | `test_qa_hardening.py::test_wal_concurrent_read_write` |
| 16 | COSMETIC | `p_up`/`p_down`/F1 columns displayed raw floats (`0.30000000000000004`-style). | Float repr. | Rounded display columns (storage unchanged, full precision). | — (cosmetic) |
| 17 | COSMETIC | README's three findings charts existed on disk but weren't embedded. | — | Embedded via relative paths (render verified mechanically; see DoD). | `test_docs_contract.py::test_every_relative_markdown_link_resolves` |

## 2. Failure drills — all pass with graceful, honest behavior

| Drill | Result |
|---|---|
| `kill -9` each of the four launchd agents (paper bot, fast-lab bot, playbook companion, dashboard) | All four respawned by launchd (`KeepAlive SuccessfulExit=false`); engines restored cash + positions from the DB on restart; verified via launchctl-reported PIDs (avoiding the pgrep self-match trap). |
| Network down (injected `ccxt.NetworkError` on every fetch) | Cycle completes; error recorded to alerts; **no fabricated prices, no position action**. Now a permanent test: `test_qa_hardening.py::test_cycle_survives_network_down`. |
| Signal with no price available | `_open_position` refuses (no trade on a guess). Permanent test. |
| DB contention (WAL hammer) | 300 writes vs 3 concurrent readers — zero dropped writes. Permanent test. |
| Dashboard with DB unreachable | Plain-language message + remedy (`make service-status`); no stack trace. |
| Kill-rule deadline simulation (clock ≥ 2026-08-07, criteria failing) | Sentinel + DB row written; lockout **survives restart**; there is no bypass parameter — re-enabling requires code changes. (Build-2 test suite re-run green.) |
| Sentiment 6-month evaluation guard | Refuses to run before 2027-01-05 (180-day guard) — verified by invocation. |
| Graceful stop | `close_all_positions('shutdown')` closes everything and records exit reason (tested). |
| Telegram paths (all TEST-marked messages) | Kill-path alert ✅ delivered · monthly digest ✅ delivered · companion daily check ✅ delivered (reported BTC **BELOW** 200-day by −14.9%, cross-checked correct). |

Scheduled events verified live: companion daily check (with missed-run and stale-data branches), monthly digest generation, kill-rule daily evaluation, sentiment-eval date lock.

## 3. Resource report

31-minute sampling window, 128 samples, all four services live, Apple Silicon:

| Service | CPU (mean) | RSS (mean) |
|---|---|---|
| Dashboard | 0.2% | 202 MB |
| Fast-lab bot (1m) | 9.3% | 353 MB |
| Paper bot (daily ML) | 0.0% | 807 MB |
| Playbook companion | 0.0% | 44 MB |
| **Total** | **≈ 9.0% of one core** | **mean 1.32 GB, max observed 2.05 GB** |

Disk: data+logs grow ≈ 1.3 MB/day → **≈ 0.5 GB over 12 months**; all app logs rotation-capped (50/20/5 MB). The machine, not the project, is the disk risk — see finding 13.

## 4. Data integrity

- **Coverage:** 36 months × 3 symbols, `verify_coverage.py` reports **100% gap-free**, current to within one bar (a 3-bar catch-up ran during the pass).
- **Equity reconciliation (the CRITICAL item):** identity `cash = initial + Σ closed pnl − Σ open entry cost (incl. entry fee)` — daily lab **$0.0000**, fast lab **$0.0000** after the finding-1 fix and ledger repair. Pre-fix deltas were −$8.00 and −$2.00.
- **Honesty consequence of the repair:** the July digest was regenerated from the corrected ledger. The daily lab's live profit factor on its 8-trade record fell from **4.52 to 0.84** — the pre-fix number had been flattered by the missing entry fees. The committed digest is the corrected one.
- **ML records:** retrain-log champion == on-disk champion for both labs; every keep-vs-replace row carries its reasoning; displayed probabilities rounded, stored at full precision.

## 5. Security re-check

- `.env` mode 600, gitignored; `.env.example` placeholders only (and now exactly matches what the code reads).
- Git history secret scan re-run across all commits: **clean**.
- Log scan: secrets appear only masked (e.g. `******9157`); no raw env values echoed anywhere.
- Live-mode guards intact and tested: refuses withdrawal-enabled API keys and a world-readable `.env`; `allow_shorting` defaults to False; paper mode is the only installed mode.

## 6. Documented commands — executed verbatim

`make install` ✓ · `make db-init` ✓ · `python scripts/backfill.py --months 36` ✓ (idempotent catch-up) · `python scripts/verify_coverage.py` ✓ · `make test` ✓ · `make service-status` ✓ · `make status` ✓ · `python scripts/clear_kill_switch.py` ✓ ("Kill switch is NOT triggered") · `python scripts/generate_digest.py` ✓ · `python scripts/run_sentiment_eval.py` ✓ (correctly refuses early) · log tail commands ✓ · and the five study scripts from README's Reproduce section, re-run end-to-end this pass:

| Command | Exit |
|---|---|
| `python scripts/run_signal_library.py` | 0 |
| `python scripts/run_daily_momentum.py` | 0 |
| `python scripts/run_ml_study.py` | 0 |
| `python scripts/run_fastlab_study.py` | 0 |
| `python scripts/run_ml_fast_study.py` | 0 |

Frozen evidence artifacts (metrics JSONs, charts, window tables) were restored from git after the reruns — the published results remain the pre-registered originals, not silently regenerated ones.

## 7. Executed Definition of Done

- [x] **Every dashboard tab handles loading/empty/error/stale states; zero demo data; honest banners un-hideable; freshness indicator present** — findings 3/4/5/8 fixed; source-level contract enforced by `tests/test_dashboard_contract.py` (9 tests: no `np.random`/demo tokens, banners render before data, freshness header before all tabs).
- [x] **Playbook tab math hand-verified; never-trades statement visible; form validated** — cost basis and lump-sum comparison hand-checked; "NEVER places orders" text asserted in tests; caps added (finding 7).
- [x] **All four services crash-tested and respawn with correct state** — §2 drill 1; positions/cash restored from DB (restore path also carries `entry_commission` post-fix).
- [x] **All failure drills pass with graceful, honest behavior** — §2 table; three drills promoted to permanent tests.
- [x] **Kill rule, digest, companion, and sentiment-lock verified by simulation; Telegram paths proven with TEST messages** — §2; all three TEST messages delivered.
- [x] **Equity reconciles to the cent against the trade ledger for both labs** — $0.0000 and $0.0000 (§4) after fixing CRITICAL finding 1.
- [x] **36-month data coverage still 100%, current to within one bar** — §4.
- [x] **Every documented command executes verbatim; every link resolves; charts render on github.com** — §6; all 23 markdown files pass the link test (now permanent: `test_docs_contract.py`); chart blobs verified present on `origin/main` at the exact relative paths README cites — the visual github.com check is the one item only a logged-in human can do (repo is private): **user, one glance, please**.
- [x] **Fresh-virtualenv install succeeds** — finding 14: clean venv, `--no-cache-dir`, full import smoke test, exit 0.
- [x] **Secrets scan clean; logs never echo env values** — §5.
- [x] **Full test suite green, count > 143, all fixes covered by regression tests** — **162 passed, 4 skipped** (baseline 143 → +19). New this pass: `test_dashboard_contract.py` (9), `test_qa_hardening.py` (5), `test_docs_contract.py` (5), plus the corrected round-trip/reconciliation test. (The 4 skips are the long-standing optional-dependency skips, unchanged.)
- [x] **QA_REPORT.md committed and pushed; remote verified** — the QA-pass commit hash and remote-HEAD verification are recorded in the "Final state" block below (added in a one-line bookkeeping commit after the push, since a commit cannot contain its own hash).

---

## Final state

- Tests: **162 passed, 4 skipped** (baseline 143).
- QA-pass commit: `0aee0f2` on `main`, pushed; remote HEAD verified equal to local (`0aee0f2bdf19cf226d91db7f28b1bb6e885e06f9`).
- Every box above is ticked with evidence; no finding remains open; the mandate's scope boundary held (zero strategy work, zero new features, zero new studies).

**The system is FROZEN as of commit `0aee0f2` (2026-07-10): no further changes without a new explicit mandate.**

---

## 8. Post-freeze addendum — status check-in, 2026-07-22 (user-mandated)

A full-system check-in on 2026-07-22 (12 days of unattended operation) verified: all four services healthy (one respawn total, the deliberate 7/13 QA kickstart), both labs reconciling to the cent at full float precision, 36-month coverage re-verified gap-free and current to within one bar, remote HEAD equal to local, zero unauthorized changes. It surfaced two defects, fixed here under the same QA discipline with the user's explicit approval:

| # | Severity | Finding | Root cause | Fix | Regression test |
|---|----------|---------|------------|-----|-----------------|
| 18 | MEDIUM | Monthly digest silently skipped if the host slept through the entire UTC 1st of the month — the trigger was a bare `now.day == 1` check with no catch-up, unlike the DCA-day message's LATE-recovery window. | Laptop-sleep edge case not covered in Build 3. | `should_generate_digest()` in `src/playbook/companion.py`: fires on the 1st, and catches up within the first `LATE_WINDOW_DAYS` (7) days when the month's digest file is missing; never regenerates an existing one. Runner routed through it. | `tests/test_postfreeze_checkin.py` (4 tests: fires on 1st, catches up, respects existing file, window closes, runner uses helper) |
| 19 | MEDIUM | Daily bot logged "Retraining did not improve performance (>2% required), keeping current models" nightly from the **legacy v1 ensemble** path, which is a structural no-op — no training or comparison ever happened. The line invited confusion with the real (weekly, separately logged) champion/challenger retrainer. | Legacy code path's log message written before the v1 no-op reality. | Ensemble's `retrain()` now returns `noop: True`; `main.py` logs the truth ("Legacy v1 ensemble: nothing to retrain…") while preserving the retrain-timer reset. | `tests/test_postfreeze_checkin.py` (2 tests: no-op flag returned; truthful branch ordered before the comparison, timer reset intact) |

Two record additions in the same mandate (documentation only): the **awake-hours sampling bias** of the live Fast Lab record is now a one-line disclosed limitation in `docs/FASTLAB_RESULTS.md` and `docs/ARTICLE_DRAFT.md` (the live record samples only hours the host is awake — measured July 2026: daily checks ran 5–13 h behind schedule); and OPERATING.md §4 now flags the **2027-01-05 sentiment evaluation as a MANUAL step** — the only scheduled event that does not run itself.

Services restarted to load the fixes (paper bot + companion; both flat/stateless at restart, launchd-verified back up). Tests after addendum: **168 passed, 4 skipped**.

**The freeze is re-declared as of this addendum's commit: no further changes without a new explicit mandate.**

---

## 9. Post-freeze addendum 2 — external evidence integration, 2026-07-22 (user-mandated, docs only)

A user-supplied compilation of the global literature on trading profitability was integrated into the record. **Zero code, config, strategy, or criteria changes** — five documentation edits:

1. **`docs/EVIDENCE_REVIEW.md` added** — the external literature compilation (Brazil/Chague 97%, Taiwan/Barber-Odean <1% with costs tripling losses, SEBI 91%, ESMA 74–89%, BIS crypto ~75%, Virtu one-losing-day/49%-profitable-exits, Harvey-Liu-Zhu t > 3.0, McLean-Pontiff decay, prop-firm ~7%-ever-paid, finfluencer antiskill), with a header stating it is external corroboration compiled after our studies concluded, never an input to our gates or verdicts.
2. **README.md** — new section "The broader evidence (we didn't discover this — we reproduced it)": our coin-flip result beside Harvey/Liu/Zhu's t > 3.0; our fee wall beside the Taiwan net-vs-gross cost finding and the Binance fee-tier asymmetry; our predictive ≠ profitable beside Virtu's spread-capture signature; our kill rule beside the field's pre-registration remedy.
3. **`docs/ARTICLE_DRAFT.md`** — the same four pairings woven in place (§4, §5, §6, §7), plus one closing-section line: the pre-registered Aug 7 stop is the field's answer to data-mining made physical.
4. **OPERATING.md §3** — an **annotation, not a criteria change**, recording that the academic multiple-testing standard (t > 3.0, Harvey/Liu/Zhu 2016) is stricter than our pre-registered t ≥ 2.0; any future pass at 2.0 gets flagged for scrutiny against 3.0 before being believed. The pre-registered criteria remain untouched in both directions, because editing a pre-registered bar after registration — in either direction — is the tampering the document exists to prevent.
5. **PLAYBOOK.md** — new informational section "What the broader evidence says": the literature's bottom line for sub-$10k capital (low-cost diversified index exposure), the BIS retail-crypto outcome numbers, and the observation that the playbook's discipline transfers to either choice. No directive; the reader decides.

Verification: full test suite (including the docs-contract link/coherence tests covering every touched file) green; README chart/link blobs re-verified on `origin/main` after push.

**The freeze is re-declared as of this addendum's commit: no further changes without a new explicit mandate.**

---

## 10. Post-freeze addendum 3 — macOS-update outage, 2026-07-22 (operational bug fix)

| # | Severity | Finding | Root cause | Fix | Regression test |
|---|----------|---------|------------|-----|-----------------|
| 20 | **HIGH** | After a macOS update + reboot, **all four launchd agents failed to start** — no PID, last exit `78 (EX_CONFIG)`, nothing written to any log. The lab was fully down until manually revived. | The update reset TCC privacy grants. launchd is then **denied opening `StandardOutPath`/`StandardErrorPath` files inside `~/Downloads`** (a TCC-protected folder) when the job's executable is a non-Apple binary (the venv Python) — the job dies at spawn, before the process exists, so nothing can log the reason. Isolated by experiment: the identical job runs perfectly with its launchd log paths outside Downloads; Apple-binary jobs are exempt; in-process file access inside the project was never affected. | `install_service.sh` now writes launchd's own stdout/stderr files to **`~/Library/Logs/tradingbot/`** (the standard, unprotected location); the app's rotating logs stay in `$REPO/logs` unchanged. Plists regenerated, all four services reinstalled and verified healthy (health endpoint all-green, dashboard HTTP 200, Fast Lab retrain ran, companion up). | `test_postfreeze_checkin.py::test_launchd_log_paths_avoid_tcc_protected_folders` |

Known residual (not a defect, flagged for the operator): the project living under `~/Downloads` means any future TCC reset can surface new denials of this class. Moving the project out of a TCC-protected folder (e.g. `~/Trading_Bot_2`) would remove the class entirely — an operational choice for the user, not made unilaterally under the freeze.

**The freeze is re-declared as of this addendum's commit: no further changes without a new explicit mandate.**
