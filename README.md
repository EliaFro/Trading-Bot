# AI Crypto Trading System — Project Archive

**Status (2026-07-10): wound down by decision, cleanly.** The active plan is
manual DCA with a monthly trend rule — see **[PLAYBOOK.md](PLAYBOOK.md)**.
This repository is preserved intact for a possible future deployment at
meaningful capital.

## What this project proved (the short version)

Three years of verified Binance data, an honest walk-forward harness, five
hypothesis-driven strategy iterations, two fee tiers:

- **Intraday strategies (5m/15m RSI / MA-cross / breakout / ensemble): dead.**
  Gross edge ~0.3–0.45% per trade against 0.25–0.30% round-trip costs; every
  configuration net-negative in every regime. Full autopsy:
  [docs/PHASE2_RESULTS.md](docs/PHASE2_RESULTS.md)
- **Daily trend-following (TSMOM, weekly rebalance): profitable in backtest**
  (PF 2.0–3.7) **but failed the gate** — sample too thin to prove (24–56
  trades, t ≤ 1.67), drawdown 21–26% at deployable size, and at $100–500 the
  dollar edge is smaller than hosting costs. Full study:
  [docs/DAILY_MOMENTUM_RESULTS.md](docs/DAILY_MOMENTUM_RESULTS.md)
- **The infrastructure works**: gap-free data pipeline, realistic paper
  engine (fills verified to the cent), hard risk rails, kill switch,
  monitoring, alerting, dashboard, 44-test suite. It was never the problem.

## How to resume (the $2k+ path)

If the account is ever funded to **$2,000+** (the economic floor computed in
DAILY_MOMENTUM_RESULTS.md §5), the tested restart path is:

1. Read the verdict first: [docs/DAILY_MOMENTUM_RESULTS.md](docs/DAILY_MOMENTUM_RESULTS.md)
   — especially §4 (why the evidence is the literature, not this backtest)
   and §3 (expect −20–26% drawdowns, not smooth gains).
2. The frozen strategy profile is [config/tsmom_frozen.yaml](config/tsmom_frozen.yaml).
   Do **not** re-tune it; re-*validate* it on fresh data:
   ```bash
   make install && make db-init
   python scripts/backfill.py --months 36
   python scripts/verify_coverage.py --months 36
   python scripts/run_daily_momentum.py        # numbers should rhyme with the report
   ```
3. Wire the TSMOM profile into the engine (daily bars + weekly cycle;
   the equal-weight variant requires consciously raising the 10% position
   rail in `src/trading/safety.py` — it is a hard ceiling by design).
4. Run the **full 14-day paper gate** (Phase 3) before any live key touches
   the engine; Phase 4's live rails (withdrawal-permission refusal, kill
   switch, reconciliation) are already built and tested.

## Running what exists

```bash
make help          # all commands
make run           # paper-trading bot (health: :8080, metrics: :9100)
make dashboard     # Streamlit dashboard on :8501
make test          # 47-test suite
make docker-up     # containerized deployment (auto-restart, log rotation)
```

Secrets live only in `.env` (gitignored, chmod 600, template in
`.env.example`). Market data, logs, models, and the quarantined legacy code
are all excluded from version control.

## Map

| Path | What |
|---|---|
| `PLAYBOOK.md` | The active plan (manual, 10 min/month) |
| `docs/ARCHITECTURE_REPORT.md` | Phase 0 audit of the original codebase |
| `docs/PHASE2_RESULTS.md` | Intraday strategy validation (verdict: dead) |
| `docs/DAILY_MOMENTUM_RESULTS.md` | TSMOM study (verdict: FAIL gate; conditional promise at $2k+) |
| `config/tsmom_frozen.yaml` | Frozen TSMOM profile for a future restart |
| `src/` | Engine, strategies, backtester, monitoring, dashboard |
| `scripts/` | init_db, backfill (gap-repairing), verify_coverage, walk-forward studies |
| `tests/` | Fill economics, safety rails, DB, indicators, notifier |
| `_quarantine/` | Dead legacy files (preserved, excluded from git) |
