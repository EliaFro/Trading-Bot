# Operating the ML Learning Lab

*Paper money only. The banner says "learning," not "earning" — this document
tells you how to run it, how to read it, and the exact bar that would have to
be cleared before that banner could ever honestly change.*

---

## 1. Start / stop / check (native macOS service — no Docker)

The lab runs as two launchd agents: they **start at login and auto-restart
on crashes** (verified with a kill -9 test — positions survive hard crashes
because state lives in the database).

```bash
cd ~/Downloads/Trading_Bot_2

make service-install     # install + start both agents (bot + dashboard)
make service-uninstall   # stop + remove both agents
make service-status      # are they alive?
make status              # one-line engine status (mode, equity, positions)
tail -f logs/trading.log # live bot log (Ctrl+C to stop watching)
```

- Dashboard: **http://localhost:8501** (🧪 ML Lab is the second tab)
- Health check: **http://localhost:8080/health**
- A **graceful** stop (`make service-uninstall`) closes open paper positions
  by design; a crash does not — either way the ML re-establishes its
  positions on its next daily pass. Nothing is lost across restarts.
- **Never run `make run` (native foreground) while the service is
  installed** — two bots writing one database.
- Alerts (retrains, drawdown >8%, errors, daily P&L) go to Telegram if
  `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are set in `.env`; they always land
  in the dashboard and the `alerts` table regardless.
- The Mac must be awake for the bot to act. If it sleeps overnight, the bot
  simply catches up at the next wake — daily decisions tolerate this. To
  keep it always-on: System Settings → prevent sleep on power adapter.

## 2. Reading the ML Lab tab, panel by panel

**The banner** — the verdict of record. It cites the Stage 1 numbers
(ML −2.8% vs TSMOM +0.4%, after fees, same calendar). Nothing below the
banner can override it; only §3's bar can change it.

**Overfitting gauge** (in-sample vs validation accuracy of the current champion)
- *Good reading:* small gap (< 5 points) with validation meaningfully above
  33.3% (three-class chance). Rare and precious.
- *Bad reading:* big in-sample number, chance-level validation — the red
  "memorized noise" verdict. **A high in-sample score is never good news.**
- Expect bad readings. The current champion reads IS 86% / val 36% —
  gap +50 points. The gauge exists to keep saying that out loud.

**Learning curve** (validation accuracy per retrain, over weeks)
- *Good:* the green validation line drifting upward, above the 33.3% chance
  line, across many retrains.
- *Bad (and normal):* it wobbles around chance forever. Wobble is not
  learning; direction sustained over months is.

**Feature importance + evidence tiers** — the "stable ≠ meaningful" panel.
- *Good:* importance concentrated in 🟢/🟠 features (well-supported/moderate,
  e.g. `btc_rv_20` — dozens-to-hundreds of independent observations).
- *Bad:* weight on 🔴 thin-evidence features (`month` = ~3 samples in 3 years).
  The red callout names them. If the model keeps leaning on 🔴 features,
  it is fitting coincidences — no matter how *stable* their ranking is.
- What to watch over weeks: does weight migrate toward better-evidenced
  features as data accumulates? That would be genuine hygiene improvement
  (still not an edge by itself).

**Equity vs benchmarks** — ML paper account against hold-BTC and cash.
- *Good:* beating hold-BTC over a long window **including a bull stretch**.
- *Trap to avoid:* "beating" BTC during a downturn by sitting in cash is
  regime luck (Stage 1's −2.8% "beat" −33% this way). Judge only across a
  full cycle.

**Retrain log** — the guard, audited. Most rows should say **KEPT_OLD**:
challengers usually aren't genuinely better, and rejecting them is the guard
working. A log full of REPLACED rows with flat validation scores would mean
churn, not progress.

## 3. The exact bar for the banner to change ("learning" → "demonstrated an edge")

All five conditions, measured on **live paper decisions only** (logged in
`ml_predictions`/`trades` — no backtests, no retro-fitting), after fees:

1. **Sample**: ≥ 100 completed live ML trades (~2–3 years at this cadence).
   No shortcuts by counting backtest trades.
2. **Profitability**: profit factor > 1.15 after fees AND max drawdown < 15%
   over the whole live record — the unchanged Phase 2 gate.
3. **Beats the simple alternative**: higher after-fee return than TSMOM-60d
   computed on the *identical* live period (the dashboard's benchmark line).
   Beating cash or a falling BTC does not count.
4. **Statistical meaning**: the live 20-day-window returns have a t-statistic
   ≥ 2.0 (so the record is unlikely to be luck), and validation accuracy
   across retrains averages ≥ 5 points above chance.
5. **No thin-evidence dependence**: 🔴-tier features contribute < 10% of
   ensemble importance (an edge built on `month` is not an edge).

Anything less — a good month, a narrowing gap, a nice-looking stretch of the
equity line — is **activity, not progress**. The honest expectation, stated
in docs/ML_PLAN.md before any code was written and confirmed by Stage 1, is
that this bar is probably never cleared. The lab's product is watching a
learning process measured truthfully, and the guard, the gauge, and the
evidence tiers are there so that if nothing real ever emerges, the system
says so every single week.

## 3b. Fast Lab kill criteria (pre-registered 2026-07-10, immutable)

The ⚡ Fast Lab (1m/5m paper bot) is a **learning accelerator, never a profit
path** — its banner states the Phase 2 fee-wall verdict from day one. Its
strategy-search role ends **permanently** if, after the offline studies
(docs/FASTLAB_PLAN.md) plus the live-paper window ending **2026-08-07**,
nothing has cleared the multiple-testing-corrected significance bar after
full costs (fees + spread + slippage). After that date it may keep running
solely as an observation instrument for watching the ML learn — no further
strategy iterations at this horizon, ever. Same standing rules: no
gate-loosening, no cherry-picking after results, noise control is the
baseline, in-sample numbers are never celebrated.

## 4. Maintenance (rarely needed)

```bash
make test                                  # full test suite
python scripts/verify_coverage.py          # data integrity check
python scripts/clear_kill_switch.py        # only after reading why it fired
tail -n 200 logs/trading.log               # recent bot log
```

**⚠️ 2027-01-05 — MANUAL STEP, put it on your calendar:** the one-time
sentiment evaluation does **not** run itself. On or after that date, run
`python scripts/run_sentiment_eval.py` by hand (it refuses to run earlier).
Every other scheduled event — daily checks, monthly digest, the kill rule —
is self-executing; this is the only one that waits for a human.

The databases, logs, and models live in `./data`, `./logs`, `./models` on
your machine; the launchd services survive restarts without losing state.
(Docker files remain in the repo for reference but the supported deployment
is launchd.) The money plan remains [PLAYBOOK.md](PLAYBOOK.md) — this lab
never touches it.
