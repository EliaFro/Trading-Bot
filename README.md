# A random strategy beat the Turtles.

**I built an honest evaluation harness and tested 29 famous trading strategies, machine learning at two scales, and every retail-tradeable horizon on 36 months of crypto data. Nothing survived multiple-testing correction. A coin-flip strategy outperformed ninety years of trading literature. This repository is the method, the evidence, and the running instrument.**

*(Everything here runs on paper money. The only real-money artifact is [PLAYBOOK.md](PLAYBOOK.md) — a 10-minute-a-month manual plan that deliberately contains no cleverness.)*

---

## The question

Can a retail account ($100–$10k, standard exchange fees) systematically day-trade crypto profitably — using published technical strategies, or machine learning, at any horizon from 1 minute to weekly?

## The harness (the actual contribution)

Every strategy and model faced the identical, pre-registered evaluation:

- **Walk-forward only** — train on the past, test once on the never-seen future, roll forward. In-sample results are never reported as findings. ([docs/ML_PLAN.md](docs/ML_PLAN.md))
- **Purge + embargo** — labels that peek past a training boundary are purged; an embargo gap prevents adjacency leakage.
- **Anti-lookahead as executable tests** — every feature and signal must be *truncation-invariant* (identical at time T whether or not the future exists in the data); the full ML pipeline must find **no skill on random walks**. 140+ tests enforce this.
- **Honest execution** — next-bar fills through a LIMIT model that lapses on gaps, 0.10%/side fees (0.075% sensitivity), 0.05% slippage, **measured** bid-ask spreads with conservative floors ([docs/spread_measurements.json](docs/spread_measurements.json)), and a per-trade cost decomposition: gross − fees − spread − slippage = net.
- **Multiple-testing correction** — N is counted; naive p-values face Bonferroni and a White's Reality Check bootstrap; and a **100-random-strategy noise control** runs through the same pipeline as the luck baseline for every claim.
- **Pre-registration** — predictions with probabilities were written down before each study ran, and scored after.

## The findings

| study | result | evidence |
|---|---|---|
| Intraday classics (RSI/MA/breakout, 5m–15m, walk-forward, 36 mo) | Gross edge ~0.4%/trade vs 0.3% round-trip cost → **every configuration net-negative in every regime** | [docs/PHASE2_RESULTS.md](docs/PHASE2_RESULTS.md) |
| **29 published strategies** (Wilder, Bollinger, Ichimoku, Turtles, TSMOM…) | 24 profitable-looking; **2 naive-significant vs 1.5 expected by luck; 0 survive Bonferroni; Reality Check p=0.15; the best of 100 random strategies (t=2.49) beat the best published one (t=2.18)** | [docs/SIGNAL_LIBRARY_RESULTS.md](docs/SIGNAL_LIBRARY_RESULTS.md) |
| Daily trend (TSMOM), the strongest family | Profitable (PF 2–3.7) but statistically unprovable in 3 years (t≤1.67), −21–26% drawdowns at deployable size, and at small accounts the edge is smaller than hosting costs | [docs/DAILY_MOMENTUM_RESULTS.md](docs/DAILY_MOMENTUM_RESULTS.md) |
| ML, daily horizon (trees + linear, 3k samples) | The 92%-in-sample model was worst out-of-sample (36%); the linear floor won; **ML lost to simple momentum** | [docs/ML_RESULTS.md](docs/ML_RESULTS.md) |
| ML, 1-minute horizon (LSTM/CNN fair trial, ~1M samples) | **Genuine skill: +11 points above chance, every window, every regime** — worth ~1bp/trade against a 31bp cost. Deep ≈ trees. **Predictive ≠ profitable, demonstrated with real prediction** | [docs/FASTLAB_RESULTS.md](docs/FASTLAB_RESULTS.md) |
| Multi-timeframe (Elder triple-screen, 8 variants) | −87% to −99%; gross/trade ≤0.008% vs 0.313% cost — a 40× shortfall no filter can fix | [docs/FASTLAB_RESULTS.md](docs/FASTLAB_RESULTS.md) |

Key charts: `docs/phase2_charts/signal_library.png` (the published-vs-noise t-distribution overlap — the whole finding in one image), `docs/phase2_charts/fastlab_partB.png` (gross edge vs cost per trade), `docs/phase2_charts/ml_study.png` (the overfitting gauge in action).

## The lessons (each one measured, not asserted)

1. **Predictive ≠ profitable.** 1-minute crypto is genuinely predictable (+11 pts over chance, robust) and genuinely untradeable (the skill is worth 1/25th of the toll).
2. **The fee wall explains retail trading.** ~97% of the intraday round-trip cost is fees+slippage, not spread. Below ~0.1% per-trade gross edge, strategy quality is irrelevant.
3. **Stable ≠ meaningful.** A feature can rank high in every retrain and rest on 3 observations (`month`, over 3 years). Count independent observations, not consistency.
4. **Best-of-N is luck's favorite costume.** Test 29 things and ~1.5 pass at p<0.05 for free; 9 of 100 coin-flip strategies "passed" too. Corrections aren't pedantry — they're the difference between a finding and a story.
5. **In-sample accuracy is an anti-signal.** The prettiest training numbers came from the worst live models, every time.

## The running instrument

The repo ships a live paper-trading observatory (four macOS launchd services — see [OPERATING.md](OPERATING.md)): a daily ML lab and a 1m "Fast Lab" that retrain on schedule behind a keep-old-unless-better guard, a dashboard that displays the overfitting gauge, evidence tiers, and the live fee decomposition, monthly auto-generated evidence digests, and a **self-executing pre-registered kill rule** ([src/trading/kill_rule.py](src/trading/kill_rule.py)) that permanently closes strategy search at the fast horizon on 2026-08-07 unless something clears the corrected bar (nothing is on track to).

## Reproduce

```bash
make install && make db-init
python scripts/backfill.py --months 36      # ~4M candles from Binance public API
python scripts/verify_coverage.py           # gap-free proof
make test                                   # 140+ tests incl. anti-lookahead suite
python scripts/run_signal_library.py        # the 29-strategy study + noise control
python scripts/run_daily_momentum.py        # TSMOM
python scripts/run_ml_study.py              # daily ML
python scripts/run_fastlab_study.py         # multi-timeframe (kill-rule-gated)
python scripts/run_ml_fast_study.py         # deep ML at 1m (kill-rule-gated)
```

Scope for contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) — this is an evaluation harness and a finding; strategy submissions go through the corrected gate or not at all. License: [MIT](LICENSE).
