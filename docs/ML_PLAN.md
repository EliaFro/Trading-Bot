# ML Learning System — Plan (approved before any code)

**Date:** 2026-07-10 · **Mode:** paper money only, forever, in this version · **Prime directive:** honest measurement over impressive numbers. A run that concludes "ML didn't help" is a successful run.

---

## 1. The horizon decision (the fee wall, addressed first)

**The ML core operates on the DAILY horizon with a 5-trading-day prediction target, decisions once per day, executed at the next day's open.** Expected turnover: roughly 20–80 round trips per symbol per year — the same fee-safe territory as TSMOM, where Phase 2b measured fees at 1.2–3.5% of gross P&L.

Justification, from our own evidence: Phase 2 proved that at intraday frequency the round-trip cost (0.25–0.30%) exceeds the per-trade gross edge of any signal we could construct (~0.3–0.45%). An intraday ML model would need to be *better than every signal we tested by a factor of ~2* just to break even — and the literature gives no reason to expect that from price-only features. At the daily/5-day horizon, typical move magnitudes (±5–8% for these assets) give a weakly predictive model actual room to pay costs. **The fee reality is also baked directly into the labels** (§3): the model is never even shown a "profitable" example that wasn't profitable after fees.

ML's two roles, both fee-safe:
- **Role A (primary): direction prediction.** Classify the 5-day-forward *executable, after-fee* return as UP / FLAT / DOWN; trade the UP class long-only.
- **Role B (secondary experiment): regime/sizing overlay.** The same feature set predicts whether *TSMOM-60d* will be profitable over the next 5 days; ML gates/sizes the simple strategy rather than generating entries. If Role A fails and Role B works, that itself is a publishable-honest finding: "ML can't pick direction but can size a momentum book."

## 2. Models — and why the deep-learning stubs stay dead

**Random Forest, Gradient Boosting, and L2 Logistic Regression (the linear floor), combined by the honest EnsembleModel slot** (soft vote weighted by each model's *trailing out-of-sample* performance, never in-sample). All three exist in `AdvancedModelTrainer`'s model pool already; tuning uses its `TimeSeriesSplit` path with a small optuna budget (≤25 trials, on training data only).

**No transformer, no TCN, no RL agent.** 36 months of daily bars = ~1,036 samples per symbol, ~3,100 pooled. Deep sequence models on three thousand samples is a guaranteed overfit ritual — the 32 KB stub weights stay retired, and the dashboard will not pretend otherwise. If this system someday ingests years of hourly data with a proven fee-safe hourly edge (it currently has neither), that decision can be revisited. The honest "AI core" for small tabular financial data is gradient-boosted trees; that is what actually works, and saying otherwise would be marketing.

## 3. Labels (no future-peeking beyond the declared horizon)

For each symbol and day T, the label is computed from **executable prices only**:

```
entry  = open[T+1]                    (we decide at T's close; earliest fill is next open)
exit   = open[T+6]                    (5 trading days later, same executability rule)
net    = (exit / entry - 1) - round_trip_cost     (0.30% baseline / 0.25% BNB)
label  = UP if net > +1.0%,  DOWN if net < -1.0%,  else FLAT
```

The ±1% dead zone keeps noise-days out of the training signal. Labels use nothing beyond T+6 — and the **purge rule** (§5) removes the last 6 days of every training window because their labels peek into the test period.

## 4. Features (~60, every one computable at bar T from data ≤ T)

| group | features | source |
|---|---|---|
| momentum | trailing returns 1/3/5/10/20/60/90d (the TSMOM family — deliberately included; if ML just rediscovers momentum, feature importance will show it and we'll say so) | daily bars |
| volatility/risk | realized vol 10/20/60d, ATR%, drawdown-from-90d-high, vol-of-vol | daily bars |
| mean-reversion | RSI(14), distance from SMA50/SMA200 (%), Bollinger position | `src/utils/indicators` |
| volume | volume z-score 20d, volume trend, OBV slope | daily bars |
| cross-asset | ETH/BTC and SOL/BTC relative returns 5/20d, BTC vol (market factor) | daily bars |
| pattern/statistical | trailing-window statistical + fractal groups (Hurst, entropy, streaks) from `AdvancedFeatureExtractor` on 50-day windows | existing extractor |
| calendar | day-of-week, month | index |

**Excluded, with reasons stated:** sentiment (no 36-month history exists — we only started collecting yesterday; it may join the feature set after ~6 months of live accumulation, and will be evaluated with the same OOS standard); order-book/funding data (not yet collected).

Scalers and any feature selection are **fit on the training window only**, applied frozen to test.

## 5. Walk-forward protocol (the same religion, adapted to ML sample needs)

```
train: rolling 400 daily bars  →  purge: drop last 6 train days (labels peek)
→ embargo: 2 days  →  test: 20 days, touched once  →  roll forward 20 days
```

- **~30 OOS windows** on 36 months (the first ~408 days are consumed by the first training window — ML needs more history than a 60-day window can provide; this makes the setup *stricter*, not looser: less OOS data to be lucky in).
- Hyperparameters: tuned per training window via `TimeSeriesSplit` inside the train window only. **Never a shuffled split.**
- **Model selection never touches reported OOS**: the keep-old-unless-better decision uses a validation slice carved from the *end of the training window* (last 60 train days). The reported OOS windows are used exactly once, for reporting. This closes the subtle leak where "pick the best of N models on OOS" quietly overfits the OOS.
- Execution: the identical Phase 2 `simulate()` — marketable LIMIT that lapses on gaps, 0.05% slippage, both fee tiers, long-only, 10% rail and 33% equal-weight sizing variants both reported.
- Benchmarks **recomputed on the ML's exact OOS calendar**: hold-BTC, hold-basket, and TSMOM-60d (the best simple strategy). Apples to apples or nothing.

## 6. Time-series ML trap checklist (confirmed in writing, each with an enforcement mechanism)

| trap | handling | enforced by |
|---|---|---|
| Lookahead in features | every feature at T computed from a window ending at T | unit test: features at T computed on full data == features at T computed on data truncated at T |
| Train/test contamination | rolling window + **6-day purge** + 2-day embargo | window builder asserts `max(train_label_end) < test_start` |
| Labels peeking past horizon | labels fixed at T+1→T+6 open-to-open, nothing later | labeler takes `horizon` as its only future reference; unit test on synthetic data |
| Shuffle leakage in tuning | `TimeSeriesSplit` only, inside train | trainer configured with the existing TimeSeriesSplit path; shuffled splitters not imported |
| Scaler/selector leakage | fit on train, frozen for test | pipeline object per window |
| Selection-on-test leakage | keep-old-unless-better decided on train-tail validation slice, never on reported OOS | retrain loop reads only the validation metric |
| Execution fantasy | same `simulate()` as Phase 2, fees in the labels too | shared code path, no reimplementation |
| Class imbalance games | class weights, and FLAT is never traded | trainer config |

## 7. What the dashboard will show ("watch it learn")

New **ML Lab** tab, all data from real tables (no demo fallbacks, as established):

1. **Learning curve**: OOS accuracy / F1 / after-fee window return per retrain, over time.
2. **Overfitting gauge**: IS-vs-OOS gap per retrain, with plain-language verdicts rendered on the gauge: gap < 5 pts "healthy" · 5–15 pts "watch" · >15 pts **"this model has memorized noise"**.
3. **Feature importance** (RF/GB): top 15, per retrain, with stability across retrains — if the top features churn every retrain, that's noise-fitting and the dashboard says so.
4. **Equity comparison**: ML paper account vs hold-BTC vs TSMOM-60d vs the simple ensemble, one chart.
5. **Retrain log**: timestamp, old-model OOS vs new-model validation, kept/rejected, why.

Backing storage: `ml_predictions` table (new), `model_versions` (existing, metrics JSON extended), retrain events into `model_versions` + alerts.

## 8. Retraining loop (Stage 2)

Weekly (matches the walk-forward step): rebuild features → train candidates → validate on train-tail slice → **replace the live model only if validation improvement > min_improvement (2%, existing config)** → log before/after → the paper engine picks up the new model atomically. Every retrain emits a Telegram/dashboard record. The existing `main.py` retrain scaffolding (interval, min-improvement guard, model_versions persistence) is finally wired to something real.

## 9. Honest prediction, before building (so it's on the record)

The literature on ML for daily price-only prediction is mostly negative after costs: tree ensembles on technical features typically **match, not beat, simple momentum** — because their top features turn out to *be* momentum, rediscovered with extra variance. My honest priors:

- **~65% probability: the ML ensemble does NOT beat TSMOM-60d out-of-sample after fees.** Most likely outcome: comparable returns, higher turnover, feature importance dominated by the 20–90d momentum features (i.e., an expensive TSMOM).
- **~20%: modest genuine value-add**, most plausibly from volatility/cross-asset features cutting exposure in high-vol regimes (Role B has better odds than Role A).
- **~15%: clearly worse than TSMOM** (overfitting the 2024 bull's specific texture despite every guard).
- **Beating hold-BTC on raw return over this mostly-bull span: unlikely** (TSMOM captured less than half of it); on risk-adjusted terms: plausible.

If the measured outcome is the 65% case, the deliverable is a dashboard that *shows* that, in public, per retrain — which is exactly the learning system requested.

## 10. Build stages

| stage | deliverable | proof |
|---|---|---|
| **1. Offline milestone** | feature builder + labeler + walk-forward ML evaluation on 36 months; `docs/ML_RESULTS.md` with the verdict: does ML beat (a) hold-BTC, (b) TSMOM-60d, OOS after fees — plus IS-vs-OOS gaps and feature importance | study output, both fee tiers, both sizing variants |
| 2. Live retraining loop | weekly retrain in the paper bot, keep-old-unless-better end-to-end, retrain log | forced-retrain test + log inspection |
| 3. Dashboard ML Lab | the five panels above, real data only | screenshots against live DB |
| 4. 24/7 paper deployment | Docker, monitors, alerts include retrain events | uptime + alert test |

Stage 1 is the go/no-go: if the offline verdict is "ML adds nothing," stages 2–4 still get built (the point is watching honest learning), but the dashboard's default verdict banner will say what the data says.
