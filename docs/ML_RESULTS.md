# ML Learning Core — Stage 1 Walk-Forward Results

## VERDICT: The ML ensemble does NOT beat TSMOM-60d out-of-sample after fees (−2.8% vs +0.4% rail-sized; −9.8% vs +1.1% equal-weight), and its win over hold-BTC (−2.8% vs −33.0%) is regime luck, not skill. ML lost money.

**Date:** 2026-07-10 · **Protocol:** docs/ML_PLAN.md, all traps enforced by tests (7/7 passing, including the no-skill-on-random-noise leak test) · **OOS:** 24 windows, 2025-03-06 → 2026-06-29 (13 sideways / 6 bear / 5 bull — the 400-day training requirement consumes the 2023–24 bull, so the ML is tested mostly on the hard part of the cycle; the TSMOM comparison uses the *identical* calendar, so the comparison is fair even though the span is unluckily composed).

---

## 1. Trading results (portfolio across 3 symbols, identical execution)

| strategy | sizing | fee | OOS return | PF | maxDD | trades | fees % of gross |
|---|---|---|---|---|---|---|---|
| **ML ensemble** | rail 10% | 0.10% | **−2.8%** | 0.73 | −5.7% | 95 | ~8% |
| **ML ensemble** | equal-wt 33% | 0.10% | **−9.8%** | 0.71 | −18.3% | 95 | ~9% |
| TSMOM-60d (same calendar) | rail 10% | 0.10% | +0.4% | 1.15 | −4.1% | ~30 | ~3% |
| TSMOM-60d (same calendar) | equal-wt 33% | 0.10% | +1.1% | 1.12 | −13.9% | ~30 | ~3% |
| hold BTC | — | — | −33.0% | — | — | — | — |
| hold basket | — | — | −35.8% | — | −64.5% | — | — |

BNB fee tier changes nothing (ML −9.3% vs −9.8%): fees are not what killed it — **prediction quality is**. The ML traded ~3× as often as TSMOM and was wrong more often than right after the ±1% dead zone.

## 2. The overfitting gauge — exactly what it was built to show

Mean across 24 retrains, balanced accuracy (3-class chance = 0.333):

| model | in-sample | out-of-sample | **IS−OOS gap** | plain language |
|---|---|---|---|---|
| gradient_boosting | **0.921** | 0.357 | **+0.564** | **this model has memorized noise** |
| random_forest | 0.748 | 0.376 | +0.372 | mostly memorized noise |
| ensemble | 0.801 | 0.390 | +0.411 | inherits the trees' memorization |
| logreg (linear floor) | 0.538 | **0.392** | +0.146 | least overfit — and the **best** OOS |

The most instructive line in this whole study: **the simplest model won out-of-sample.** Gradient boosting looked spectacular in training (92%!) and was the *worst* live. This is the exact pattern the dashboard's overfitting gauge exists to make visible, and it showed up in run one.

Also note: OOS 0.390 is *slightly above chance* — the models aren't pure noise — but "slightly above chance at classification" converted into **negative money after costs**. Predictive ≠ profitable is the second lesson of this study.

## 3. Feature importance — what the models actually leaned on (your ask #2)

**My 65% prediction was wrong in the specific mechanism.** The models did *not* primarily rediscover momentum:

| feature group | share of importance |
|---|---|
| **volatility** | **20.8%** |
| **cross-asset (BTC market factor)** | **20.1%** |
| statistical (skew/kurt/autocorr/Hurst) | 15.8% |
| momentum | 15.4% |
| mean-reversion | 12.3% |
| volume | 10.1% |
| calendar | 5.4% |

Top individual features: `btc_rv_20` (BTC 20-day volatility — #1 by a wide margin), `btc_ret_20`, `kurt_20`, `month`, `vol_of_vol`. The models learned "what regime are we in" (volatility & market-factor structure), not "which way is price going."

**Stability — the thing you wanted to learn to recognize:** rank correlation of feature importances across consecutive retrains = **0.91 mean, 0.82 minimum**. The models look at the *same* features every retrain — this is *not* random churn. Which teaches the subtler lesson: **stable importance is necessary but not sufficient.** Exhibit A: `month` is the #4 feature and is stably ranked — yet with 3 years of data, each calendar month has been seen ~3 times. A feature can be *consistently* fitted and still be noise; the tell isn't instability, it's asking "how many independent observations back this pattern?" Three Julys is not seasonality, it's a coincidence with good posture.

Reading the two signals together: the models found real regime *structure* (vol state — genuinely stable, economically sensible) but no directional *edge* (which is why classification hovers barely above chance and trading loses). That decomposition — structure without edge — is the most common honest outcome in financial ML, and now you've seen it in your own data.

## 4. Per-regime classification (ensemble OOS balanced accuracy)

| regime | windows | OOS bal-acc |
|---|---|---|
| sideways | 13 | 0.412 |
| bear | 6 | 0.369 |
| bull | 5 | 0.360 |

Marginally best in sideways markets, near-chance in trends — consistent with the models keying on volatility structure rather than direction.

## 5. Honest scorecard vs the plan's predictions

- "~65%: ML doesn't beat TSMOM" → **happened** (though via vol-features, not momentum-rediscovery — prediction right, mechanism wrong).
- "In-sample accuracy is worthless and must not be celebrated" → demonstrated: 92% IS collapsed to 36% OOS.
- Gate check (unchanged): PF 0.73 vs >1.15 ✗ · maxDD ✓(rail)/✗(equal-wt) · 95 trades vs ≥200 ✗ · positive windows below 60% ✗. **FAIL** — stated without cushioning.
- The pipeline itself is validated: leak tests pass, no-skill-on-noise passes, the same machinery is ready for the live learning loop.

## 6. What Stage 2–4 will watch (this result is the system working, not failing)

The learning system now goes live on paper with this exact machinery: weekly retrains, keep-old-unless-better on the validation slice, and the dashboard ML Lab showing the learning curve, the IS-OOS gap gauge, and feature-importance stability in real time. The current model earns the banner: **"ML has not demonstrated an edge over simple momentum — status: learning."** If six months of live sentiment history, or different feature families, ever change that verdict, the dashboard will show it — by the same rules that produced this report.

*Reproduce: `python scripts/run_ml_study.py` · Raw numbers: `docs/ml_metrics.json` · Chart: `docs/phase2_charts/ml_study.png` (equity vs benchmarks · learning curve with gap band · importance with error bars).*
