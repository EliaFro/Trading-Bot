# Fast Lab — Offline Study Results (Parts B & C)

## VERDICT: Nothing survives costs at 1m/5m — but Part C found something real: the ML models genuinely predict 30-minute direction (+11 points above chance, every window, every regime, deep and shallow alike), and that authentic skill is worth ~1 basis point per trade against a 31-basis-point toll. Real edge, unharvestable at this frequency. The strategy-search kill clock is running (2026-08-07).

**Pre-registered design and predictions:** docs/FASTLAB_PLAN.md (locked before any run). **Execution:** measured spreads (BTC one tick; conservative floors 0.01–0.02% applied), 0.10% fee + 0.05% slippage per side, next-bar fills that lapse on gaps. Every rule/dataset lookahead-tested (128 tests passing).

---

## Part B — Elder triple-screen (multi-timeframe), N=8 variants

| variant | OOS return | PF | trades | gross/trade | cost/trade | t |
|---|---|---|---|---|---|---|
| ema26+stoch 5m (best) | −88.5% | 0.15 | 21,074 | +0.004% | 0.313% | −25.0 |
| macdh+stoch 5m | −87.5% | 0.16 | 20,470 | +0.008% | 0.313% | −50.0 |
| all 1m variants | −99.0% | ≤0.03 | ~45,000 | ±0.007% | 0.313% | −3.4 to −6.0 |
| all others | −98% | ≤0.04 | 37–44k | ≈0% | 0.313% | ≤−22 |

- Naive p<0.05: **0/8** · Bonferroni: **0** · Reality Check best t = **−3.36** (p=1.0).
- Noise control (100 random, matched frequency): 0/100 significant; best random t = **−45** — at 20–45k trades/year *everything* is annihilated by costs; luck can't even look good.
- **Ceiling check (the point of Part B §5):** gross edge per trade −0.005% to +0.008% vs 0.313% round trip — a ~40× shortfall. **No filter, sizing rule, or oscillator can bridge that; the strategy family is arithmetic-dead at this frequency**, exactly as Phase 2 predicted and the plan pre-registered (predicted gross 0.05–0.20%; reality was worse).

## Part C — the deep models' fair trial (15 stratified windows, 36 months, ~50k train rows/window)

**Classification — real skill, cleanly measured** (balanced accuracy, 3-class chance = 0.333):

| model | IS | OOS | gap | reading |
|---|---|---|---|---|
| random_forest | 0.494 | **0.451** | +0.043 | best OOS — trees win again |
| lstm | 0.464 | 0.442 | +0.022 | healthy, matches trees |
| ensemble | 0.464 | 0.443 | +0.021 | healthy |
| cnn | 0.469 | 0.440 | +0.029 | healthy |
| logreg | 0.454 | 0.432 | +0.023 | linear floor close behind |
| gradient_boosting | 0.347 | 0.338 | +0.009 | underfit at this scale |

Three findings, stated plainly:
1. **The skill is real.** +10–12 points above chance out-of-sample, in all 15 windows (min 0.414), across bull/bear/sideways, with tens of thousands of test decisions — this is not luck and not leakage (the pipeline passes the random-walk no-skill test). Pre-registered P(real classification skill) ≈ 50% → **happened**. Short-horizon price movement is partially predictable.
2. **Deep learning got its fair trial and did not beat the trees.** LSTM/CNN ≈ 0.44 vs RF 0.45 at ~50k samples/window. Pre-registered P(deep beats trees) ≈ 35% → **did not happen**. The overfitting gauges are the healthiest of the whole project (gaps +0.01–0.04) — with enough data, nothing memorizes; the models are honest and still can't beat a forest.
3. **And none of it survives the toll booth.** Trading the predictions with full costs:

| family | trades | gross/trade | cost/trade | net/trade | PF | +windows | t/trade |
|---|---|---|---|---|---|---|---|
| ensemble | 8,975 | **+0.012%** | 0.314% | −0.302% | 0.22 | 0/15 | −37 |
| best tree | 7,904 | +0.009% | 0.315% | −0.306% | 0.23 | 0/15 | −35 |
| best deep | 10,808 | +0.009% | 0.314% | −0.305% | 0.21 | 0/15 | −43 |

Pre-registered P(anything beats full costs) ≈ 3% → **0%, confirmed**. The models' genuine predictive edge converts to ~+1.2bp of gross per trade; the round trip costs 31bp. **A 25× shortfall that no model quality can fix — the edge would need to be twenty-five times larger, not slightly better.**

## Prediction scorecard (accountability)

| pre-registered prediction | probability given | outcome |
|---|---|---|
| any MTF variant clears corrected bar | ~5% | ✗ none (0/8) |
| any MTF gross edge > cost | ~15% | ✗ none (max +0.008% vs 0.313%) |
| deep models beat trees OOS | ~35% | ✗ RF won |
| real classification skill above chance | ~50% | ✓ +11 pts, every window |
| anything beats full costs | ~3% | ✗ none |

## What this buys the learning lab

The Fast Lab's observation role now has something genuinely worth watching: a model with *measurable, honest, regime-robust* skill operating in public against a cost wall it cannot beat — the cleanest live demonstration possible of "predictive ≠ profitable." The dashboard's cost decomposition shows the ~31bp toll consuming the ~1bp edge on every trade, in real time. Per the pre-registered kill criteria: unless something clears the corrected bar after full costs by **2026-08-07** (nothing is on track to), the strategy-search role at this horizon ends permanently and the Fast Lab remains an observation instrument only.

*Disclosed limitation of the live record: it runs on a laptop, so it samples only the hours the machine is awake (measured July 2026: daily service checks ran 5–13 h behind schedule) — a market-hours selection bias the offline studies, which saw every bar, did not have.*

*Reproduce: `python scripts/run_fastlab_study.py` (Part B), `python scripts/run_ml_fast_study.py` (Part C) · Raw: `docs/fastlab_partB_metrics.json`, `docs/fastlab_partC_metrics.json`, `docs/spread_measurements.json` · Chart: `docs/phase2_charts/fastlab_partB.png`.*
