# Phase 2 — Walk-Forward Validation Results (12-month + 36-month studies)

**Updated:** 2026-07-10 · **Data:** Binance spot OHLCV via public API, stored in SQLite, coverage verified gap-free.

---

# VERDICT: NO CONFIGURATION PASSES — AT EITHER FEE LEVEL — ON 36 MONTHS SPANNING ALL THREE REGIMES.

**This strategy family (intraday RSI mean-reversion / MA crossover / breakout / their ensemble, long-only spot, 5m–15m) is dead for the purpose of this project.** Not "needs more tuning" — dead, with a mathematical argument below for why no further filter iteration can save it. The 3-iteration budget goes unspent: spending it would have been curve-fitting theater, and the rules of this study exist precisely to prevent that.

Live trading remains locked. Phase 3 infrastructure (Docker, health checks, alerting) is validated separately as infrastructure-only.

---

## 1. Data coverage (36 months, verified)

All 12 symbol/timeframe combinations: **100.00% complete, zero interior gaps** (`scripts/verify_coverage.py`; two interrupted-backfill holes in BTC 5m of 274 and 145 days were detected and repaired — gap-repair is now a permanent part of `scripts/backfill.py`).

| symbol | 1m | 5m | 15m | 1h | range |
|---|---|---|---|---|---|
| BTC/USDT | 1,581,142 | 316,226 | 105,407 | 26,351 | 2023-07-07 → 2026-07-10 |
| ETH/USDT | 1,581,129 | 316,223 | 105,407 | 26,351 | 2023-07-08 → 2026-07-10 |
| SOL/USDT | 1,581,129 | 316,224 | 105,407 | 26,351 | 2023-07-08 → 2026-07-10 |

The period contains a full cycle: the 2023-Q4→2025-Q1 bull (BTC ~$30k→$111k), the 2025 top, and the 2025-26 bear (→~$63k). Regime mix of the 51 OOS windows: **21 bull / 17 sideways / 13 bear** (objective per-window classification: BTC 20-day return > +5% bull, < −5% bear).

## 2. Methodology (identical to the 12-month study — nothing loosened)

Train 60d → embargo 2d → test 20d, rolling 20d → **51 OOS windows** (~2.8 years OOS). Grid selection on train windows only (after-fee PF, ≥15-trade floor; untradeable windows sit out). Execution identical to the paper engine and conservative on every ambiguity (next-bar marketable LIMIT entries that lapse if the bar never trades through; 0.05% slippage; SL beats TP intrabar; gaps fill at the worse open; 0.55 confidence gate; 10% position cap; $10 min notional; long-only).

**Fee sensitivity:** the entire study ran twice — 0.10%/side (taker baseline) and 0.075%/side (BNB discount). Fee-rate change only; identical fill logic, no assumed maker fills. Gate unchanged: OOS PF > 1.15 after fees, max DD < 15%, ≥200 OOS trades, ≥60% positive windows. A config passing only at 0.075% would be marked CONDITIONAL PASS. None did.

Configuration under test: the surviving Phase 2 config family (iteration-2 flags: calibrated confidence + fixed-fractional risk sizing + directional regime gate). Strategy logic frozen — zero new iterations were run on the 36-month data (see §5).

## 3. Results — every configuration, both fees, one-line verdicts

`+win` = positive walk-forward windows out of 51. PORTFOLIO = the strategy across all 3 symbols, equal weight (the actual deployment unit).

| configuration | ret @0.10% | PF @0.10% | PF @0.075% | maxDD | trades | +win | **verdict** |
|---|---|---|---|---|---|---|---|
| rsi_mean_reversion PORTFOLIO 15m | −5.7% | 0.77 | 0.84 | −7.2% | 1,124 | 15/51 | **FAIL** |
| rsi_mean_reversion SOL 15m *(best single)* | −4.3% | 0.86 | 0.92 | −7.5% | 407 | 22/51 | **FAIL** |
| rsi_mean_reversion BTC 15m | −4.6% | 0.74 | 0.83 | −6.1% | 343 | 18/51 | **FAIL** |
| rsi_mean_reversion ETH 15m | −8.2% | 0.70 | 0.75 | −9.1% | 374 | 15/51 | **FAIL** |
| rsi_mean_reversion PORTFOLIO 5m | −19.1% | 0.56 | 0.63 | −19.4% | 2,825 | 7/51 | **FAIL** |
| ma_crossover PORTFOLIO 5m | −6.8% | 0.59 | 0.63 | −6.9% | 710 | 13/51 | **FAIL** |
| ma_crossover 15m (all symbols) | 0.0% | — | — | 0.0% | **0** | 0/51 | **FAIL** (never trades: cross+volume+ADX+uptrend never coincide in 3 years) |
| breakout PORTFOLIO 15m | −15.6% | 0.65 | 0.68 | −16.5% | 1,442 | 6/51 | **FAIL** |
| breakout PORTFOLIO 5m | −34.8% | 0.54 | 0.58 | −34.9% | 3,944 | 2/51 | **FAIL** |
| ensemble PORTFOLIO 15m | −19.4% | 0.46 | 0.52 | −19.5% | 2,393 | 1/51 | **FAIL** |
| ensemble PORTFOLIO 5m | −44.4% | 0.31 | 0.37 | −44.4% | 5,986 | 0/51 | **FAIL** |

(Every remaining per-symbol row also FAILS — complete tables in `docs/phase2_metrics_m36_taker.json` / `_bnb.json`; per-window tables with regime labels in `docs/phase2_windows_m36_taker.md` / `_bnb.md`; equity charts in `docs/phase2_charts/m36_*.png`.)

**No PASS. No CONDITIONAL PASS.** The BNB discount improves every PF by 0.03–0.07 — real money, nowhere near enough.

## 4. Regime breakdown — where the edge lives and dies

RSI mean-reversion 15m (the family's best), at the **discounted** 0.075% fee:

| symbol | regime | windows | mean window ret | +win% | PF | trades |
|---|---|---|---|---|---|---|
| BTC | bull | 21 | +0.04% | 52% | **1.12** | 152 |
| BTC | sideways | 17 | −0.03% | 35% | 0.90 | 107 |
| BTC | bear | 13 | −0.24% | 15% | 0.49 | 84 |
| ETH | bull | 21 | −0.03% | 52% | 0.94 | 183 |
| ETH | sideways | 17 | −0.27% | 18% | 0.53 | 112 |
| ETH | bear | 13 | −0.12% | 46% | 0.75 | 79 |
| SOL | bull | 21 | +0.12% | 71% | **1.18** | 209 |
| SOL | sideways | 17 | −0.09% | 41% | 0.84 | 115 |
| SOL | bear | 13 | −0.26% | 15% | 0.59 | 80 |

The same pattern holds for every other strategy (all regime tables in the JSON artifacts): **the edge "lives" only in bull windows, and even there it averages PF ≈ 1.08 portfolio-wide — below the 1.15 gate.** Sideways and bear are uniformly negative.

## 5. Why no further iteration can save this family (the futility bound)

A regime filter — any filter — can only *select a subset* of trading windows. The table above is the **hindsight-optimal** subset: it uses each window's actual future BTC return, information no causal filter can have. Even this cheating filter yields PF ≈ 1.08 at reduced fees, on only 21 of 51 windows. Any implementable (causal) filter is bounded above by it and will do worse — our SMA200 gate already approximates it. Therefore:

* **Filter-type iterations cannot reach PF 1.15.** The supremum is ~1.08.
* **Per-trade-economics iterations (exits, entries, execution) were exhausted** in the five 12-month iterations: completion exits (helped SOL, hurt BTC/ETH), deeper entries (collapsed samples), direction-only gates (hurt breakout), passive fills (PF 3.2 but 10 trades/year — unharvestable).
* The underlying cause is structural: these signals are pure price transforms whose gross edge per trade (~0.3–0.45% of notional) sits at or below the real round-trip cost (0.30%/0.25%). Three years of data across all regimes says this is not a market-luck artifact.

Spending the 3-iteration budget against a proven upper bound would manufacture an overfit, not an edge. **STOP clause exercised.**

## 6. Plain-language verdict and what to test next

**Is this strategy family dead? Yes.** Intraday RSI/MA/breakout long-only signals on 5m–15m crypto bars cannot pay Binance spot fees, in any regime, on three years of data, under honest execution. The one thing the system verifiably does well is capital preservation: max drawdown 7% while BTC drew down >40%, and the regime gate correctly held cash through the bear.

**Three fundamentally different signal types I would test next, and why:**

1. **Higher-timeframe trend/momentum (daily bars, 20–90-day lookbacks, weekly rebalance).** The single most robust documented anomaly in crypto. Trade frequency drops ~100× so the 0.3% round-trip cost becomes irrelevant (~10–30 trades/year vs 1,100); it harvests exactly the fat regime moves this study proved exist (the bull windows) instead of slicing them into fee-sized pieces. Our own data shows the shape: buy-and-hold over the full 36 months was strongly positive, and the losses came from the bear leg a trend filter sits out. We already have 3 years of daily-equivalent data; backtestable today with the same walk-forward harness. **Lowest cost, highest prior, first choice.**
2. **Cross-sectional relative value (ETH/BTC, SOL/BTC relative-strength rotation).** Long-only implementable (rotate capital toward relative strength, not shorting), and by construction immune to the primary-trend problem that killed every directional signal here — the bet is on *relative* mispricing between correlated assets, an edge source orthogonal to market direction.
3. **Funding-rate and liquidation-driven mean reversion.** Perp funding extremes and liquidation cascades mark *forced* flows — the genuine dislocations RSI only weakly proxies. Binance publishes funding rates free; this is a real information edge rather than a price transform, and it generates few, high-conviction events (fee-friendly). Higher implementation cost; test after #1/#2.

**Recommended next step:** run signal family #1 (daily trend/momentum) through this exact harness — same gate, same conservatism, 36 months, walk-forward. If crypto trend can't pass the gate either, the honest conclusion is that this account size/fee tier can't support systematic day trading, and the project should pivot to long-horizon systematic investing or stop.

> **Update 2026-07-10:** signal family #1 has been tested — see **docs/DAILY_MOMENTUM_RESULTS.md**. Short version: all 8 pre-registered configs are *profitable* (PF 2.0–3.7 rail-sized, fees only 1.2–3.5% of gross) but **all FAIL the gate** on trade count / window consistency (structural at weekly frequency) and, at deployable size, on drawdown; the 3-year sample is not statistically significant on its own (t ≤ 1.67); and at $100–500 the expected dollar edge is smaller than hosting costs. Full economics, regime breakdown, and the plain-language recommendation are in that report.

---

## Appendix A — original 12-month study (2025-07 → 2026-07, bear year)

Retained verbatim for the audit trail: baseline (zero trades — original formulas structurally incompatible with the engine's confidence gate), and 5 iterations (calibration → directional regime → completion exits → recombination → passive fills), best result rsi_mean_reversion 15m portfolio PF 0.84 / −1.13% / 327 trades vs hold-BTC −43%. IS-vs-OOS gap small throughout (0.97 vs 0.84) — never an overfitting story. Details: `docs/phase2_windows_baseline.md` … `_iter5.md`, `docs/phase2_metrics_*.json`, charts `docs/phase2_charts/`.

*Reproduce any run: `python scripts/run_walkforward.py --tag <tag> --fee <0.001|0.00075> --flags calibrated regime_filter [...]`.*
