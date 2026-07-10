# Signal Family #1 — Daily Time-Series Momentum (TSMOM), Weekly Rebalance

**Date:** 2026-07-10 · **Data:** 36 months daily bars (resampled from verified gap-free 1h), BTC/ETH/SOL · **OOS span:** 2023-09-08 → 2026-06-24 (identical Phase 2 walk-forward calendar: 51 twenty-day windows — 19 bull / 18 sideways / 14 bear)

## VERDICT: ALL 8 CONFIGURATIONS FAIL THE GATE AS WRITTEN — but this family is NOT dead. Read §5 carefully: the failure modes are structural to the gate at this horizon, the sample is too thin to prove the edge, and the honest economics at $100–500 are the real blocker.

---

## 1. Design (pre-registered — nothing tuned from data)

Long when trailing L-day return > 0, else cash; L ∈ {20, 40, 60, 90} — **all four reported, none selected**. Decision every Monday UTC close, execution next day's open through the identical Phase 2 fill model (marketable LIMIT that lapses on gaps, 0.05% slippage, both fee levels). No stops — the exit is the signal flip (canonical TSMOM). Long-only spot. Two sizing variants: **rail-compliant** (10%/position → max 30% deployed) and **equal-weight** (33%/position — would require deliberately raising the Phase 4 per-position rail; the DD gate binds it harder).

Because zero parameters are optimized, the 60d train / 2d embargo windows have nothing to train — the walk-forward reduces to a pure out-of-sample evaluation on the *same* 51-window calendar, used for the consistency gate and regime breakdown. This is stated openly: per-window re-optimization of a weeks-horizon strategy on 60-day windows (~1–3 trades) would be statistical noise, and pretending otherwise would be methodology theater.

## 2. Results (fee 0.10%/side; BNB 0.075% changes nothing material — see fee column)

**Benchmarks over the same OOS span:** hold-BTC **+135.7%** (maxDD −51.2%, Sharpe 0.88) · hold-basket **+127.4%** (maxDD −67.5%, Sharpe 0.77) · cash 0%.

| config | total ret | CAGR | PF | Sharpe | maxDD | trades | +windows | t-stat | fees % of gross | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TSMOM 20d, rail 10% | +17.3% | +5.8% | 2.55 | 1.04 | −7.3% | 56 | 22/51 (43%) | 1.67 | 2.5% | **FAIL** |
| TSMOM 40d, rail 10% | +11.8% | +4.0% | 2.01 | 0.69 | −6.3% | 38 | 21/51 (41%) | 1.22 | 2.5% | **FAIL** |
| TSMOM 60d, rail 10% | +20.5% | +6.8% | 3.66 | 0.86 | −10.3% | 36 | 21/51 (41%) | 1.44 | 1.5% | **FAIL** |
| TSMOM 90d, rail 10% | +16.4% | +5.5% | 3.13 | 0.73 | −8.1% | 24 | 20/51 (39%) | 1.24 | 1.2% | **FAIL** |
| TSMOM 20d, equal-wt 33% | +58.1% | +17.5% | 2.18 | 1.04 | −22.8% | 56 | 22/51 | 1.58 | 3.2% | **FAIL** |
| TSMOM 40d, equal-wt 33% | +34.4% | +11.0% | 1.64 | 0.67 | −21.9% | 38 | 21/51 | 1.10 | 3.5% | **FAIL** |
| TSMOM 60d, equal-wt 33% | +61.3% | +18.4% | 2.66 | 0.89 | −25.9% | 36 | 20/51 | 1.45 | 2.2% | **FAIL** |
| TSMOM 90d, equal-wt 33% | +52.3% | +16.0% | 2.54 | 0.80 | −20.7% | 24 | 20/51 | 1.33 | 1.7% | **FAIL** |

**Why they fail the gate:** rail-compliant configs pass PF (2.0–3.7 ≫ 1.15) and DD (6–10% < 15%) but fail **trade count** (24–56 ≪ 200) and **positive windows** (39–43% < 60%). Equal-weight additionally fails **DD** (21–26% > 15%). Both fee levels produce identical verdicts.

**Fees are near-irrelevant, as predicted:** 1.2–3.5% of gross P&L (vs >100% for the intraday family). At this frequency the fee tier does not matter.

Chart: `docs/phase2_charts/daily_tsmom.png` (both sizing variants vs hold-BTC vs cash). Raw metrics: `docs/daily_momentum_metrics.json`.

## 3. Regime breakdown — where it wins and what it gives back

TSMOM 60d equal-weight (representative; all lookbacks show the same shape):

| regime | windows | compounded return | +window % |
|---|---|---|---|
| bull | 19 | **+160.3%** | 74% |
| sideways | 18 | −4.0% | 28% |
| bear | 14 | **−35.5%** | 7% |

Three honest observations:

1. **It captures the bull moves** — that is the entire source of profit, exactly as the family's thesis predicts.
2. **The Phase 2 defense story does NOT survive at this horizon.** The intraday system's 2% drawdowns are gone. TSMOM participates in every sharp correction to the extent of its lag (many "bear" windows here are −5%+ corrections *inside* the 2024 bull, during which TSMOM was correctly long and took the hit). Its protection is only against *extended* downtrends: −26% maxDD vs −67.5% for holding the basket. Real protection, different animal.
3. **Sideways markets bleed slowly** (whipsaw entries/exits), −4% over 18 windows — tolerable, not free.

## 4. Sample size: is this statistically meaningful? Plainly: no, not on its own.

- 24–56 round trips in 2.8 years; window-return t-stats 1.10–1.67 — **below the conventional 1.96 significance bar** in every configuration.
- Profits are concentrated: a handful of large bull-leg trades carry each config (classic trend profile). Remove the best 2–3 trades and several configs go flat.
- The gate's 200-trade requirement is **structurally unreachable** for a weekly-rebalance 3-asset strategy (~10–20 trades/year is intrinsic). Per instructions, the gate is not relaxed: the verdict stands FAIL. The evidential weight for TSMOM comes from the external literature (decades, hundreds of assets); this 3-year, 3-asset sample is *consistent with* that literature but cannot independently prove an edge. Trading it means trusting the literature, not this backtest.

## 5. Minimum account size (deliverable #4)

Mechanical floor (Binance MIN_NOTIONAL $10/order, LOT_SIZE steps negligible at these prices):

| sizing | min position | mechanical account floor |
|---|---|---|
| rail 10%/position | $10 | **~$100** |
| equal-weight 33% | $10 | **~$35** |

**The real floor is economic, not mechanical.** A 24/7 bot costs ~$60–120/year to run (cheap VPS). Expected annual dollar edge at the observed CAGRs:

| account | rail 10% (~6% CAGR) | equal-wt 33% (~18% CAGR, −26% DD) | vs. bot cost |
|---|---|---|---|
| $100 | ~$6/yr | ~$18/yr | **under water** |
| $500 | ~$29/yr | ~$92/yr | ~break-even |
| $2,000 | ~$116/yr | ~$368/yr | viable |
| $5,000 | ~$290/yr | ~$920/yr | comfortably viable |

**$100–500 is mechanically viable and economically pointless: hosting costs eat the expected edge.** The realistic floor for this product to make financial sense is **$2,000–5,000**.

## 6. The framing question, answered straight

**Yes, this is no longer day trading** — TSMOM holds for weeks (avg hold 9–30 days depending on lookback) and is flat much of the time. I understand you're accepting that product shift.

**Can a retail account your size realistically do daily-horizon systematic trading?** Mechanically yes. Honestly: **at $100–500, no version of this project beats simple periodic buying after costs.** The numbers from this exact study: holding the basket returned +127% over the OOS span; the best deployable TSMOM captured +61% (though at 26% drawdown instead of 67%). TSMOM's genuine value is *drawdown reduction at meaningful capital* — it is a risk-management product, not a small-account enrichment product. At your stated capital:

- The expected dollar edge over DCA is tens of dollars a year — less than VPS hosting.
- The one demonstrably valuable behavior (exiting extended bears) can be replicated by hand with a **monthly** check ("is BTC above its 200-day average? if not, pause buying") — no bot required.

**My honest recommendation:** at $100–500, DCA into BTC (optionally with that one manual monthly trend rule) is the rational choice, and it isn't close. This bot becomes rationally deployable at ~$2k+ using the TSMOM profile — *if* you accept that its edge rests on the external trend-following literature rather than on statistically conclusive proof from our own 3 years, and that the honest expectation is bull-capture with −20–26% drawdowns, not steady small daily gains. The original vision — many small intraday profits compounding — is dead on the evidence: three years, five strategy iterations, two fee tiers, and every intraday configuration lost to fees.

*Reproduce: `python scripts/run_daily_momentum.py`.*
