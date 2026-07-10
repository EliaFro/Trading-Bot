# Published-Strategy Library Study — Results

## VERDICT: After correcting for multiple testing, ZERO of 29 published strategies show a real edge. The two naive "winners" are exactly what luck predicts (expected 1.5 false positives, observed 2), the Reality Check clears nobody (p=0.153) — and the single best performer in the entire study was a RANDOM strategy (t=2.49, +15.9%, PF 3.69), which outscored every published rule including the Turtle system.

**Date:** 2026-07-10 · **Data:** 36 months daily, BTC/ETH/SOL, verified gap-free · **OOS:** 2023-09-08 → 2026-06-24, 51 windows (19 bull / 18 sideways / 14 bear) · **Execution:** identical conservative harness (next-open fills, LIMIT lapses, 0.05% slippage, both fee tiers, long-only, rail sizing) · **Rules:** implemented exactly as published, zero tuning (all 25 non-TSMOM rules pass truncation-invariance lookahead tests) · **Benchmark:** hold-BTC over the same span: **+135.7%**.

---

## 1. The guard, stated up front

- **N = 29** strategies tested (25 classics + 4 TSMOM lookbacks). Testing 29 things at p<0.05 expects **~1.5 false positives by pure luck**.
- Significance bar after **Bonferroni**: p < 0.05/29 = **0.0017**.
- **White's Reality Check** (5,000 moving-block bootstraps of the demeaned 51×29 window-return matrix, preserving cross-strategy correlation): the observed *best* t-statistic must beat the distribution of best-of-29-under-the-null.
- **Noise control**: 100 random strategies (entry 5%/day, exit 8%/day — trade frequency matched to the library) through the *identical* pipeline.

## 2. Headline numbers

| check | result |
|---|---|
| Profitable at face value (rail sizing, 0.10% fee) | 24 of 29 |
| Naive p < 0.05 | **2** (`turtle_s1` t=2.18 p=.017, `price_above_sma50` t=1.76 p=.042) |
| Expected false positives at that bar | **1.5** |
| Bonferroni survivors (p < 0.0017) | **0** |
| Reality Check on the best (t=2.18) | **p = 0.153 — not significant** |
| Economic gate passers (PF>1.15, DD<15%, sample, 60% +windows) | **0** (the window-consistency criterion fails everything, as it did for TSMOM) |
| Noise strategies with naive p<0.05 | **9 / 100** |
| Noise passing the naive economic gate | 0 / 100 |
| **Best strategy in the whole study** | **a random one**: seed-1042-class noise, t=2.49, PF 3.69, +15.9% |

**Read that last row again.** One hundred coin-flip strategies produced a performer better than the best of 90 years of published technical analysis (Wilder 1978, Appel, Bollinger 2001, Lane, Granville, Hosoda, the Turtles, George & Hwang 2004, Moskowitz-Ooi-Pedersen 2012). That is what "best of many" looks like under luck — and it is precisely why the naive winners cannot be trusted.

## 3. Full library table (rail sizing, 0.10%/side; BNB tier shifts nothing material)

| strategy | OOS ret | PF | maxDD | trades | +win | t | naive p |
|---|---|---|---|---|---|---|---|
| ichimoku_standard | +21.4% | 4.33 | −10.9% | 28 | 20/51 | 1.45 | .077 |
| tsmom_60d | +20.5% | 3.66 | −10.3% | 36 | 21/51 | 1.42 | .081 |
| price_above_sma50 | +19.9% | 3.12 | −5.5% | 76 | 24/51 | 1.76 | .042 |
| turtle_s1_20_10 | +18.0% | 3.17 | −3.8% | 63 | 21/51 | **2.18** | .017 |
| tsmom_20d | +17.3% | 2.56 | −7.3% | 54 | 22/51 | 1.66 | .052 |
| ema_cross_12_26 / macd_zero | +16.6% | 2.63 | −6.0% | 47 | 22/51 | 1.50 | .070 |
| tsmom_90d | +16.4% | 3.13 | −8.1% | 24 | 20/51 | 1.23 | .113 |
| roc20_momentum | +14.8% | 2.04 | −5.0% | 147 | 23/51 | 1.54 | .065 |
| sma_cross_20_50 | +14.2% | 2.57 | −6.7% | 32 | 22/51 | 1.35 | .092 |
| turtle_s2_55_20 | +13.2% | 3.30 | −4.7% | 32 | 16/51 | 1.59 | .059 |
| tsmom_40d | +11.8% | 2.01 | −6.3% | 38 | 21/51 | 1.21 | .116 |
| adx_di_system | +11.2% | 2.55 | −4.8% | 45 | 18/51 | 1.56 | .062 |
| obv_trend | +11.1% | 1.62 | −5.1% | 173 | 22/51 | 1.40 | .083 |
| aroon_25 | +10.7% | 2.08 | −4.4% | 55 | 20/51 | 1.42 | .081 |
| keltner_breakout | +10.3% | 2.54 | −3.6% | 41 | 13/51 | 1.66 | .051 |
| macd_signal_cross | +8.8% | 1.52 | −5.7% | 113 | 19/51 | 1.02 | .155 |
| bollinger_breakout | +8.6% | 1.96 | −3.9% | 65 | 17/51 | 1.35 | .091 |
| cci_lambert_trend | +7.1% | 1.65 | −3.5% | 135 | 23/51 | 1.53 | .066 |
| psar_trend | +5.7% | 1.28 | −6.2% | 127 | 21/51 | 0.74 | .231 |
| faber_sma200 | +3.7% | 2.09 | −6.1% | 36 | 18/51 | 0.44 | .331 |
| williams_r | +2.9% | 1.27 | −6.2% | 74 | 32/51 | 0.45 | .326 |
| stochastic_oversold | +2.4% | 1.34 | −4.7% | 42 | 25/51 | 0.36 | .360 |
| sma_cross_50_200 | +1.4% | 1.36 | −7.6% | 9 | 13/51 | 0.19 | .426 |
| connors_rsi2 | +0.5% | 1.15 | −2.0% | 63 | 12/51 | 0.36 | .361 |
| bollinger_reversion | +0.2% | 1.02 | −6.0% | 50 | 22/51 | 0.06 | .478 |
| cci_reversion | −0.5% | 0.95 | −6.0% | 76 | 30/51 | −0.08 | .531 |
| rsi14_reversion | −0.7% | 0.76 | −4.3% | 17 | 8/51 | −0.52 | .698 |
| high_52w_momentum | −3.0% | 0.41 | −3.5% | 20 | 4/51 | −1.01 | .842 |

## 4. Honest observations in the pattern

1. **The apparent profits are mostly the market, not the rules.** Hold-BTC made +135.7% over this span; the *best* strategy captured +21.4% of it. Long-only rules in a net-bull period drift positive because being long sometimes beats being long never — that's beta leakage, not signal. The t-test on window returns partly controls for this; the noise control finishes the job (random longs also drifted positive; 9% of them "significantly").
2. **Every trend-family rule clusters together** (t ≈ 1.2–2.2): SMA/EMA/MACD/Ichimoku/Turtle/TSMOM/ROC/ADX are all the same one bet — "crypto trends" — expressed 15 ways. They are not 15 independent pieces of evidence; the Reality Check's correlation-preserving bootstrap accounts for exactly this.
3. **Mean-reversion rules are uniformly dead** (RSI-14, CCI-reversion, Bollinger-reversion, Connors: t ≤ 0.4) — consistent with both our Phase 2 intraday findings and the ML study.
4. **The famous names did not distinguish themselves**: the golden cross (50/200) managed 9 trades and t=0.19; the 52-week-high strategy was the worst thing tested.

## 5. Plain-language verdict

**The published technical-analysis canon, tested exactly as documented on 3 years of crypto data with honest execution and honest statistics, contains nothing distinguishable from luck at this horizon.** Not because everything lost money — most rules *made* money in a market that went up — but because (a) a best-of-29 selection this good arises by chance 15% of the time, (b) zero rules clear the Bonferroni bar, and (c) blind randomness, given the same number of tries per hundred, produced a better top performer than ninety years of trading literature.

The one honest signal in the noise: the *trend family as a whole* sits consistently on the positive side (nearly every trend rule t > 1.2, while mean reversion sits at zero) — which is the same single, weak, real phenomenon TSMOM already captures, not a menu of separate edges. If you ever deploy anything from this study at meaningful capital, it is that one family, at the TSMOM-60d-style implementation already frozen in `config/tsmom_frozen.yaml` — and nothing else here earns a place beside it.

*Reproduce: `python scripts/run_signal_library.py` · Raw: `docs/signal_library_metrics.json` · Chart: `docs/phase2_charts/signal_library.png` (t-distribution of published vs noise strategies — the two histograms substantially overlap, which is the whole story in one picture).*
