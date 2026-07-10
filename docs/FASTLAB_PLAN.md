# Fast Lab — Pre-Registered Plan

**Date locked: 2026-07-10, before any Fast Lab study was run.**

## Purpose (governs everything)

The Fast Lab is a **learning accelerator, not a profit path**. Phase 2 proved
the fee arithmetic kills intraday trading; at 1 minute it is worse, and the
Fast Lab's banner states this from day one. It exists to: (a) give the ML
core ~500× more training samples, (b) make learning dynamics observable in
days instead of years, (c) test one genuinely untested hypothesis
(multi-timeframe gating), (d) show the fee wall live.

## Execution model at 1m/5m (Part A — measured, conservative, justified)

- **Spread, measured 2026-07-10 from the live Binance book** (30 samples/symbol,
  `docs/spread_measurements.json`): BTC 0.00002% (one tick), ETH 0.00056%,
  SOL 0.0126%. **Used values are conservative floors far above measurement:**
  BTC/ETH **0.01%**, SOL **0.02%** — because calm-market sampling understates
  stressed spreads. Half-spread charged on each side of every taker fill.
- Slippage: unchanged 0.05%/side (measured top-of-book depth $8–16k vs our
  $10–3,300 orders means real impact is far smaller; we keep the harsher number).
- Fees: 0.10%/side taker (0.075% BNB tier as sensitivity).
- **Total modeled round trip: ~0.31–0.32%** — ~97% of it is fees+slippage,
  ~3% spread. The fee wall at 1m is the fee, not the microstructure.
- Fills otherwise identical to Phase 2 (next-bar marketable LIMIT that lapses
  on gaps; SL beats TP intrabar; gaps fill at the worse open).
- **Fee decomposition** recorded per trade (gross − fees − spread − slippage
  = net, with the itemization exact in aggregate) and shown on the dashboard.

## Part B — multi-timeframe family (Elder triple-screen), N counted

Cited design: Elder, *Trading for a Living* (1993) — Screen 1: higher-TF tide
(EMA-26 slope or MACD-histogram rising); Screen 2: oscillator pullback
against the tide (Force Index(2) < 0 or Stochastic %K < 30); Screen 3:
trailing buy-stop above the prior bar's high; initial stop below the prior
bar's low; exit on tide flip or oscillator overbought. Long-only spot.

**Variants (all documented components, zero tuning): 2 gates × 2 oscillators
× 2 entry TFs (1m, 5m) = N=8.** Higher TF = 4h (resampled from 1h, gate
values available only after the 4h bar closes — lookahead-tested).

Anti-data-mining kit: Bonferroni at 0.05/8, White's Reality Check across the
8-variant window-return matrix, and a 100-random-strategy noise control at
matched trade frequency (60 on 5m, 40 on 1m) through the identical pipeline.
Walk-forward calendar identical to prior studies (62-day offset, 51 × 20-day
OOS windows, regimes labeled). Ceiling check reported per variant: mean gross
edge per trade vs total round-trip cost — **if gross < cost, no filter can
fix it and the report must say so.**

**Pre-registered predictions (Part B):**
- P(any variant clears the corrected significance bar after full costs): **~5%**
- P(any variant's mean gross edge per trade exceeds the 0.31% round trip): **~15%**
- Most likely outcome: gross edge per trade in the 0.05–0.20% range —
  structurally beneath the cost floor, the Phase 2 lesson restated at 1m.

## Part C — deep models' fair trial at 1m scale

Data: ~1.55M 1m bars/symbol (36 months). Sampled at stride 5 (overlapping
1m rows are ~redundant) → ~310k rows/symbol, ~930k pooled.

Models: the tree/linear trio (baseline, unchanged) **plus LSTM and 1D-CNN**
(small: ≤200k parameters, sequences of 60 bars × ~8 channels), which were
refused at 3,100 daily samples and now get a methodologically fair trial at
~930k samples. Torch on Apple-Silicon MPS if available, CPU otherwise.

Labels: executable open(T+1)→open(T+31) (30-minute horizon), **net of the
full modeled round trip**; ±0.10% dead zone. At this frequency most bars
will honestly label FLAT — that is the fee wall appearing in the target.

Walk-forward: **15 test windows of 20 days, spread evenly across the full
36 months** (every ~2.4 months) — pre-registered here, before results: this
covers all regimes at ~1/4 the deep-training cost of exhaustive rolling
windows; each window's training uses only strictly-prior data (60-day train,
30-minute purge, 1-day embargo). Statistical power at 1m comes from
thousands of OOS decisions per window, not calendar length. All existing
guards adapted and enforced by tests: truncation-invariance, purge/embargo
assertion, no-skill-on-random-walk, scalers fit on train only, validation-
slice selection (reported OOS never used for any decision).

**Pre-registered predictions (Part C):**
- P(deep models beat the trees on OOS balanced accuracy): **~35%**
- P(deep models show real classification skill above chance): **~50%**
  (vol/microstructure regularities exist at 1m and big samples find them)
- P(anything — deep or tree — beats FULL costs in the trading simulation):
  **~3%.** Classification skill ≠ net profit; the 0.31% round trip demands
  an edge per trade that 1m price movement rarely offers.

## Part D — deployment isolation & resource budget

Separate paper account in a separate SQLite DB (`data/fastlab.db`); OHLCV
read from the main DB (WAL mode) — the daily lab's fetcher already maintains
1m/5m/1h. Separate launchd service, separate dashboard tab (⚡ Fast Lab) with
the honest banner and live fee decomposition. Retrain caps (pre-registered):
trees at most every 24h, deep models at most weekly. Estimated footprint:
~5–15% of one CPU core average (1m decision loop + hourly data reads),
1–2 GB RAM transient during retrains, ~0.5 GB disk. The daily lab and
dashboard must stay responsive; if they don't, the Fast Lab's loop slows
itself down, not them.

## Part E — pre-registered kill criteria (also in OPERATING.md)

Offline studies (B and C) **plus a 4-week live-paper window ending
2026-08-07**. If by then nothing has cleared the multiple-testing-corrected
significance bar **after full costs**, the Fast Lab's strategy-search role
**ends permanently** — no further strategy iterations at 1m/5m, ever. It may
continue to run solely as an observation instrument for watching the ML
learn. No gate-loosening, no timeframe/symbol cherry-picking after results,
in-sample numbers never celebrated, the noise control is the baseline for
every claim, verdicts stated first.
