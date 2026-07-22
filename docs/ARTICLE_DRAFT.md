# I Spent a Project Trying to Beat Crypto Markets. A Coin Flip Beat Me To It.

*Draft — structure, numbers, and charts final; voice to be edited by the author. All results reproducible from the repository; every claim links to its evidence document.*

---

## 1. The dream, stated honestly

I wanted what everyone wants: a bot that day-trades crypto around the clock, harvesting many small gains that compound while I sleep. I had a half-built codebase, three years of minute-by-minute Binance data within reach, and the modern retail toolkit — technical indicators, walk-forward backtesting, machine learning, even a pre-trained financial language model.

What I ended up with is, I think, more valuable than the bot: a measurement instrument that is very hard to fool, and a set of numbers that explain — mechanically, not moralistically — why the retail day-trading dream fails, why it *looks* like it works, and what the honest alternative is.

The single result that summarizes everything: **I implemented 29 famous published trading strategies exactly as documented — Wilder's RSI, Bollinger Bands, Ichimoku, the Turtle system, time-series momentum — and ran 100 random coin-flip strategies through the same pipeline. The best coin-flip strategy beat the best published strategy.** (t = 2.49 vs 2.18; [SIGNAL_LIBRARY_RESULTS.md](SIGNAL_LIBRARY_RESULTS.md).)

## 2. The instrument: how to stop lying to yourself with a backtest

Backtests are machines for generating false confidence. Before testing anything, I built the defenses, and made each one an *executable test* rather than a promise:

- **Walk-forward evaluation.** Parameters and models see only the past; each future window is touched exactly once. In-sample performance is recorded but never celebrated — it turned out to be an *anti-signal* (§6).
- **Purge and embargo.** A label that looks 5 days ahead means the last 5 days of every training window silently contain the test's future; they get purged, plus an embargo gap.
- **Truncation-invariance tests.** Every feature and signal must produce the identical value at time T whether or not data after T exists. This one-line idea catches almost every accidental lookahead.
- **The no-skill-on-noise test.** The entire ML pipeline runs on synthetic random walks and must find nothing. If it finds skill in noise, it's leaking — the test fails the build.
- **Honest execution.** Fills happen on the *next* bar through a limit-order model that misses fills on gaps; costs are 0.10%/side fees, 0.05% slippage, plus bid-ask spreads I *measured from the live order book* (fun fact: BTC's spread is one tick — 0.00002% — the costs that matter are fees). Every simulated trade records a decomposition: gross move − fees − spread − slippage = net.
- **Multiple-testing correction and a noise control.** If you test N strategies at p<0.05, ~N/20 pass for free. So N is counted, the significance bar is Bonferroni-adjusted, a Reality Check bootstrap prices the "best of N" effect — and 100 random strategies run through the identical pipeline as the luck baseline for every claim.
- **Pre-registration.** Before each study, predictions with probabilities went into the plan document, and got scored after. My hit rate is in the repo, including the misses.

## 3. Finding #1: the baseline was zero — literally

The inherited codebase's strategies had never traded once: their confidence formulas were mathematically incapable of clearing their own execution gate (a moving-average crossover's "strength" is zero *at the crossover, by definition*). The first honest backtest of the "working" system produced **zero trades**. Every prior in-sample result had been structurally impossible.

Lesson zero of retail algo-trading: most bots you can download have never really been tested, including by their authors.

## 4. Finding #2: the fee wall

After recalibrating the strategies so they could actually trade, twelve months of walk-forward testing on 5- and 15-minute bars produced a beautifully consistent picture ([PHASE2_RESULTS.md](PHASE2_RESULTS.md)):

- The best intraday configuration earned ~**+0.43% gross per trade**.
- A round trip costs ~**0.30%** (mostly exchange fees).
- Net result, after five hypothesis-driven improvement rounds and two fee tiers: **negative in every market regime.** Gross-positive, fee-negative — a strategy that works perfectly except for existing in the real world.

Extending to 36 months (a full bull-bear cycle) didn't change it. The regime breakdown put a hard ceiling on hope: even a filter with *perfect hindsight*, trading only the windows that turned out bullish, would have landed below breakeven-after-fees. When the hindsight-optimal version of your idea loses, iteration is over. (That realization — a computable "futility bound" — saved weeks of self-deception.)

I later learned this is the oldest result in the field, measured at national scale: the complete-record Taiwan study (Barber, Lee, Liu & Odean) found day traders losing 23.9 bps/day *net* against only −7 bps gross — transaction costs more than **tripled** the loss ([EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md)). The fee wall isn't my discovery; it's the industry's load-bearing wall.

## 5. Finding #3: ninety years of trading literature vs. one hundred coin flips

The centerpiece study: 29 published strategies, implemented exactly as their authors documented, zero tuning, each passing lookahead tests, all facing 51 out-of-sample windows across bull, bear, and sideways regimes.

At face value the literature looks great: **24 of 29 were profitable.** The famous Turtle breakout system posted a profit factor of 3.17 and the study's best t-statistic, 2.18 — "significant" at p=0.017.

Then the corrections arrive:

- Testing 29 things at p<0.05 hands you **~1.5 winners by pure luck**. We observed 2.
- Bonferroni bar (p < 0.0017): **zero survivors.**
- Reality Check: a best-of-29 as good as t=2.18 arises **15% of the time from nothing.**
- And the kill shot: of 100 random strategies with matched trade frequency, 9 were "significant" — and the best of them (t=2.49, +15.9%, PF 3.69) **outperformed every published strategy in the study.**

*(Chart: `phase2_charts/signal_library.png` — the t-statistic histograms of published strategies and coin flips, substantially overlapping. That overlap is the finding.)*

The academic literature reached the same verdict about itself: Harvey, Liu & Zhu audited 313 published return factors (Review of Financial Studies, 2016), concluded a genuine discovery now needs **t > 3.0** after multiple-testing correction — a bar only ~9 of 313 clear — and wrote that "most claimed research findings in financial economics are likely false." My best published strategy's t = 2.18 wouldn't have gotten in their door either ([EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md)).

The profits weren't fake, exactly — they were *the market's*: hold-BTC returned +135% over the span, and the best strategy captured +21% of that while "winning." Long-only rules in a rising market drift positive whether or not they contain information. Nearly every trend-flavored rule clustered together statistically, because they are one bet — "crypto trends" — wearing fifteen costumes; and mean-reversion rules were uniformly dead. The only honest residue in the whole canon is that single trend bet, which brings us to—

## 6. Finding #4: machine learning — genuinely predictive, genuinely unprofitable

ML got two fair trials with every guard active.

**Daily horizon (~3,000 samples):** gradient boosting hit 92% accuracy in training and 36% out-of-sample — three points *above* pure chance, wearing a 56-point overfitting gap. The humble logistic regression beat every fancier model out-of-sample. The system's dashboard literally displays, in red, "**this model has memorized noise**" about its own champion. ([ML_RESULTS.md](ML_RESULTS.md))

**One-minute horizon (~1,000,000 samples)** — the fair trial deep learning rarely gets in these arguments, with LSTMs and CNNs trained on Apple-Silicon GPU across 15 windows spanning three years ([FASTLAB_RESULTS.md](FASTLAB_RESULTS.md)):

- **The skill is real.** Out-of-sample balanced accuracy 0.44–0.45 against a 3-class chance of 0.333 — **+11 points, in every window, in every regime**, with tiny train/test gaps (at that data scale, nothing memorizes). Short-horizon crypto price movement is *partially predictable*. The efficient-market purists are, at the minute scale, measurably wrong.
- **Deep learning still didn't beat a random forest** (0.442 vs 0.451).
- **And none of it matters**, because the genuine skill converts to **+1.2 basis points of gross edge per trade**, and the toll booth charges **31 basis points**. Trading the predictions: profit factor 0.22, zero positive windows out of fifteen. The model is right about the market and still loses 0.30% per trade — *predictive ≠ profitable*, demonstrated with actual prediction rather than its absence.

This is precisely how the firms that *do* make money are built — just from the other side of the toll booth. Virtu Financial's IPO filing disclosed **one losing day in 1,238** while profitably exiting only **~49% of individual positions**: near-certain daily profits with coin-flip position outcomes is the signature of spread capture at enormous volume, not direction forecasting ([EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md)). Direction — the only edge retail can chase — is the part that doesn't pay.

*(Chart: `phase2_charts/fastlab_partB.png` — gross edge per trade vs round-trip cost, side by side. The blue bars are barely visible next to the red ones. That's the entire retail high-frequency dream in one image.)*

## 7. The arithmetic that explains an industry

Put the three numbers in one place:

| | per trade |
|---|---|
| what 1-minute ML skill is worth | ~+0.012% |
| what the best intraday signals earned | ~+0.4% gross |
| what a retail round trip costs | ~0.30% |

Everything about retail trading culture falls out of this arithmetic. Why do backtests look great? Because gross edges are real and in-sample numbers ignore the toll. Why do influencer strategies "work" in bull markets? Because long-only anything works in bull markets. Why does the industry sell *courses, signals, and bots* rather than trading? Because teaching the dream pays fees to the teacher, while living it pays fees to the exchange. Nobody in the pipeline needs to be lying — the survivor-bias and best-of-N machinery does the deceiving automatically, at scale, for free.

And why do institutions extract what retail can't? They pay maker rebates instead of taker fees, at horizons where 1bp of edge times enormous volume clears their (much lower) toll. The numbers are public: Binance's base retail fee is 0.10% per side, its top VIP tier pays ~0.011% maker (0.00% maker on futures), and designated market makers on many venues pay *negative* fees — they are paid to trade ([EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md)). The edge I measured is probably *someone's* profit. It cannot be a $500 account's.

## 8. What actually won

Thirty-six months, ten thousand lines of evaluation code, five strategy families, two ML scales, one conclusion with a straight face: **buying periodically and holding won.** The one defensible refinement — worth having and executable by hand in ten minutes a month — is a trend filter: pause new buys when BTC closes below its 200-day average. It doesn't beat holding on raw return; it cuts the worst drawdowns from −67% to −26% at meaningful capital, and it costs nothing.

So the project's real-money output is a one-page playbook and a Telegram reminder service. The bot? It still runs — as an observatory. It retrains on schedule behind a guard that keeps old models unless new ones genuinely improve, displays its own overfitting in red, labels which of its features rest on three observations versus three hundred, decomposes every paper trade into edge-minus-toll — and it carries a **pre-registered, self-executing kill rule**: on 2026-08-07, unless something clears the corrected bar after full costs (nothing is on track to), strategy search at the fast horizon closes permanently, by code, without asking anyone's mood. (One disclosed limitation of that live record: it runs on my laptop, so it only samples the hours the machine is awake — a selection bias the offline studies, which saw every bar, did not have.)

## 9. What I'd tell someone starting where I started

1. Build the referee before the athletes. The harness is worth more than any strategy you'll feed it.
2. Put costs *inside* your labels and your first sanity check. Most "edges" die right there, before a single model trains.
3. Run coin flips through everything. If you can't tell your best idea from your best coin flip, you don't have an idea yet.
4. Pre-register your predictions. Being on the record changes what you allow yourself to believe afterward.
5. When the hindsight-optimal version of a plan loses money, stop. There is no tuning your way past that bound.
6. And if a system's in-sample numbers are spectacular, treat that as the alarm, not the achievement.

The August 7 stop, committed in code before the results existed, is just the field's own answer to data-mining made physical: decide the bar first, then let it bind ([EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md)).

The markets kept the money. I kept the instrument, and the numbers. Given how most retail trading stories end, that's a win.

---

*Reproduction: github repo, `make install && make test`, then the scripts in the README. Charts referenced: `signal_library.png`, `fastlab_partB.png`, `ml_study.png`. Evidence documents: PHASE2_RESULTS.md · SIGNAL_LIBRARY_RESULTS.md · DAILY_MOMENTUM_RESULTS.md · ML_RESULTS.md · FASTLAB_RESULTS.md.*
