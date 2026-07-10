# Contributing

This project is an **evaluation harness and a finding**, not a strategy
collection. Its value is the method: walk-forward evaluation, purge/embargo,
executable anti-lookahead tests, honest execution costs, noise controls, and
multiple-testing correction.

## What's welcome

- Harness improvements: better statistical corrections, tighter leak tests,
  more realistic execution modeling (with citations/measurements), clearer
  reporting.
- Reproduction reports: run the studies on other assets/periods/venues and
  share the corrected results — *especially* when they disagree with ours.
- Bug reports on anything that could leak future information. These are
  treated as the highest-severity issues in the repo.

## What's not

- **Strategy submissions are run through the corrected gate or not at all.**
  If you propose a signal, it enters a counted N, faces Bonferroni + Reality
  Check against the noise baseline, pays full measured costs, and its
  in-sample numbers are never discussed. PRs that celebrate uncorrected or
  in-sample results will be closed with a link to
  docs/SIGNAL_LIBRARY_RESULTS.md (where a coin flip beat the Turtles).
- Anything that touches live trading. This repository is paper-only by
  design; the Fast Lab horizon is additionally under a permanent,
  self-executed closure rule (src/trading/kill_rule.py).

## Ground rules (the project's standing constitution)

No gate-loosening. No cherry-picking after results. Pre-register predictions
for any new study. The noise control is the baseline for every claim.
Verdict first, evidence second, plain language always.
