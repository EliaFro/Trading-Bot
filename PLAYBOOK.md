# The 10-Minute-a-Month Bitcoin Playbook

*Replaces the trading bot. One purchase rule, one risk rule, once a month.
Based on 36 months of tested data — see docs/DAILY_MOMENTUM_RESULTS.md for the receipts.*

---

## The schedule

**Buy day: the 1st of every month** (or the next day you have 10 minutes).
Same fixed amount every month — pick a number you would not miss if it went to zero, and never raise it because prices are going up.

Buy **BTC only**, on **Binance spot**, with a plain **market order**. At these amounts the 0.1% fee and spread are pennies — don't optimize them.

## The one rule (this is the whole system)

On your buy day, check whether Bitcoin is **above or below its 200-day moving average**:

- **Above the line → buy your fixed amount as scheduled.**
- **Below the line → skip the buy. Keep the cash.** Add the skipped money to your next buy when the price is back above the line.

**Never sell based on this rule.** It controls when you *add* money, nothing else. (Selling adds whipsaw, stress, and taxes; the historical value of this rule is simply not shoveling new money into extended downtrends.)

## How to check the 200-day MA (2 minutes, free)

1. Go to **tradingview.com** (no account needed) → search **BTCUSD**.
2. Set the chart to **1D** (daily).
3. Click *Indicators* → search **"Moving Average"** → add it → set **Length = 200**.
4. Is today's price above or below that line? That's your answer.

(Alternative: the Binance app chart with an MA(200) on the 1D view shows the same thing.)

You can save the TradingView layout once and the whole monthly check becomes: open bookmark, look, buy or don't.

## Account hygiene (once, then quarterly)

- 2FA on, withdrawal allow-list on, no API keys with trading rights left active.
- Every quarter, consider moving accumulated coins to self-custody if the balance has grown meaningful.

## What to ignore (equally important)

- **News and price predictions** — none of it changes the rule.
- **Red days / "buying the dip"** — the rule already handles downtrends better than your gut will.
- **FOMO on green days** — buying extra because it's pumping is how the rule dies.
- **Altcoins, leverage, futures, staking-yield schemes** — out of scope, permanently.
- **Checking the price between buy days** — nothing you see can trigger an action, so looking is pure stress.
- **Anyone selling a bot or signal group** — you have 36 months of walk-forward evidence in this repo of what intraday signals earn after fees: less than zero.

## What to honestly expect

- Bitcoin can still draw down 50–70%. The rule reduces how much of *your new money* enters those periods; it does not make crypto safe.
- Some skipped months will look "wrong" in hindsight (price recovers fast). That's the cost of the rule working over years instead of weeks.
- This playbook's edge is measured in *avoided mistakes*, not in beating the market.

**Total time: ~10 minutes a month. If you're spending more than that, you're doing it wrong.**
