#!/usr/bin/env python3
"""
Record the DCA buys you actually make (manual ledger — PLAYBOOK.md).

    python scripts/playbook_log.py add --amount 50
    python scripts/playbook_log.py add --amount 50 --price 63200 --date 2026-07-01
    python scripts/playbook_log.py list
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from scripts.init_db import init_db
from src.utils.database import DatabaseManager
from src.playbook import companion

STATE_DB_PATH = './data/playbook.db'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    add = sub.add_parser('add', help='record a buy you made')
    add.add_argument('--amount', type=float, required=True, help='USD spent')
    add.add_argument('--price', type=float, default=None,
                     help='BTC price paid (default: that day\'s close)')
    add.add_argument('--date', type=str, default=None, help='YYYY-MM-DD')
    add.add_argument('--note', type=str, default='')
    sub.add_parser('list', help='show the ledger')
    args = parser.parse_args()

    if not Path(STATE_DB_PATH).exists():
        init_db(STATE_DB_PATH)
    state_db = DatabaseManager(STATE_DB_PATH)
    main_db = DatabaseManager('./data/trading_system.db')

    if args.cmd == 'add':
        buy_date = date.fromisoformat(args.date) if args.date else None
        rec = companion.log_buy(state_db, main_db, args.amount,
                                price=args.price, buy_date=buy_date,
                                note=args.note)
        print(f"logged: {rec['buy_date']}  ${rec['amount_usd']:,.2f} -> "
              f"{rec['btc_amount']:.8f} BTC @ ${rec['btc_price']:,.2f}")

    summary = companion.ledger_summary(state_db, main_db)
    log = summary['log']
    if log.empty:
        print("ledger is empty")
        return
    print(f"\n{'date':12s} {'usd':>10s} {'price':>12s} {'btc':>12s}  note")
    for _, r in log.iterrows():
        print(f"{r['buy_date']:12s} {r['amount_usd']:>10,.2f} "
              f"{r['btc_price']:>12,.2f} {r['btc_amount']:>12.8f}  "
              f"{r['note'] or ''}")
    if summary.get('invested'):
        print(f"\ninvested ${summary['invested']:,.2f} -> "
              f"{summary['btc_total']:.8f} BTC "
              f"(cost basis ${summary['cost_basis']:,.2f})")
        print(f"current value ${summary['current_value']:,.2f} "
              f"@ ${summary['latest_price']:,.2f}")
        if 'lump_sum_value' in summary:
            print(f"lump-sum-on-day-one comparison: "
                  f"${summary['lump_sum_value']:,.2f}")


if __name__ == '__main__':
    main()
