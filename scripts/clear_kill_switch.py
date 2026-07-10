#!/usr/bin/env python3
"""
Clear the persistent kill switch after reviewing why it fired.

The kill switch triggers when equity drops max_drawdown (15%) below the
starting balance. It closes all positions and halts trading — across
restarts — until you run this script.

Usage:
    python scripts/clear_kill_switch.py           # show state
    python scripts/clear_kill_switch.py --clear   # clear it (asks to confirm)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.database import DatabaseManager
from src.trading import safety


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--yes', action='store_true', help='skip confirmation')
    args = parser.parse_args()

    db = DatabaseManager(os.getenv('DB_PATH', './data/trading_system.db'))
    reason = safety.is_kill_switch_triggered(db)

    if not reason:
        print("Kill switch is NOT triggered. Trading is allowed.")
        return

    print(f"Kill switch IS triggered:\n  {reason}\n")
    if not args.clear:
        print("Re-run with --clear to reset it (only after you understand "
              "why it fired — also reset the starting-equity baseline below).")
        return

    if not args.yes:
        answer = input("Clear the kill switch AND reset the starting-equity "
                       "baseline to current equity? [y/N] ")
        if answer.strip().lower() != 'y':
            sys.exit('Aborted.')

    safety.clear_kill_switch(db)
    # Reset the drawdown baseline so it doesn't immediately re-trigger
    from sqlalchemy import text
    with db.engine.connect() as conn:
        conn.execute(text("DELETE FROM system_config "
                          "WHERE config_key LIKE '%_starting_equity'"))
        conn.commit()
    print("Kill switch cleared and baseline reset. Restart the bot to resume.")


if __name__ == '__main__':
    main()
