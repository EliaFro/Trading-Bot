#!/usr/bin/env python3
"""
Playbook Companion service — reminder + ledger for the manual DCA plan.

NEVER places orders. Never touches an API key with trade permission.
Never suggests deviating from the schedule. (src/playbook/companion.py
carries the same guarantee, enforced by a source-scan test.)

Runs as a launchd agent: wakes hourly, acts once per UTC day at/after
PLAYBOOK_CHECK_HOUR_UTC (default 06:00), sends only the messages defined
in PLAYBOOK.md's contract, and triggers the monthly evidence digest on
the 1st.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),
              RotatingFileHandler('logs/playbook.log',
                                  maxBytes=5 * 1024 * 1024, backupCount=2)])
logger = logging.getLogger('playbook')

from scripts.init_db import init_db
from src.utils.database import DatabaseManager
from src.utils.notifier import Notifier
from src.playbook import companion

CHECK_HOUR_UTC = int(os.getenv('PLAYBOOK_CHECK_HOUR_UTC', '6'))
STATE_DB_PATH = './data/playbook.db'


async def main():
    if not Path(STATE_DB_PATH).exists():
        init_db(STATE_DB_PATH)
    state_db = DatabaseManager(STATE_DB_PATH)
    companion.ensure_tables(state_db)
    main_db = DatabaseManager('./data/trading_system.db')
    notifier = Notifier(db=state_db)
    logger.info(f"Playbook companion up (daily check at "
                f"{CHECK_HOUR_UTC:02d}:00 UTC)")

    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()
            already = companion.get_state(state_db, 'last_daily_run') == today
            if now.hour >= CHECK_HOUR_UTC and not already:
                sent = companion.run_daily_check(main_db, state_db, notifier)
                logger.info(f"daily check done, messages: {sent or ['none']}")

                # monthly evidence digest on the 1st (Build 3)
                if now.day == 1:
                    try:
                        from scripts.generate_digest import generate_digest
                        path = generate_digest(notifier=notifier)
                        logger.info(f"monthly digest -> {path}")
                    except Exception as e:
                        logger.error(f"digest generation failed: {e}")
        except Exception as e:
            logger.exception(f"companion tick failed: {e}")
        await asyncio.sleep(3600)


if __name__ == '__main__':
    asyncio.run(main())
