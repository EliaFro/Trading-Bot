# AI Crypto Trading System
# Usage: make help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup install db-init db-reset backfill run dashboard test stop \
        status kill-switch clear-kill-switch logs clean docker-build docker-up docker-down

help:
	@echo "AI Crypto Trading System"
	@echo ""
	@echo "First-time setup:"
	@echo "  make setup            venv (Python 3.11) + deps + database"
	@echo "  make backfill         download 12 months of market history"
	@echo ""
	@echo "Running:"
	@echo "  make run              start the bot (PAPER mode)"
	@echo "  make dashboard        start the Streamlit dashboard (:8501)"
	@echo "  make stop             stop bot and dashboard"
	@echo "  make status           engine status from the health endpoint"
	@echo ""
	@echo "Safety:"
	@echo "  make kill-switch          show kill-switch state"
	@echo "  make clear-kill-switch    clear it after review"
	@echo ""
	@echo "Development:"
	@echo "  make test             run the test suite"
	@echo "  make logs             tail today's log"
	@echo "  make db-reset         DROP and recreate the database"

setup: install db-init
	@echo "Setup complete. Next: make backfill && make run"

install:
	@test -d .venv || /opt/homebrew/bin/python3.11 -m venv .venv
	@$(PIP) install --upgrade pip -q
	@$(PIP) install -r requirements.txt -q
	@echo "Dependencies installed ($$($(PYTHON) --version))"

db-init:
	@$(PYTHON) scripts/init_db.py

db-reset:
	@$(PYTHON) scripts/init_db.py --reset

backfill:
	@$(PYTHON) scripts/backfill.py --months 12

run:
	@$(PYTHON) src/main.py --mode paper

run-live:
	@echo "⚠️  LIVE TRADING — real money. Ctrl+C within 5s to cancel."
	@sleep 5
	@$(PYTHON) src/main.py --mode live

dashboard:
	@.venv/bin/streamlit run src/dashboard/app.py --server.port 8501

stop:
	@pkill -f "src/main.py" 2>/dev/null || true
	@pkill -f "streamlit run" 2>/dev/null || true
	@echo "Stopped."

status:
	@curl -s http://localhost:8080/status | $(PYTHON) -m json.tool || echo "Engine not running"

kill-switch:
	@$(PYTHON) scripts/clear_kill_switch.py

clear-kill-switch:
	@$(PYTHON) scripts/clear_kill_switch.py --clear

test:
	@$(PYTHON) -m pytest tests/ -v

logs:
	@tail -f logs/trading.log

clean:
	@find . -name "__pycache__" -type d -not -path "./.venv/*" -not -path "./_quarantine/*" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache
	@echo "Cleaned."

docker-build:
	@docker compose build

docker-up:
	@docker compose up -d
	@echo "Dashboard: http://localhost:8501"

docker-down:
	@docker compose down
