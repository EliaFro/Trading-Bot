#!/bin/bash
# Install (or remove) the ML learning lab as macOS launchd agents:
# unattended operation — starts at login, auto-restarts on crash.
#
#   bash scripts/install_service.sh            # install + start both agents
#   bash scripts/install_service.sh uninstall  # stop + remove both agents
#
# No Docker required. Bot and dashboard run natively from .venv.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
UID_N="$(id -u)"
BOT_LABEL="com.tradingbot.paper"
DASH_LABEL="com.tradingbot.dashboard"
FAST_LABEL="com.tradingbot.fastlab"
PLAYBOOK_LABEL="com.tradingbot.playbook"

make_plist() {
    local label="$1"; shift
    local logname="$1"; shift
    local plist="$AGENTS_DIR/$label.plist"
    mkdir -p "$AGENTS_DIR" "$REPO/logs"
    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>ProgramArguments</key>
    <array>
$(for arg in "$@"; do echo "        <string>$arg</string>"; done)
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict><key>SuccessfulExit</key><false/></dict>
    <key>ThrottleInterval</key><integer>30</integer>
    <key>StandardOutPath</key><string>$REPO/logs/$logname.out.log</string>
    <key>StandardErrorPath</key><string>$REPO/logs/$logname.err.log</string>
</dict>
</plist>
EOF
    echo "wrote $plist"
}

boot_out() {
    launchctl bootout "gui/$UID_N/$1" 2>/dev/null || true
}

if [[ "${1:-}" == "uninstall" ]]; then
    boot_out "$BOT_LABEL"
    boot_out "$DASH_LABEL"
    boot_out "$FAST_LABEL"
    boot_out "$PLAYBOOK_LABEL"
    rm -f "$AGENTS_DIR/$BOT_LABEL.plist" "$AGENTS_DIR/$DASH_LABEL.plist" \
          "$AGENTS_DIR/$FAST_LABEL.plist" "$AGENTS_DIR/$PLAYBOOK_LABEL.plist"
    echo "Agents stopped and removed."
    exit 0
fi

[[ -x "$REPO/.venv/bin/python" ]] || { echo "ERROR: .venv missing — run 'make install' first"; exit 1; }

# never run two bots against one database
pkill -f "src/main.py --mode paper" 2>/dev/null && sleep 3 || true
pkill -f "scripts/fastlab_bot.py" 2>/dev/null && sleep 2 || true

make_plist "$BOT_LABEL" "launchd_bot" \
    "$REPO/.venv/bin/python" "src/main.py" "--mode" "paper"
make_plist "$DASH_LABEL" "launchd_dashboard" \
    "$REPO/.venv/bin/python" "-m" "streamlit" "run" "src/dashboard/app.py" \
    "--server.port" "8501" "--server.address" "127.0.0.1" \
    "--server.headless" "true" "--logger.level" "error" \
    "--server.fileWatcherType" "none"
make_plist "$FAST_LABEL" "launchd_fastlab" \
    "$REPO/.venv/bin/python" "scripts/fastlab_bot.py"
make_plist "$PLAYBOOK_LABEL" "launchd_playbook" \
    "$REPO/.venv/bin/python" "scripts/playbook_companion.py"

boot_out "$BOT_LABEL"
boot_out "$DASH_LABEL"
boot_out "$FAST_LABEL"
boot_out "$PLAYBOOK_LABEL"
launchctl bootstrap "gui/$UID_N" "$AGENTS_DIR/$BOT_LABEL.plist"
launchctl bootstrap "gui/$UID_N" "$AGENTS_DIR/$DASH_LABEL.plist"
launchctl bootstrap "gui/$UID_N" "$AGENTS_DIR/$FAST_LABEL.plist"
sleep 2
launchctl bootstrap "gui/$UID_N" "$AGENTS_DIR/$PLAYBOOK_LABEL.plist"

echo
echo "Installed. The lab now starts at login and restarts itself on crashes."
echo "  bot health:  http://localhost:8080/health"
echo "  dashboard:   http://localhost:8501"
echo "  stop it all: bash scripts/install_service.sh uninstall"
