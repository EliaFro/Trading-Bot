# env_file.sh — loader only. ALL secrets live in .env (never here).
#
# Usage:   source env_file.sh
#
# This exports every variable defined in .env into the current shell.
# .env must be chmod 600 (owner-only). Never commit .env; commit .env.example.

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and fill in your values." >&2
    return 1 2>/dev/null || exit 1
fi

set -a
. ./.env
set +a
