#!/bin/sh
# Container entrypoint: web app, plus the Telegram bot when a token is set.
# Single-container hosts (e.g. Render) can't share the corpus disk across
# services, so the bot runs alongside uvicorn here instead.
set -e

if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "start.sh: TELEGRAM_BOT_TOKEN set — starting telegram bot"
    python -m app.telegram &
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
