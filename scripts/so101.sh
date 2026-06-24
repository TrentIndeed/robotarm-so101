#!/usr/bin/env bash
# Launch the SO-101 interactive app (all modes in one menu).
# Bash / Git Bash equivalent of so101.ps1.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/Scripts/python.exe"           # Windows venv layout (Git Bash)
[ -x "$PY" ] || PY="$DIR/.venv/bin/python"   # POSIX venv layout
exec "$PY" -m so101 "$@"
