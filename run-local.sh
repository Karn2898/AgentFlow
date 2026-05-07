#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--cli" ]]; then
  exec python -m agentflow.cli chat
fi

exec python -m agentflow.cli ui