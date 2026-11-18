#!/usr/bin/env bash
# Manual launcher (systemd runs python -m advisor directly with the same env file).
#   deploy/start.sh               # TAK + meshbus
#   deploy/start.sh --simulate    # fake tactical picture
set -euo pipefail

cd /opt/tac-advisor
source .venv/bin/activate
[ -f /etc/tac-advisor.env ] && set -a && . /etc/tac-advisor.env && set +a

exec python -m advisor "$@"
