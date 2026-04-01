#!/usr/bin/env bash
set -euo pipefail

cd /opt/tac-advisor
source .venv/bin/activate

ARGS=""

# Check for Meshtastic device
if [ ! -e /dev/ttyUSB0 ] && [ ! -e /dev/ttyACM0 ]; then
    ARGS="$ARGS --no-mesh"
fi

exec python -m advisor $ARGS