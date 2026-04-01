#!/usr/bin/env bash
# Install Tactical Advisor on the Pi
# Usage: sudo bash deploy/install.sh
set -euo pipefail

APP_DIR="/opt/tac-advisor"

echo "=== Tactical Advisor — Pi Install ==="

# 1. System packages
echo "[1/4] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip

# 2. Copy project files
echo "[2/4] Copying project to $APP_DIR..."
mkdir -p "$APP_DIR/models"
cp -r advisor requirements.txt deploy "$APP_DIR/"

# 3. Python venv + deps
echo "[3/4] Setting up Python environment..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

# 4. Systemd service
echo "[4/4] Installing systemd service..."
cp "$APP_DIR/deploy/tac-advisor.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable tac-advisor

chmod +x "$APP_DIR/deploy/start.sh"

# Drop pi-profiles configs if pi-profiles is installed
if [ -d "/opt/pi-profiles/profiles.d" ]; then
    echo "Detected pi-profiles — installing profile configs..."
    cp profiles.d/*.conf /opt/pi-profiles/profiles.d/
fi

echo ""
echo "=== Install complete ==="
echo ""
echo "Services:"
echo "  Tactical Advisor → tac-advisor.service (http://<pi-ip>:8090)"
echo ""
echo "Model setup:"
echo "  Place a GGUF model file in $APP_DIR/models/"
echo "  Recommended: qwen3-0.6b-q5_k_m.gguf (fast) or llama-3.2-3b (better reasoning)"
echo ""
echo "Usage:"
echo "  sudo systemctl start tac-advisor"
echo "  journalctl -u tac-advisor -f"
echo "  sudo pi-profiles tac-advisor          # standalone"
echo "  sudo pi-profiles bft-advisor          # run alongside BFT"