#!/usr/bin/env bash
# Install Tactical Advisor on the Pi. Idempotent and non-interactive.
# Does NOT enable the service at boot — pi-profiles starts it.
# Usage: sudo bash deploy/install.sh
#        sudo INSTALL_LOCAL_LLM=0 bash deploy/install.sh   # skip llama-cpp-python (Gemini only)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
APP_DIR="/opt/tac-advisor"
DATA_DIR="/var/lib/tac-advisor"
ENV_FILE="/etc/tac-advisor.env"
INSTALL_LOCAL_LLM="${INSTALL_LOCAL_LLM:-1}"

echo "=== Tactical Advisor — Pi Install ==="

echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip
if [ "$INSTALL_LOCAL_LLM" = "1" ]; then
    # llama-cpp-python builds from source on arm64
    apt-get install -y -qq build-essential cmake
fi

echo "[2/5] Copying project to $APP_DIR..."
mkdir -p "$APP_DIR" "$DATA_DIR/models"
if [ "$REPO_DIR" != "$APP_DIR" ]; then
    rm -rf "$APP_DIR/advisor"
    cp -r "$REPO_DIR/advisor" "$REPO_DIR/requirements.txt" "$REPO_DIR/deploy" "$REPO_DIR/profiles.d" "$APP_DIR/"
fi
chmod +x "$APP_DIR/deploy/start.sh"
# Older installs kept models in /opt/tac-advisor/models
if [ -d "$APP_DIR/models" ]; then
    find "$APP_DIR/models" -maxdepth 1 -name '*.gguf' -exec mv -n {} "$DATA_DIR/models/" \;
fi

echo "[3/5] Setting up Python environment (llama-cpp-python can take 15+ min to build)..."
[ -x "$APP_DIR/.venv/bin/python" ] || python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
if [ "$INSTALL_LOCAL_LLM" = "1" ]; then
    "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q
else
    grep -v '^llama-cpp-python' "$APP_DIR/requirements.txt" | "$APP_DIR/.venv/bin/pip" install -q -r /dev/stdin
fi

echo "[4/5] Config..."
if [ ! -f "$ENV_FILE" ]; then
    install -m 600 "$REPO_DIR/deploy/tac-advisor.env.example" "$ENV_FILE"
    echo "Created $ENV_FILE (add GEMINI_API_KEY there for cloud mode)"
fi

echo "[5/5] Installing systemd service (not enabled at boot — pi-profiles starts it)..."
cp "$REPO_DIR/deploy/tac-advisor.service" /etc/systemd/system/
systemctl daemon-reload

if [ -d "/opt/pi-profiles/profiles.d" ]; then
    cp "$REPO_DIR/profiles.d/"*.conf /opt/pi-profiles/profiles.d/
    echo "Installed pi-profiles profile: tac-advisor"
fi

echo ""
echo "=== Install complete ==="
echo "  Web UI:  http://<pi-ip>:8090"
echo "  Start:   sudo pi-profiles tac-advisor     (or: sudo systemctl start tac-advisor)"
echo "  Logs:    journalctl -u tac-advisor -f"
echo ""
echo "LLM: offline model -> put a GGUF file in $DATA_DIR/models/"
echo "     e.g. qwen3-0.6b-q5_k_m.gguf (fast) or a 3B Q4 model (better, slower)"
echo "     cloud  -> set GEMINI_API_KEY in $ENV_FILE (local model stays the offline fallback)"
