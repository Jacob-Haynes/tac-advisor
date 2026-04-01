# Tactical Advisor

Local LLM tactical advisor for Raspberry Pi 5. Reads CoT from FreeTAKServer, builds tactical context, answers queries via web UI and Meshtastic.

## Architecture

```
FreeTAKServer ←(TCP 8087)→ CoT Listener → Context Builder
                                                ↓
Meshtastic ←(serial)→ Query Handler ← LLM Engine ← GGUF Model
                                                ↓
                                         Web UI (port 8090)
```

## Hardware

- Raspberry Pi 5
- GGUF model file (Qwen3 0.6B recommended for speed, Llama 3.2 3B for quality)
- Optional: Meshtastic radio for mesh queries

## Install

```bash
sudo bash deploy/install.sh
```

Then place a GGUF model in `/opt/tac-advisor/models/`.

## Usage

```bash
# Standalone
sudo pi-profiles tac-advisor

# Combined with BFT
sudo pi-profiles bft-advisor

# Manual (simulation mode — no FTS or model needed for UI testing)
source .venv/bin/activate
python -m advisor --simulate
```

## Web UI

Browse to `http://<pi-ip>:8090` for the query interface. Quick buttons for SITREP, threat assessment, unit dispersion, and drone threat summaries.

## Meshtastic Queries

Send a text message starting with `ADVISOR:` from any mesh node:

```
ADVISOR: give me a SITREP
ADVISOR: threat assessment
```

Response sent back as a mesh text message (truncated to 200 chars).

## Model Setup

Download a GGUF model and place it in the `models/` directory:

- **Qwen3 0.6B** (`qwen3-0.6b-q5_k_m.gguf`) — fast, ~400MB, good for basic queries
- **Llama 3.2 3B** — better reasoning, ~2GB, slower on Pi