# Tactical Advisor

LLM tactical advisor for Raspberry Pi 5. It reads the CoT picture from OpenTAKServer (or any TAK server with a plain TCP CoT port) and builds a text context. It answers queries from a web UI and from `ADVISOR:` mesh texts. The LLM is either a local GGUF model (llama.cpp, CPU only, works offline) or Google Gemini (optional cloud mode). The local model is the offline fallback.

## Architecture

```
OpenTAKServer ←(TCP 8087, CoT)→ CoT Listener → Context Builder ─┐
   ↑ ATAK PLI/chat, drone-detector,                              ↓
   ↑ acoustic, op-sentry, change-detect, rf-spectrum      LLM Engine ── Gemini (if GEMINI_API_KEY)
                                                                  │  └─ local GGUF (offline fallback)
mesh-bridge ←(MQTT meshbus/rx, meshbus/tx)→ Query Handler ───────┤
                                                                  ↓
                                                         Web UI (port 8090)
```

- **TAK.** OpenTAKServer only streams CoT to a TCP client after that client sends an event with a uid and `<contact callsign>`. The listener sends a small "hello" SA on connect and every 60 s. Its stale time is already reached, so ATAK does not draw it. It reconnects with backoff if OTS is down.
- **Picture.** Friendly PLI and chat, emergencies, UAVs from any source, and detections from the ISR kit's sensor tools all go into the context. Sensor tools are recognised by uid prefix (`ACOUSTIC-`, `opsentry-`, `change-detect-`, `RFSPEC-`, `fpv-vision-`, `DRONE-`) or by naming themselves in the remarks. A SITREP therefore says which sensor reported what. A sensor's own position marker counts as a sensor, not a friendly unit. `GET /api/context` shows the exact text the LLM gets.
- **Mesh.** The radio belongs to [blufor-tracker](https://github.com/Jacob-Haynes/blufor-tracker)'s `mesh-bridge`. The advisor reads `ADVISOR:` texts from MQTT `meshbus/rx` and replies to the sender on `meshbus/tx` (127.0.0.1:1883). `--serial [PORT]` opens a radio directly for standalone use. Never do that while `mesh-bridge` is running.

## Hardware

- Raspberry Pi 5 (no accelerator needed)
- A GGUF model file for offline use (Qwen3 0.6B for speed, a 3B Q4 model for quality), and/or a Gemini API key
- Optional: the mesh bridge for mesh queries

## Install

```bash
sudo bash deploy/install.sh                       # includes llama-cpp-python (builds from source, slow)
sudo INSTALL_LOCAL_LLM=0 bash deploy/install.sh   # Gemini only
```

The install is idempotent and does not enable the service at boot; pi-profiles starts it. Layout:

| What | Where |
|---|---|
| Code + venv | `/opt/tac-advisor`, `/opt/tac-advisor/.venv` |
| GGUF models | `/var/lib/tac-advisor/models/` |
| Config and secrets | `/etc/tac-advisor.env` (mode 600, from `deploy/tac-advisor.env.example`) |

## LLM Backend

Set in `/etc/tac-advisor.env`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `auto` | `auto`: Gemini if `GEMINI_API_KEY` is set, else local. `gemini`: cloud first. `local`: never use the network |
| `GEMINI_API_KEY` | empty | Enables cloud mode |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model name |
| `GEMINI_TIMEOUT_S` | `20` | Per-request timeout |
| `ADVISOR_MODEL` | auto | GGUF filename in the models dir (first `.gguf` found otherwise) |

If a Gemini call fails (no internet, bad key, quota), the answer comes from the local model. Gemini is then skipped for 2 minutes, so an offline Pi doesn't wait for a timeout on every query. The UI status shows which backend answered. With neither a key nor a model, the UI still runs and queries return a setup message.

## Usage

```bash
# On the Pi (brings up opentakserver + mesh-bridge + tac-advisor)
sudo pi-profiles tac-advisor

# Manual
source .venv/bin/activate
python -m advisor --simulate             # fake picture incl. sensor events, no TAK server
python -m advisor --backend local        # force the offline model
python -m advisor --no-mesh
```

## Web UI

Browse to `http://<pi-ip>:8090` for the query interface. It has quick buttons for SITREP, threat assessment, unit dispersion and drone threat summaries.

## Mesh Queries

Send a text message starting with `ADVISOR:` from any mesh node:

```
ADVISOR: give me a SITREP
ADVISOR: any sensor activity?
```

The answer goes back to the sender as a mesh text under 200 characters. Queries are answered one at a time; if more than 5 are waiting, the sender gets "busy".

## Model Setup

Download a GGUF model into `/var/lib/tac-advisor/models/`:

- **Qwen3 0.6B** (`qwen3-0.6b-q5_k_m.gguf`): fast, ~500 MB, fine for short answers
- **A 3B model at Q4** (e.g. Llama 3.2 3B): better reasoning, ~2 GB, several seconds per answer on a Pi 5

## Tests

```bash
pip install -r requirements.txt pytest httpx   # llama-cpp-python is not needed for the tests
python -m pytest
```
