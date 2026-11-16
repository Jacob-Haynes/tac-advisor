# Tactical Advisor

LLM tactical advisor for Raspberry Pi 5. It reads the CoT picture from OpenTAKServer (or any TAK server with a plain TCP CoT port) and builds a text context. It answers queries from a web UI and from `ADVISOR:` mesh texts. The LLM can be:
- **Ollama on the HQ Mac** over the hotspot (best answers),
- **Google Gemini** (cloud, needs internet),
- **a local GGUF model** on the Pi (llama.cpp, CPU only, always available offline).

The local model is the fallback for both.

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
| `LLM_BACKEND` | `auto` | `auto`: every configured backend in order (LLM server, then Gemini, then local). `openai_compat`: LLM server, then local. `gemini`: Gemini, then local. `local`: never use the network |
| `LLM_URL` | empty | LLM server **base URL, no path**, e.g. `http://isr-hq.local:11434`. The advisor calls Ollama's `/api/chat` with thinking off. If the server has no `/api/chat` (LM Studio, llama.cpp server, vLLM), it uses `/v1/chat/completions` |
| `LLM_MODEL` | `qwen3.5:27b` | Model name on that server |
| `LLM_TIMEOUT_S` | `90` | Per-request timeout. The first query loads the model into memory |
| `LLM_API` | `auto` | Force `ollama` or `openai` |
| `LLM_KEEP_ALIVE` | `30m` | Asks Ollama to keep the model loaded between queries |
| `GEMINI_API_KEY` | empty | Enables cloud mode |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model name |
| `GEMINI_TIMEOUT_S` | `20` | Per-request timeout |
| `ADVISOR_MODEL` | auto | GGUF filename in the models dir (first `.gguf` found otherwise) |

If a network backend fails (Mac asleep or off the hotspot, no internet, bad key), the next one answers. The local GGUF model is the last resort. A failed backend is skipped for 2 minutes, so the Pi doesn't wait for a timeout on every query. The UI status and `GET /api/status` show which backend answered. With nothing configured, the UI still runs and queries return a setup message.

## HQ Mac: Ollama Setup (macOS, Apple Silicon)

The HQ Mac (M4 Max, 48 GB, `isr-hq.local`) runs the LLM for the advisor and a vision model for the other tools.

1. Install Ollama from <https://ollama.com/download> and start the app. qwen3-vl needs Ollama 0.12.7 or newer.
2. Let the Pi reach it. By default Ollama only listens on localhost:
   ```bash
   launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
   ```
   Then quit and reopen the Ollama app (from the Ollama FAQ). `launchctl setenv` does not survive a reboot, so run it again after restarting the Mac. If macOS asks whether Ollama may accept incoming connections, allow it.
3. Pull the models:
   ```bash
   ollama pull qwen3.5:27b    # advisor LLM: 17-20 GB, 256K context, thinking (the advisor turns it off)
   ollama pull qwen3-vl:8b    # vision model for the other tools: 6.2 GB
   ```
   Together that is about 26 GB. On a 48 GB Mac, macOS lets the GPU use about 36 GB of memory by default, so both models plus their context are expected to fit. This has not been tested on the HQ Mac yet; the platform session will confirm it. Ollama keeps up to 3 models loaded at once by default.

   If they don't fit (Ollama unloads one model to load the other, or answers are very slow), use a smaller advisor model: `qwen3.5:9b` (6.6-7.6 GB, about 14 GB with qwen3-vl:8b). Pull it and set `LLM_MODEL=qwen3.5:9b` in `/etc/tac-advisor.env` on the Pi. Faster but larger alternative: `qwen3.5:35b` (22 GB, a mixture-of-experts model, quicker per token; about 28 GB with qwen3-vl:8b, which leaves less room for context). On the 32 GB Drone-ops Mac use `qwen3.5:9b`. Sizes are from ollama.com/library, checked 2026-11-16.
4. Check from the Pi:
   ```bash
   curl http://isr-hq.local:11434/api/tags
   ```
5. In `/etc/tac-advisor.env` on the Pi set `LLM_URL=http://isr-hq.local:11434`, then `sudo systemctl restart tac-advisor`. The first SITREP takes longer while the model loads.

Keep the Mac awake and on mains power during the demo. If it sleeps, the advisor silently falls back to the Pi's local model, and the UI status shows that.

## Running the Advisor on a Mac

The advisor can also run on a Mac on the hotspot:

```bash
TAK_HOST=isr-pi.local MQTT_HOST=isr-pi.local MQTT_USER=isrkit MQTT_PASS=<isrkit password> \
  LLM_URL=http://localhost:11434 python -m advisor
```

RabbitMQ's `guest` account only works from the Pi itself. Off the Pi, use the `isrkit` user that the platform creates.

## Heartbeat

Every 10 s the advisor publishes a retained JSON message to `isrkit/heartbeat/tac-advisor` for the launcher:
- `status`: `ok`, or `degraded` when TAK, the LLM or the mesh is missing (the reason is in `detail`);
- `mode`: `llm` when the LLM server (Ollama on the HQ Mac) is answering, `gemini` for Gemini, `cpu` for the Pi's local model, `simulate` in simulation. The backend and model are in `detail`.

It publishes `stopped` on a clean exit. `HEARTBEAT=0` turns it off.

## Usage

```bash
# On the Pi (brings up opentakserver + mesh-bridge + tac-advisor)
sudo pi-profiles tac-advisor

# Manual
source .venv/bin/activate
python -m advisor --simulate             # fake picture incl. sensor events, no TAK server
python -m advisor --backend local        # force the offline model
LLM_URL=http://isr-hq.local:11434 python -m advisor --simulate   # Mac LLM with a fake picture
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
