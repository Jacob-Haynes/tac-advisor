# Tactical Advisor

LLM tactical advisor for Raspberry Pi 5. It reads the CoT picture from OpenTAKServer (or any TAK server with a plain TCP CoT port) and builds a text context. It answers queries from a web UI, from `ADVISOR:` mesh texts and from ATAK chat (GeoChat). The LLM can be:
- **Ollama on the HQ Mac** over the hotspot (best answers),
- **Google Gemini** (cloud, needs internet),
- **a local GGUF model** on the Pi (llama.cpp, CPU only, always available offline),
- **the offline template SITREP**: no model at all. It counts the picture (positions, alerts by type, drone and sensor detections by source, last-seen times) and is always the last resort.

So a query always gets an answer: setup and error messages go to the log, never to the mesh or the user.

## Architecture

```
OpenTAKServer ←(TCP 8087, CoT)→ CoT Listener → Context Builder ─┐
   ↑ ATAK PLI/chat, drone-detector,                              ↓
   ↑ acoustic, op-sentry, change-detect, rf-spectrum      LLM Engine ── Gemini (if GEMINI_API_KEY)
                                                                  │  ├─ local GGUF (offline fallback)
                                                                  │  └─ template SITREP (no model, last resort)
mesh-bridge ←(MQTT meshbus/rx, meshbus/tx)→ Query Handler ───────┤
                                                                  ↓
                                                         Web UI (port 8090)
```

- **TAK.** The listener uses `advisor/tak_client.py` (kit contract v5). 3 s after each connect it registers as `isrkit-tac-advisor-<ISR_HOST>`, callsign `ADVISOR <ISR_HOST>`, and re-sends that every 60 s. Real OpenTAKServer on PostgreSQL drops CoT from a sender with no EUD row, only relays to a connection that registered after OTS's internal channel opened, and needs unique callsigns. Without a `SENSOR_LAT/LON` the registration is already stale, so ATAK never draws it. Other kit tools' registrations (`isrkit-…` uids) count as sensor stations, not friendly units. It reconnects with backoff if OTS is down. `ADVISOR_CALLSIGN` overrides the callsign, which must stay unique per station.
- **Picture.** Friendly PLI and chat, emergencies, UAVs from any source, and detections from the ISR kit's sensor tools (acoustic-detector, op-sentry, change-detect, rf-spectrum and fpv-vision; those repos are private) all go into the context. Sensor tools are recognised by uid prefix (`ACOUSTIC-`, `opsentry-`, `change-detect-`, `RFSPEC-`, `fpv-vision-`, `DRONE-`) or by naming themselves in the remarks. A SITREP therefore says which sensor reported what. A sensor's own position marker counts as a sensor, not a friendly unit. `GET /api/context` shows the exact text the LLM gets.
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
| `LLM_BACKEND` | `auto` | `auto`: every configured backend in order (LLM server, then Gemini, then local, then template). `openai_compat`: LLM server, then local, then template. `ollama` / `openai`: like `openai_compat`, forcing that request shape. `gemini`: Gemini, then local, then template. `local`: local, then template; never uses the network. `template`: template SITREP only |
| `LLM_URL` | empty | LLM server **base URL, no path**, e.g. `http://isr-hq.local:11434`. See `LLM_API` for which request is sent |
| `LLM_MODEL` | `qwen3.5:27b` | Model name on that server |
| `LLM_TIMEOUT_S` | `90` | Per-request timeout. The first query loads the model into memory |
| `LLM_API` | `auto` | `ollama`: Ollama's native `POST {LLM_URL}/api/chat` with `think: false` (qwen3.5 answers in seconds instead of thinking for over a minute). `openai`: OpenAI `POST {LLM_URL}/v1/chat/completions` (llama.cpp server, LM Studio, vLLM). Against Ollama it also sends `reasoning_effort: "none"`, because Ollama ignores `think` on that endpoint. `auto`: Ollama native when the server is Ollama (port 11434, "ollama" in the host, or `GET /api/version` answers), otherwise OpenAI; on 404/405 or the wrong response shape it tries the other once. Replies are capped at `LLM_MAX_TOKENS` (512). The UI and heartbeat show what answered, e.g. `ollama:qwen3.5:27b` |
| `LLM_KEEP_ALIVE` | `30m` | Asks Ollama to keep the model loaded between queries |
| `GEMINI_API_KEY` | empty | Enables cloud mode |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model name |
| `GEMINI_TIMEOUT_S` | `20` | Per-request timeout |
| `GRID_FORMAT` | `mgrs` | How positions appear in the LLM context: `mgrs` (`30U WB 87427 67248`), `bng` (OS National Grid, `SU 17582 39000`, falls back to MGRS outside GB) or `latlon`. Converted in `advisor/grid.py` (pure Python, no extra deps). The model only ever sees the converted ref, never raw lat/lon, and is told to copy refs verbatim. Each item also gets distance and direction from the nearest friendly unit, else from the OP (`SENSOR_LAT/LON`), e.g. `1.3 km NW of HANGOVER` |
| `ADVISOR_MODEL` | auto | GGUF filename in the models dir (first `.gguf` found otherwise) |

If a network backend fails (Mac asleep or off the hotspot, no internet, bad key), the next one answers. Then the local GGUF model, and finally the template SITREP. A failed backend is skipped for 2 minutes, so the Pi doesn't wait for a timeout on every query. The UI status and `GET /api/status` show which backend answered (`template:sitrep` for the template). With nothing configured, the UI still runs and queries get the template SITREP; the setup hint is in the log.

### Counts the model can't drop

The context starts with **PICTURE COUNTS**: one line per category in the picture (drones, people/vehicles, RF, acoustic, change, other sensors, unit positions, alerts, messages), counted the same way as the template SITREP and with the SIMULATED/RECORDED labels. The system prompt requires every line in the answer. If a model still leaves a category out (in rehearsal, qwen3.5:27b dropped 5 drone detections), the advisor adds "Counts: …" for the missing ones. On the mesh it shortens the model's text to make room within 200 characters.

### Template SITREP (offline)

Built straight from the CoT picture, with no model. On the mesh it fits in one text:

```
SITREP 1522Z (offline, no LLM); 5 units, 5 active; 2 simulated drone detections (drone-detector 2); 3 sensor detections (1 recorded) (op-sentry 2, fpv-vision 1); last 12s ago
```

The web UI gets a longer version with one line each for units, alerts, drones and sensors.

### Simulated and recorded data

Sibling tools label exercise data (kit contract C4.3): CoT remarks start `SIMULATED: ` or `RECORDED: `, or callsigns start `SIM-`. The advisor tags those items `[SIMULATED]` / `[RECORDED]` in the LLM context, adds a count line ("2 simulated drone detections") and tells the model to say so. The template SITREP counts them the same way. `--simulate` labels all its own fake data as simulated.

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
   Together that is about 26 GB. On a 48 GB Mac, macOS lets the GPU use about 36 GB of memory by default, so both models plus their context are expected to fit. This has not been verified on that Mac yet. Ollama keeps up to 3 models loaded at once by default.

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

## ATAK Chat Queries

From ATAK on the TAK server (Wi-Fi or hotspot), send `ADVISOR: SITREP` (or `@ADVISOR sitrep`, any case) to **All Chat Rooms**, or a direct message to the contact `ADVISOR <host>`. The advisor answers in the same place: in All Chat Rooms, or as a DM back to you. The reply is the full answer, not trimmed to 200 characters. OpenTAKServer 1.8.3 stores chat text in a 255-character column and drops anything longer, so a long answer arrives as several messages, `SITREP 1/5 …`, each at most 250 characters, split at line or sentence breaks. It comes from its registered identity (`isrkit-tac-advisor-<ISR_HOST>`, callsign `ADVISOR <ISR_HOST>`).

It ignores its own messages, other kit tools, and chat without the prefix. GeoChat that OTS made from a mesh text (`<contact endpoint="MQTT">`) is left to the mesh handler, which answers over the radio. The same query from the same sender arriving over both mesh and chat within 30 s is answered once. `GEOCHAT=0` turns chat queries off. Code: `advisor/geochat.py`.

## Model Setup

Download a GGUF model into `/var/lib/tac-advisor/models/`:

- **Qwen3 0.6B** (`qwen3-0.6b-q5_k_m.gguf`): fast, ~500 MB, fine for short answers
- **A 3B model at Q4** (e.g. Llama 3.2 3B): better reasoning, ~2 GB, several seconds per answer on a Pi 5

## Tests

```bash
pip install -r requirements.txt pytest httpx   # llama-cpp-python is not needed for the tests
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
