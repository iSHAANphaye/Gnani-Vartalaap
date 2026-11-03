# 🛡️ 3. Security & Access Document — Project Vartalaap

## 1. Authentication & API Security

### Internal vs. External Credentials
1. **User Authentication:**
   - For the hackathon demonstration and live prototype, Vartalaap operates with **Ephemeral Session Tokens**. When a user connects to the web interface, the server generates a cryptographically secure random session ID (`sess_<uuid4>`).
   - No persistent passwords or personal identifiable information (PII) are stored.
2. **Gnani Cloud API Authentication:**
   - All outgoing requests to Gnani’s sovereign speech endpoints (`api.vachana.ai`) require the HTTP header:
     `X-API-Key-ID: <GNANI_API_KEY>`
   - The API key is stored securely in server-side environment variables (`.env`) and is **strictly never exposed or streamed to the client-side JavaScript bundle**. All calls to Gnani are proxied through the FastAPI backend server.

---

## 2. User Roles & Permissions Matrix

Because Vartalaap has an interactive live evaluation mode and benchmark runners, access is partitioned into three functional tiers:

| Role | Access Level | Allowed Actions | Blocked Actions |
|---|---|---|---|
| **Evaluator / User (Default)** | Public / Client Session | Start/stop voice interaction, speak into microphone, view live telemetry HUD, toggle between Standard Bot and Vartalaap modes. | Cannot access raw server logs, modify engine silence constants, or view backend API keys. |
| **Benchmarker / Developer** | Localhost / Internal | Trigger batch test scripts (`benchmark/run_benchmark.py`), upload pre-recorded WAV test clips, download benchmark metrics JSON. | Cannot expose private API keys in git commits or unauthenticated endpoints. |
| **System Daemon (Vartalaap Engine)** | High-Privilege Backend | Open WebSockets to Gnani Prisma STT, pipe raw audio chunks to Silero VAD, query RTX 5060 GPU tensor pipelines. | Blocked from persisting unencrypted user audio to public disk directories. |

---

## 3. Data Privacy & Compliance (DPDP Act 2023 Principles)

In alignment with India's **Digital Personal Data Protection (DPDP) Act** and Gnani's sovereign AI mission:
1. **Zero Permanent Audio Retention:**
   - Raw microphone PCM audio buffers exist solely in transient memory (`RAM`) during the active turn window.
   - Once a turn is resolved and transcribed, the raw PCM audio chunks in the ring buffer are immediately flushed and garbage-collected.
2. **Ephemeral Transcripts:**
   - Text transcripts are displayed in the user's browser for the duration of the browser tab. Upon closing the tab or clicking `Reset Session`, conversation state is wiped.
3. **No External Third-Party Tracking:**
   - The web HUD contains zero Google Analytics, Meta Pixels, or third-party telemetry beacons.

---

## 4. Comprehensive Error Handling Guide

| Failure Point | Root Cause | System Response / User Feedback | Graceful Degradation / Recovery |
|---|---|---|---|
| **Microphone Access Denied** | User blocks browser audio permissions or has no mic connected. | Modal Alert: *"Microphone access required. Please allow audio permissions in your browser bar."* | Disable `Start Call` button; provide fallback audio file upload mode so evaluators can still test pre-recorded clips. |
| **Gnani STT WebSocket Disconnect** | Network timeout or temporary outage on `api.vachana.ai`. | Telemetry HUD badge flashes red: `STT_STREAM_INTERRUPTED`. | Automatic exponential backoff reconnection within 500ms; if offline, fall back to Gnani REST STT or mock intent engine. |
| **Gnani API Quota Exceeded (HTTP 429)** | Exceeded free platform credits (5,000 credits). | Toast Notification: *"Gnani credit ceiling reached. Engine entering local fallback mode."* | Gracefully switch to simulated text responses so turn-detection logic and UI telemetry remain fully demonstrable. |
| **GPU Out of Memory (OOM)** | RTX 5060 VRAM spikes from concurrent model loading. | Log warning: `CUDA OOM. Evicting tensor cache.` | Automatically failover to CPU inference mode for the semantic rule classifier (<20ms). |
| **Corrupted / Zero-Byte Audio Packets** | Client sends malformed binary data over WebSocket. | Server drops frame silently; increment error metric counter. | Prevent crash; maintain existing audio buffer history. |

---

## 5. Edge Cases & Resilience Engineering

1. **The "Infinite Monologue" Edge Case:**
   - *Scenario:* User speaks continuously for 60 seconds without pausing.
   - *Handling:* Vartalaap enforces a `MAX_TURN_DURATION_SEC = 45.0`. At 45 seconds, the engine generates a soft turn-break event, forcing interim sentence processing so the bot does not buffer memory indefinitely.
2. **Background Environmental Noise (Traffic, Fans, Household):**
   - *Scenario:* Ambient street noise or ceiling fan sound triggers false VAD speech events.
   - *Handling:* Silero VAD is configured with `threshold=0.5` + a high-pass frequency filter. Frames with low speech probability are discarded before being sent to Gnani STT, preserving API credits.
3. **Simultaneous Speech / Audio Overlap (Barge-in):**
   - *Scenario:* User speaks while the bot's speaker is playing.
   - *Handling:*
     - If the incoming tokens match the **Cultural Backchannel Lexicon** (*"haanji"*, *"achha"*, *"seri"*, *"theek hai"*), barge-in is **suppressed**. The bot keeps talking.
     - If the user speaks an imperative command (*"Ruko"*, *"Cancel karo"*, *"Wait"*), an immediate `FLUSH_AUDIO_BUFFER` signal is sent to the client in `<60ms`, stopping playback instantly.
4. **Empty or Whitespace-Only Transcripts:**
   - *Scenario:* User coughs or taps microphone; VAD detects energy but STT yields no words.
   - *Handling:* Turn engine checks `len(interim_transcript.strip()) > 0`. If empty, the silence timer does not trigger a bot response; system reverts to `IDLE_LISTENING`.
