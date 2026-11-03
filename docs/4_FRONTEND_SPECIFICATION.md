# 🎨 4. Frontend Specification Document — Project Vartalaap

## 1. Visual Theme & Design Philosophy

The Vartalaap Telemetry HUD is designed as a mission-control, high-fidelity developer cockpit. It uses a modern dark aesthetic inspired by audio engineering DAWs (Digital Audio Workstations) and live observability dashboards (Datadog/Grafana), making the internal state of conversational turn-taking visually intuitive for judges.

---

## 2. Color Palette & Design Tokens

| Token Name | Hex Code | Visual Swatch / Role |
|---|---|---|
| **Background Dark** | `#0b0f19` | Main application canvas background |
| **Surface Dark (Card/Panel)** | `#111827` | Elevated card container background |
| **Surface Border** | `#1f2937` | Subtle divider lines and card borders |
| **Primary Accent (Indigo/Blue)** | `#3b82f6` | Active buttons, primary sliders, waveforms |
| **Listening Active (Emerald Green)** | `#10b981` | VAD active, user speech detected |
| **Thinking / Holding (Amber/Orange)** | `#f59e0b` | Conjunction or hesitation pause detected (timer extended) |
| **Backchannel Ignored (Cyan)** | `#06b6d4` | Cultural acknowledgment token detected; barge-in suppressed |
| **Bot Speaking (Violet/Purple)** | `#8b5cf6` | Bot audio active, TTS streaming |
| **Text Primary** | `#f9fafb` | Primary titles, active transcript text |
| **Text Secondary** | `#9ca3af` | Telemetry labels, millisecond counters |
| **Danger / Alert** | `#ef4444` | Disconnected states, cutoff indicators |

---

## 3. Typography Scale

- **Primary Font Family:** `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif`
- **Monospace Font Family (Telemetry & Numbers):** `'JetBrains Mono', 'Fira Code', monospace`

| Level | Size | Weight | Line Height | Usage |
|---|---|---|---|---|
| **Display Title** | `24px (1.5rem)` | 700 (Bold) | `1.2` | App Header Title |
| **Section Heading** | `18px (1.125rem)` | 600 (Semi-bold) | `1.3` | Panel Titles ("Live Turn State", "Transcript Stream") |
| **Body Standard** | `14px (0.875rem)` | 400 (Regular) | `1.5` | Transcripts, status explanations |
| **Telemetry Metric** | `20px (1.25rem)` | 700 (Bold, Mono) | `1.0` | Live timer countdown (`1420ms / 1800ms`), TTFA gauge |
| **Badge / Label** | `11px (0.6875rem)` | 600 (Semi-bold, Mono) | `1.0` | Status pills (`HOLDING_FLOOR`, `BACKCHANNEL_IGNORED`) |

---

## 4. Component Styles & Layout Rules

### A. Real-Time Audio Waveform Visualizer
- **Container:** HTML5 `<canvas>` element (Full card width, height: 120px).
- **Behavior:** Renders smooth sinusoidal frequency/oscilloscope ripples using Web Audio API `AnalyserNode`.
- **Dynamic Color Shifts:**
  - Shifts to **Emerald Green** when User is speaking.
  - Shifts to **Amber** during thinking pauses (*"matlab..."*).
  - Shifts to **Purple** when Bot TTS is streaming back.

### B. Live Telemetry State Badges
- High-contrast pill badges with animated pulsing dot:
  - `[● LISTENING]` — Green pill (`bg-emerald-950/60 text-emerald-400 border border-emerald-800`).
  - `[● EXTENDING FLOOR (+1500ms)]` — Amber pill (`bg-amber-950/60 text-amber-400 border border-amber-800`).
  - `[● BACKCHANNEL: HAANJI DETECTED]` — Cyan pill (`bg-cyan-950/60 text-cyan-400 border border-cyan-800`).
  - `[● TURN COMPLETE (<300ms)]` — Blue pill (`bg-blue-950/60 text-blue-400 border border-blue-800`).

### C. Live Dynamic Timer Bar
- A progress bar showing elapsed silence vs. the dynamic threshold:
  - Fill animates from 0% to 100%.
  - When a hesitation word (*"matlab"*) is spoken, the bar instantly re-scales from 600ms max to 2000ms max, visually demonstrating the algorithm at work!

### D. A/B Mode Toggle Switch
- Pill-shaped switch at top right:
  - `Standard Voice Bot (500ms static)` $\longleftrightarrow$ `Vartalaap Indic Turn Engine`.
  - When in Standard mode, warning banner displays: *"Standard mode active: will cut off on thinking pauses."*

---

## 5. Client-Server API & Integration Specification

### A. Bi-Directional WebSocket Protocol
- **Endpoint:** `ws://localhost:8000/ws/audio`

#### 1. Client $\rightarrow$ Server Messages

##### Binary Frame (Raw Audio Stream)
- **Format:** Binary Blob / ArrayBuffer
- **Encoding:** 16-bit Linear PCM, 16,000 Hz, Mono, 512-sample or 1024-sample chunks (every 32ms–64ms).

##### JSON Control Message: Toggle Mode
```json
{
  "action": "SET_MODE",
  "mode": "VARTALAAP" // or "STANDARD"
}
```

##### JSON Control Message: End Conversation
```json
{
  "action": "STOP_CALL"
}
```

#### 2. Server $\rightarrow$ Client Messages

##### Interim Transcript Update
```json
{
  "type": "TRANSCRIPT_STREAM",
  "text": "Mera electricity bill pay karna hai matlab ki",
  "is_final": false,
  "language_detected": "hi-IN"
}
```

##### Live Telemetry Event
```json
{
  "type": "TELEMETRY",
  "speaker": "USER",
  "vad_active": false,
  "elapsed_silence_ms": 450,
  "current_threshold_ms": 1800,
  "state": "HOLDING_FOR_THOUGHT",
  "reason": "Cognitive filler 'matlab ki' identified",
  "backchannel_detected": null
}
```

##### Bot Audio Chunk (TTS Streaming)
```json
{
  "type": "AUDIO_CHUNK",
  "audio_base64": "UklGRiYAAABXQVZFZm10IBAAAA...",
  "format": "audio/wav",
  "is_final_chunk": false
}
```

##### Call Summary & Metrics
```json
{
  "type": "SESSION_SUMMARY",
  "duration_seconds": 34.2,
  "user_turns": 4,
  "cut_offs_prevented": 3,
  "backchannels_preserved": 2,
  "average_turn_latency_ms": 310
}
```

### B. HTTP REST Endpoints (Benchmarking & Fallback)
| Method | Endpoint | Description | Payload | Response |
|---|---|---|---|---|
| `GET` | `/api/health` | Healthcheck & API readiness | None | `{"status": "ok", "gnani_stt": true, "gnani_tts": true, "gpu": "RTX 5060"}` |
| `POST` | `/api/benchmark/upload` | Upload audio clip for batch A/B evaluation | Multipart form (`file: .wav`) | JSON comparison metrics |
| `GET` | `/api/benchmark/report` | Download latest benchmark evaluation report | None | JSON file download |
