# 🏗️ 2. Technical Architecture Document — Project Vartalaap

## 1. System Architecture Overview

Vartalaap is built as a high-performance, low-latency asynchronous pipeline in Python, orchestrating browser audio streaming, acoustic Voice Activity Detection, cloud-based Indic speech services (Gnani.ai), and an on-device semantic turn evaluation engine running on an NVIDIA RTX 5060 GPU.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        BROWSER / CLIENT LAYER                          │
│  - Web Audio API (16kHz 16-bit Mono PCM Chunking)                      │
│  - WebSocket Client (Bi-directional Audio & State Telemetry)           │
│  - Visual HUD (Canvas Audio Visualizer + Telemetry State Display)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ WebSockets (ws://localhost:8000/ws)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FASTAPI APPLICATION SERVER                       │
│  - Asynchronous Event Loop (asyncio)                                   │
│  - Audio Ingestion & Buffer Management                                 │
└───────┬───────────────────────────┬────────────────────────────┬───────┘
        │                           │                            │
┌───────▼───────────────┐   ┌───────▼────────────────┐   ┌───────▼───────┐
│ Layer 1: Acoustic VAD │   │ Layer 2: Streaming ASR │   │ Layer 3: Turn │
│  - Silero VAD (ONNX)  │   │  - Gnani Prisma STT    │   │ Decision Hub  │
│  - Energy/Frame check │   │  - WebSocket Client    │   │  - State Mach │
│  - Latency: < 2ms     │   │  - Interim tokens      │   │  - Dynamic ms │
└───────────────────────┘   └────────────────────────┘   └───────┬───────┘
                                                                 │
                                ┌────────────────────────────────┴───────┐
                                │ Layer 4: Semantic Classifier (RTX 5060)│
                                │  - Indic Token Completeness Evaluator  │
                                │  - Cultural Backchannel Gate (Regex/POS│
                                │  - Latency: < 12ms                     │
                                └────────────────────────────────┬───────┘
                                                                 │ (Turn Complete)
┌────────────────────────────────────────────────────────────────▼───────┐
│ Layer 5: Conversational Brain & Voice Synthesis                        │
│  - Gnani Evon / Fast Indic LLM / Dialogue Engine                       │
│  - Gnani Timbre v3.0 Expressive TTS (WebSocket/SSE Chunk Streaming)    │
│  - Latency Budget: < 400ms TTFA                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Explicit Tech Stack & Version Specifications

| Component | Technology | Version | Architectural Reasoning |
|---|---|---|---|
| **Runtime Environment** | Python (via `uv`) | `3.11.17` | High-speed CPython runtime with optimized async event loop and native PyTorch CUDA 12 support. |
| **Package / Env Manager** | `uv` | `0.13.1` | Instantaneous dependency resolution, 10x faster than standard pip, reproducible lockfiles. |
| **Backend Framework** | FastAPI + Uvicorn | `fastapi >= 0.115`, `uvicorn >= 0.32` | Industry standard for high-concurrency async WebSockets and minimal overhead. |
| **Acoustic VAD** | Silero VAD | `silero-vad >= 5.1` | Ultra-lightweight pre-trained neural VAD running on CPU in <2ms per 30ms audio chunk. |
| **GPU Acceleration** | PyTorch + CUDA | `torch >= 2.5` (CUDA 12.4/12.6) | Native acceleration on the local **NVIDIA RTX 5060 (8GB VRAM)** for semantic token inference. |
| **Streaming Speech STT** | Gnani Prisma STT v2.5 | Cloud API (`api.vachana.ai`) | Sovereign Indic ASR natively trained on code-mixed telephony audio; 14 Indian languages. |
| **Expressive Speech TTS** | Gnani Timbre v3.0 | Cloud API (`api.vachana.ai`) | Indian language synthesis with 13 inline expressive audio tags (`<excited>`, `<pauses>`). |
| **Conversational Brain** | Gnani Evon / Inya API | Cloud API / OpenAI compatible | Indic-optimized foundational reasoning model with 38.6% fewer tokens for Indian scripts. |
| **Frontend UI** | HTML5 + Tailwind CSS + Vanilla JS | Modern ES6+ Web Audio API | Zero-build simplicity, instant browser reloading, native AudioContext PCM streamer. |

---

## 3. Complete File & Folder Structure

```
c:\Antigravity Dev\Gnani Hackathon\
├── .env                              # API credentials & port config (NEVER COMMITTED)
├── .env.example                      # Template environment file
├── .gitignore                        # Git exclusion rules
├── pyproject.toml                    # UV project configuration and dependencies
├── requirements.txt                  # Pinned dependencies
├── README.md                         # Public hackathon submission documentation
├── docs/                             # Standard pre-build specifications
│   ├── 1_PRD.md                      # Product Requirements Document
│   ├── 2_TECHNICAL_ARCHITECTURE.md   # This document
│   ├── 3_SECURITY_AND_ACCESS.md      # Security, roles, error handling
│   ├── 4_FRONTEND_SPECIFICATION.md   # UI design system & integration specs
│   └── 5_FEATURE_TICKETS.md          # Granular implementation tickets
├── app/                              # Core application source code
│   ├── __init__.py
│   ├── main.py                       # FastAPI entrypoint, HTTP routes, WebSocket handlers
│   ├── config.py                     # Pydantic BaseSettings for config management
│   ├── core/                         # Core Vartalaap Turn-Taking Engine
│   │   ├── __init__.py
│   │   ├── vad.py                    # Silero VAD wrapper & frame chunker
│   │   ├── state_machine.py          # Dynamic Turn State Machine (States, Timers)
│   │   ├── backchannel.py            # Cultural Backchannel Gate & Lexicons
│   │   └── semantic.py               # Indic Syntactic Completeness Evaluator (RTX 5060)
│   ├── services/                     # External API Clients
│   │   ├── __init__.py
│   │   ├── gnani_stt.py              # Gnani Prisma STT WebSocket client
│   │   ├── gnani_tts.py              # Gnani Timbre TTS streaming client
│   │   └── llm_service.py            # Dialogue / LLM response orchestrator
│   ├── telemetry/                    # Metrics, benchmark logger & analytics
│   │   ├── __init__.py
│   │   ├── metrics.py                # Latency, cut-off counter, barge-in stats
│   │   └── reporter.py               # Summary generator for demo reports
│   └── static/                       # Web HUD & Assets
│       ├── index.html                # Telemetry dashboard & live conversation HUD
│       ├── app.js                    # Web Audio recording, WebSocket & canvas visualizer
│       └── styles.css                # UI styles & dark theme tokens
└── benchmark/                        # Pre-recorded test audio & comparison tests
    ├── test_samples/                 # Hindi, Hinglish, Tamil test clips
    │   ├── hinglish_pause.wav
    │   ├── backchannel_haanji.wav
    │   └── clean_complete.wav
    └── run_benchmark.py              # Automated A/B benchmark comparing Standard vs Vartalaap
```

---

## 4. In-Memory Data Models & Schemas

Because Vartalaap is an ultra-low-latency real-time voice system, data persistence is maintained in fast in-memory async data structures, with optional session dumps to JSON.

### A. Conversation Turn State Model (`TurnContext`)
```json
{
  "session_id": "sess_98231afb_20261102",
  "current_speaker": "USER", // "USER" | "BOT" | "IDLE"
  "vad_active": true,
  "last_speech_timestamp": 1798889123.45,
  "silence_duration_ms": 420.0,
  "dynamic_silence_threshold_ms": 1800.0, // Dynamically assigned
  "interim_transcript": "Mera account balance matlab ki",
  "syntactic_status": "INCOMPLETE_HESITATION", // "COMPLETE" | "INCOMPLETE_HESITATION" | "HANGING_CONJUNCTION"
  "detected_filler_or_conjunction": "matlab ki",
  "is_backchannel": false,
  "turn_decision": "HOLD_FLOOR" // "HOLD_FLOOR" | "GRANT_TURN" | "SUPPRESS_BARGE_IN"
}
```

### B. Telemetry Event Model (Streamed to Browser HUD)
```json
{
  "type": "TELEMETRY_UPDATE",
  "timestamp": 1798889123.50,
  "event": "STATE_CHANGE",
  "details": {
    "state": "HOLDING_FOR_THOUGHT",
    "reason": "Cognitive filler 'matlab ki' detected; timer expanded to 1800ms",
    "elapsed_silence_ms": 420,
    "threshold_ms": 1800,
    "confidence": 0.94
  }
}
```

### C. Benchmark Run Record Model
```json
{
  "sample_id": "test_hinglish_thinking_pause_01",
  "audio_duration_sec": 5.4,
  "transcript": "Mera electricity bill pay karna hai... [pause 900ms]... consumer number bolu?",
  "standard_bot_result": {
    "cut_off_occurred": true,
    "cut_off_timestamp_sec": 2.6,
    "bot_responded_to": "Mera electricity bill pay karna hai"
  },
  "vartalaap_bot_result": {
    "cut_off_occurred": false,
    "turn_granted_timestamp_sec": 5.65,
    "total_dead_air_latency_ms": 250,
    "bot_responded_to": "Mera electricity bill pay karna hai... consumer number bolu?"
  }
}
```

---

## 5. Environment & Configuration Specifications

All credentials, URLs, and runtime parameters are managed via a centralized `app/config.py` using `pydantic-settings`. Keys are **never hardcoded**.

```ini
# ==============================================================================
# VARTALAAP CONFIGURATION (.env)
# ==============================================================================

# Server Network Configuration
HOST=127.0.0.1
PORT=8000
DEBUG=True

# Gnani AI Credentials & Base URLs
GNANI_API_KEY="your_gnani_api_key_here"
GNANI_STT_WS_URL="wss://api.vachana.ai/stt/v3/stream"
GNANI_STT_REST_URL="https://api.vachana.ai/stt/v3"
GNANI_TTS_WS_URL="wss://api.vachana.ai/api/v1/tts"
GNANI_TTS_REST_URL="https://api.vachana.ai/api/v1/tts/inference"

# Speech Synthesis Voice Config
GNANI_DEFAULT_VOICE="Shivani"
GNANI_TTS_MODEL="timbre-v3.0"
GNANI_DEFAULT_LANG="hi-IN"

# Conversational LLM Config
LLM_PROVIDER="gnani" # "gnani" (Evon) or "openai" or "mock"
LLM_MODEL="evon-v3.3"
LLM_API_KEY="your_llm_api_key_here"

# Vartalaap Engine Parameters
DEFAULT_SILENCE_THRESHOLD_MS=600
MIN_SILENCE_THRESHOLD_MS=250
MAX_SILENCE_THRESHOLD_MS=2000
HESITATION_EXTENDED_MS=1800
CONJUNCTION_EXTENDED_MS=1500

# Device Acceleration
TORCH_DEVICE="cuda" # "cuda" (RTX 5060) or "cpu"
```
