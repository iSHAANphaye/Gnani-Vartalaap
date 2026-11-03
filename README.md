# Vartalaap (वार्तालाप)
### Indic-Aware Hybrid Turn Detection Engine for Conversational Voice Agents

Vartalaap is an acoustic-prosodic and lexical turn-taking coordinator designed specifically for the rhythm of Indian multilingual conversation. It couples real-time acoustic feature extraction, streaming speech recognition, and culturally grounded barge-in handling to solve premature cut-offs and awkward voice-bot dead air.

Developed for **The Great Indian AI Challenge** by Gnani.ai.

---

## The Problem

Traditional voice bots use static 500ms–800ms silence timeouts calibrated for monolingual English. In Indian conversational settings, this architecture causes two major failure modes:

1. **Code-Mixing Pauses:** Speakers mixing Hindi, Tamil, Telugu, and English often pause mid-sentence to switch vocabulary (*"Mera account balance... [pause]... matlab ki savings account ka kitna hai?"*). Static timers interpret this thinking gap as an end-of-turn and cut the user off prematurely.
2. **Cultural Backchannels as False Interruptions:** Indian conversations rely heavily on affirmative acknowledgment tokens (*"haanji"*, *"achha"*, *"theek hai"*, *"seri"*, *"ante"*). When an agent is speaking and the caller says *"haanji"*, standard barge-in systems treat it as a command, immediately cutting off the agent mid-sentence.

---

## Architecture: Dual-Stream Neuro-Symbolic Gate

Vartalaap replaces static silence timers with a multi-tier pipeline combining on-device prosodic classification with streaming lexical cues:

```
User Mic (16kHz PCM Stream via WebSocket + WebRTC AEC)
  │
  ├──► Layer 1: Acoustic VAD (Silero VAD, <2ms on CPU)
  │      └── Rolling 8.0-Second Audio Ring Buffer (RAM)
  │
  ├──► Layer 2A: Prosodic Turn Classifier (Smart-Turn v3.2 on RTX 5060 GPU, ~5ms)
  │      └── P(incomplete_acoustic)
  │
  ├──► Layer 2B: Real-Time Streaming ASR (Gnani Prisma STT WebSocket)
  │      └── Partial Token Stream ──► S(lexical_penalty)
  │
  ├──► Layer 3A: Two-Stage Cultural Backchannel Gate
  │      ├── Stage 1 (t = 2ms): Soft-duck agent playback by -18 dB
  │      └── Stage 2 (t = 180-320ms):
  │            ├── If backchannel token ("haanji", "seri"): Restore volume to 0 dB
  │            └── If genuine command ("ruko", "cancel"): Hard flush in <60ms
  │
  └──► Layer 3B: Continuous Dynamic ETS Controller
         ├── ETS(ms) = 250ms + 1750ms * clamp(P_acoustic + S_lexical, 0.0, 1.0)
         └── P(complete) > 0.75 at 200ms ──► Speculative LLM Prefetch (Gnani Evon)
```

### Dynamic Windowing Behavior

| Context | Markers | Silence Window | Action |
|---|---|---|---|
| **Hanging Conjunction** | *ke, ka, ki, aur, lekin, magar, kyunki* | 1,800 ms | Floor held. Incomplete clause detected. |
| **Cognitive Filler** | *matlab, woh, ante, seri na*, vowel elongation | 2,000 ms | Floor held; UI outputs `HOLDING: HESITATION`. |
| **Syntactic Incompleteness** | Open SOV sentence structure; rising intonation | 1,400 ms | Acoustic model flags incomplete boundary. |
| **Terminal Closure** | Terminal verb closure (*karo, hai, chahiye*) + pitch drop | 300 ms | Fast handoff. Response streamed immediately. |
| **Short Confirmation** | *Haan, Nahi, Theek hai, Yes* | 250 ms | Fast turnaround for confirmations. |
| **Mid-Turn Backchannel** | Caller says *"haanji", "achha", "seri"* while agent speaks | N/A | -18 dB soft-ducking recovers to 0 dB without stopping playback. |
| **Hard Barge-In** | Caller says *"ruko", "cancel", "stop"* | < 60 ms | Instant audio flush and state rollback. |

---

## Project Structure

```
├── app/
│   ├── config.py              # Pydantic v2 settings with SecretStr masking
│   ├── core/
│   │   ├── vad.py             # Silero VAD + 8.0s PCM audio ring buffer
│   │   ├── engine.py          # Dynamic turn orchestrator & prefetch coordinator
│   │   ├── backchannel.py     # Two-stage soft-ducking cultural gate
│   │   ├── semantic.py        # Indic postposition parser + Smart-Turn wrapper
│   │   └── normalizer.py      # Cross-script phonetic normalizer
│   ├── services/
│   │   ├── gnani_stt.py       # Gnani Prisma STT WebSocket client
│   │   ├── gnani_tts.py       # Gnani Timbre v3.0 streaming client
│   │   └── smart_turn.py      # Pipecat Smart-Turn v3.2 GPU engine
│   └── static/                # Real-time telemetry HUD & visualizer
├── benchmark/
│   ├── run_benchmark.py       # Two-tier evaluation suite (Tier A: HF, Tier B: Hinglish)
│   └── test_samples/          # Curated Indic stress-test clips
├── docs/                      # PRD, Architecture, and Engineering tickets
├── models/                    # Model weight cache (git-ignored)
└── scripts/
    ├── verify_env.py          # Hardware & PyTorch CUDA verification
    └── test_ticket02.py       # Services & Smart-Turn latency tests
```

---

## Hardware & Environment

Tested and benchmarked on:
- **OS:** Windows 11
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, compute capability `sm_120` Blackwell)
- **CUDA Runtime:** 12.8 via PyTorch 2.11 (`torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`)
- **Package Manager:** `uv` (Python 3.11)
- **Smart-Turn v3.2 Latency:** ~4.9ms – 7.1ms on local RTX 5060 GPU

---

## Quickstart

### 1. Prerequisites
Install `uv` if you haven't already:
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/iSHAANphaye/Gnani-Vartalaap.git
cd Gnani-Vartalaap

# Sync virtualenv and dependencies
uv sync

# Configure credentials
cp .env.example .env
# Edit .env with your GNANI_API_KEY
```

### 3. Verify Hardware & Dependencies
```bash
uv run python scripts/verify_env.py
uv run python scripts/test_ticket02.py
```

---

## Implementation Progress

- [x] **TICKET-01**: Repository scaffolding, UV virtualenv, and RTX 5060 CUDA 12.8 verification.
- [x] **TICKET-02**: Gnani Prisma STT & Timbre TTS client wrappers, Pydantic settings, and Smart-Turn v3.2 GPU loader.
- [ ] **TICKET-03**: Silero VAD & 8.0-second rolling audio ring buffer.
- [ ] **TICKET-04**: Two-stage soft-ducking cultural gate & cross-script normalizer.
- [ ] **TICKET-05**: Indic postposition parser & Smart-Turn feature extractor.
- [ ] **TICKET-06**: Continuous ETS state machine & speculative LLM prefetch.
- [ ] **TICKET-07**: Full-duplex FastAPI WebSocket server.
- [ ] **TICKET-08**: Diagnostic telemetry HUD with WebRTC AEC.
- [ ] **TICKET-09**: Two-tier automated benchmark suite.
- [ ] **TICKET-10**: Demo video production & submission package.

---

## License
Apache 2.0
