# 📋 Project Vartalaap — Engineering Master TODO

> **Status:** 9 / 10 Milestones Completed (90%)  
> **Target Subsystem:** Indic-Aware Hybrid Turn Detection Engine  
> **Challenge:** Gnani.ai "The Great Indian AI Internship Challenge" (Theme: Sovereign AI for India)  
> **Submission Deadline:** November 10, 2026, 23:59 IST

---

## 📊 Milestone Summary Table

| # | Milestone & Target Subsystem | Priority | Status | Commit / Delivery |
|---|---|:---:|:---:|---|
| **01** | **Repository Scaffolding, UV Virtualenv & PyTorch CUDA**<br>`app/config.py`, `.venv` | P0 | ✅ **DONE** | `255f412` — PyTorch 2.11+cu128 verified on RTX 5060 |
| **02** | **Gnani Speech Clients & Smart-Turn v3.2 Loader**<br>`app/services/` | P0 | ✅ **DONE** | `6adbdbd` — Prisma STT, Timbre v3.0 TTS, 2.5ms GPU inference |
| **03** | **Silero VAD & 8.0s Audio Ring Buffer**<br>`app/core/vad.py` | P0 | ✅ **DONE** | `47399bd` — CPU VAD (<1.0ms), 8.0s circular buffer |
| **04** | **Two-Stage Soft-Ducking Gate & Script Normalizer**<br>`app/core/backchannel.py`, `normalizer.py` | P0 | ✅ **DONE** | `a4b1abc` — -18dB ducking at 2ms, Indic cultural lexicons |
| **05** | **Indic Postposition Parser & Hybrid Classifier**<br>`app/core/semantic.py` | P0 | ✅ **DONE** | `7f84684` — Lexical penalties + Smart-Turn acoustic fusion (~6.2ms) |
| **06** | **Continuous ETS State Machine & Speculative LLM Prefetch**<br>`app/core/engine.py` | P0 | ✅ **DONE** | `33ba888` — Continuous formula $250\text{ms} + 1750\text{ms} \times (P+S)$ |
| **07** | **Full-Duplex FastAPI WebSocket Server**<br>`app/main.py` | P0 | ✅ **DONE** | `fd6724e` — `/ws/audio` binary PCM & telemetry streaming |
| **08** | **Diagnostic Telemetry HUD & 10 Design Studio Templates**<br>`app/static/`, `app/static/templates/` | P0 | ✅ **DONE** | `a1e2950` — 10 distinct interactive archetypes, default Nagra console |
| **09** | **Two-Tier Automated Benchmark Suite**<br>`benchmark/run_benchmark.py` | P1 | ✅ **DONE** | Tier A (500 HF segments) + Tier B (30 Indic clips), 0% cutoffs |
| **10** | **Demo Video Production & Submission Documentation**<br>`README.md`, video script | P0 | 🟡 **NEXT UP** | 3-minute split-screen demo video, technical report, submission |

---

## 🎯 Completed Deliverables Detail

### Milestone 01 — Scaffold & GPU Environment
- [x] Initialized UV virtualenv with Python 3.11.
- [x] Installed `torch==2.11.0+cu128` matching the NVIDIA GeForce RTX 5060 Laptop GPU (`sm_120`).
- [x] Verified CUDA acceleration (`torch.cuda.is_available() == True`).

### Milestone 02 — Gnani Stack & Smart-Turn v3.2
- [x] Built `GnaniSTTClient` (`app/services/gnani_stt.py`): Full-duplex streaming client for Prisma STT WebSocket.
- [x] Built `GnaniTTSClient` (`app/services/gnani_tts.py`): Timbre v3.0 client supporting `<excited>` and `<pauses>` prosody tags.
- [x] Downloaded official `pipecat-ai/smart-turn-v3` ONNX model into `models/smart_turn_v3.onnx`.
- [x] Built `SmartTurnEngine` (`app/services/smart_turn.py`) with CUDA execution provider (latency: 2.5ms).

### Milestone 03 — Acoustic VAD & Memory Ring Buffer
- [x] Built `AudioRingBuffer` (`app/core/vad.py`): Preallocated 8.0s circular numpy buffer (128,000 samples @ 16kHz).
- [x] Built `SileroVAD` wrapper with automatic 16-bit PCM chunk alignment; average CPU latency: ~0.98ms.

### Milestone 04 — Cultural Backchannel & Soft-Ducking Gate
- [x] Built `CrossScriptNormalizer` (`app/core/normalizer.py`): Collapses character elongations, maps Devanagari, Tamil, Telugu, and Romanized forms.
- [x] Built `TwoStageBackchannelGate` (`app/core/backchannel.py`):
  - Stage 1: Soft-ducks bot playback by -18 dB at $t = 2\text{ms}$.
  - Stage 2: Restores 0 dB on supportive backchannels (*"haanji"*, *"achha"*, *"seri"*); issues -100 dB hard flush on imperatives (*"ruko"*, *"cancel"*).

### Milestone 05 — Fused Hybrid Turn Classifier
- [x] Built `IndicLexicalParser` (`app/core/semantic.py`):
  - Postposition penalty: $+0.85$ (*ke*, *ka*, *ki*, *mein*, *se*, *ko*, *par*).
  - Conjunction penalty: $+0.80$ (*aur*, *lekin*, *magar*, *kyunki*).
  - Cognitive filler penalty: $+0.95$ (*matlab*, *woh*, *ante*, *seri na*).
  - Terminal closure reward: $-0.65$ (*karo*, *hai*, *chahiye*).
- [x] Built `HybridTurnClassifier`: Fuses acoustic features with lexical scores in ~6.2ms.

### Milestone 06 — Continuous ETS Engine & Speculative Prefetching
- [x] Built `DynamicTurnEngine` (`app/core/engine.py`):
  $$\text{ETS(ms)} = 250\text{ms} + 1750\text{ms} \times \text{clamp}(P_{\text{acoustic}} + S_{\text{lexical}}, 0.0, 1.0)$$
- [x] Speculative prefetching triggers asynchronously when $P(\text{complete}) > 0.75$ and silence exceeds 200ms.
- [x] Clean cooperative cancellation if user resumes speaking.

### Milestone 07 — Full-Duplex FastAPI WebSocket Server
- [x] Implemented `/ws/audio` binary PCM frame ingestion and telemetry streaming in `app/main.py`.
- [x] Built healthcheck endpoint `/api/health` with GPU and credential status.

### Milestone 08 — Diagnostic HUD & 10 Design Studio Templates
- [x] Built default DAW cockpit (`app/static/index.html`, `app.js`, `styles.css`) with WebRTC AEC constraints.
- [x] Built 10 Distinct Design Templates (`app/static/templates/`):
  1. `1-braun.html` — Braun / Teenage Engineering OP-1 with analog galvanometer VU needle.
  2. `2-bloomberg.html` — Bloomberg Terminal with ASCII frames & waterfall spectrogram.
  3. `3-swiss.html` — Swiss Typographic Broadsheet (Zurich 1968) with hairline oscillogram.
  4. `4-linear.html` — Silicon Valley dark precision cockpit with command dock.
  5. `5-nagra.html` — Vintage Nagra tape recorder with spinning magnetic reels.
  6. `6-sonar.html` — Submarine acoustic sonar PPI with 360° rotating sweep beam.
  7. `7-cyberpunk.html` — Akihabara cyberdeck with CRT scanlines & dual-beam oscilloscope.
  8. `8-brutalist.html` — Kinetic editorial brutalism with 3px black borders & hard drop shadows.
  9. `9-quantum.html` — Bio-acoustic laboratory with 3D-styled multi-sine wave ribbon.
  10. `10-apple.html` — Bento Grid with macOS Sequoia fluid glass & morphing Dynamic Island pill.
- [x] Built Master Showcase Gallery (`gallery.html`) accessible via `/gallery` and `/templates`.
- [x] Built `shared-engine.js` with offline simulation fallback & live 16kHz microphone streaming.

---

### Milestone 09 — Two-Tier Automated Benchmark Suite (`benchmark/run_benchmark.py`)
- [x] Built physical audio generator (`benchmark/generate_test_samples.py`) with formant resonators & pitch contours.
- [x] Generated 30 curated 16kHz WAV test clips across Categories A, B, C, D, E in `benchmark/test_samples/`.
- [x] Authored comprehensive scenario metadata manifest in `benchmark/test_samples/manifest.json`.
- [x] Implemented Tier A evaluation streaming 500 samples from `pipecat-ai/smart-turn-data-v3.2-test` (94.60% accuracy, 6.57ms GPU latency).
- [x] Implemented Tier B comparative evaluation across all 30 Indic stress scenarios:
  - Premature Cut-off Rate: Eliminated (100.0% -> 0.0%).
  - False Interruption Rate on Backchannels: Eliminated (100.0% -> 0.0%).
  - Dead-Air Turnaround: Reduced from 500.0ms to 293.8ms (-41.2% dead air).
  - Hard Barge-In Cancellation: Reduced from 230.0ms to 46.7ms (<60ms SLA met).
- [x] Exported results to `benchmark/results_summary.json` with clean terminal markdown reporting.

---

## 🚀 Next Milestone

### Milestone 10 — Demo Video Production & Final Submission
- [ ] Write 3-minute video recording script demonstrating:
  - Test 1: Hinglish thinking pause (*"Mera balance check... [pause]... matlab"*).
  - Test 2: Soft-ducking recovery on cultural backchannel (*"Haanji"*).
  - Test 3: Side-by-side A/B comparison against standard 500ms VAD.
- [ ] Finalize `README.md` with system diagrams, benchmark metrics, and reproduction guide.
- [ ] Verify clean submission package before November 10, 23:59 IST.
