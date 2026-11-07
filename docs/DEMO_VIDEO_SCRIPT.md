# 🎬 Project Vartalaap — 3-Minute Master Demo Video Script

> **Title:** Project Vartalaap: Sovereign Indic-Aware Turn-Taking Engine for Conversational AI  
> **Target Run Time:** Exactly 180 Seconds (03:00)  
> **Submission Challenge:** Gnani.ai "The Great Indian AI Internship Challenge"  
> **Primary Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8 / PyTorch 2.11)  
> **Master Visual Console:** Vintage Nagra Studio Tape Deck (`/static/templates/5-nagra.html`)  

---

## 📺 Video Production & Layout Specification

### Multi-Camera & Screen Layout
```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│                                      │                                      │
│           CAMERA A (LEFT 50%)        │          SCREEN CAPTURE (RIGHT 50%)  │
│                                      │                                      │
│       Presenter & Live Mic Setup     │       Vartalaap Vintage Nagra HUD    │
│   - Professional condenser mic       │   - Dual spinning 15 IPS tape reels  │
│   - Headphone monitor                │   - Analog galvanometer VU needle    │
│   - Live speaker intonation & cues   │   - Real-time WebRTC AEC stream      │
│                                      │                                      │
├──────────────────────────────────────┴──────────────────────────────────────┤
│                     SYSTEM TELEMETRY HUD OVERLAY (LOWER THIRD)               │
│  [STATE: HOLDING_FOR_THOUGHT]  │  ETS: 1450ms  │  DUCKING: -18 dB  │  GPU: 6.5ms     │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Audio Feed 1 (Host Voice):** Live clean speech input recorded at 48kHz, piped to browser as 16kHz mono PCM with WebRTC Echo Cancellation enabled.
- **Audio Feed 2 (System Audio):** Direct digital capture of FastAPI server synthesis (Gnani Timbre v3.0 TTS) and soft-ducking gain shifts (0 dB, -18 dB, -100 dB).
- **Video Format:** 4K UHD (3840×2160) @ 60 FPS or 1080p60.

---

## ⏱️ Timecoded Scene-by-Scene Production Breakdown

---

### SCENE 1: The Hook & The Broken Voice Bot Reality
- **Timecode:** `0:00 — 0:30` (30 seconds)
- **Layout:** Split-screen. Camera A on Left, Screen Capture of **Standard 500ms Fixed VAD Bot** on Right.
- **Visuals:** Right HUD shows standard generic voice bot interface with a static `TIMEOUT: 500ms` indicator in red.

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `0:00 - 0:08` | Host looks directly at Camera A with warm, authoritative energy. Holds an ordinary phone or speaks into mic. | `BOT_STATUS: READY`<br>`MODE: STANDARD_VAD (500ms)`<br>`ETS: FIXED 500ms` | **Host:** *"If you’ve ever spoken to a customer care voice bot in India, you know this exact frustration..."* |
| `0:08 - 0:18` | Host demonstrates a live query to the standard bot. Speaks naturally with a standard code-mixed hesitation.<br><br>**Host speaks:** *"Mera account balance... [pause 600ms]..."* | `MIC: ACTIVE`<br>`VAD_SILENCE: 505ms`<br>`STATE: PREMATURE_CUTOFF`<br>`ACTION: BOT_INTERRUPTED_USER` | **Host (to standard bot):** *"Mera account balance... [pauses to think for 600ms]..."*<br><br>*(SFX: Loud jarring beep as Standard Bot abruptly cuts in!)*<br><br>**Standard Bot Voice (Monotone):** *"I did not understand 'Mera account balance'. Please repeat your request."* |
| `0:18 - 0:30` | Host gestures in frustration. On-screen red graphic flashes: `PREMATURE CUT-OFF: 100% FAILURE`. Overlay appears showing the 4 Duplex Bottlenecks diagram. | `ERROR: PREMATURE_TURN`<br>`USER_TURN_LOST: TRUE` | **Host:** *"It cut me off mid-sentence! In Indian languages, we don’t speak in rigid monolingual sentences. We think, we pause, and we code-mix. Static silence timers break our conversations. Welcome to Project Vartalaap."* |

---

### SCENE 2: The Vartalaap Architecture & Dual-Stream Engine
- **Timecode:** `0:30 — 1:05` (35 seconds)
- **Layout:** Smooth cinematic crossfade into the **Vartalaap Nagra Console** (`/static/templates/5-nagra.html`). Sleek motion graphics diagram overlaying the dual-stream neuro-symbolic pipeline.
- **Visuals:** Dual 15 IPS aluminum tape reels gently spinning. Analog VU meter backlights glow warmly. Lower third displays real-time telemetry stats from local NVIDIA RTX 5060 GPU.

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `0:30 - 0:42` | Camera zooms toward the Nagra Tape Deck HUD. Motion graphic highlights **Layer 1** (<1.0ms CPU Silero VAD) and **8.0s RAM Ring Buffer**. | `PIPELINE: VARTALAAP_v2.0`<br>`VAD: SILERO_CPU (<1.0ms)`<br>`BUFFER: 8.0s RING (128k)` | **Host:** *"Vartalaap is an Indic-aware hybrid turn-taking engine built for Gnani.ai’s Great Indian AI Challenge. Instead of a dumb timer, Vartalaap deploys a dual-stream neuro-symbolic architecture."* |
| `0:42 - 0:54` | Graphic traces the two streams: Left arrow to **Smart-Turn v3.2 ONNX** on NVIDIA RTX 5060; Right arrow to **Gnani Prisma STT** WebSocket parsing trailing tokens. | `GPU: NVIDIA RTX 5060`<br>`PROSODY_LATENCY: 6.57ms`<br>`STREAMING_ASR: GNANI_PRISMA` | **Host:** *"Stream 1 runs Pipecat Smart-Turn v3.2 locally on our NVIDIA RTX 5060 GPU in just 6.5 milliseconds, reading raw acoustic pitch contours. Stream 2 parses real-time tokens from Gnani's Prisma STT for Indic postpositions, conjunctions, and fillers."* |
| `0:54 - 1:05` | Lower HUD highlights the continuous Dynamic ETS formula: $\text{ETS} = 250\text{ms} + 1750\text{ms} \times \text{clamp}(P + S)$ alongside the Two-Stage Soft-Ducking Gate. | `ETS_FORMULA: ACTIVE`<br>`MIN_ETS: 250ms`<br>`MAX_ETS: 2000ms`<br>`SOFT_DUCK_GATE: -18 dB` | **Host:** *"These fuse into a continuous Expected Turn Silence machine that dynamically scales the listening window from 250 milliseconds all the way to 2 seconds, coupled with a Two-Stage Soft-Ducking Gate. Let’s see it live."* |

---

### SCENE 3: Live Demo 1 — The Hinglish Thinking Pause
- **Timecode:** `1:05 — 1:35` (30 seconds)
- **Layout:** Live Split-Screen. Left: Host speaking into microphone. Right: Live Nagra Console reacting in real-time.
- **Visuals:** Host initiates a complex banking query with a deliberate 850ms thinking pause after the word *"matlab"*.
- **Telemetry Action:** Dynamic ETS progress bar smoothly expands to **1250ms**. State pill shifts to vibrant amber: `HOLDING_FOR_THOUGHT`. Nagra reels slow to a smooth idle tension without tripping.

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `1:05 - 1:15` | Host speaks naturally: *"Mera account balance... [pause 850ms]... matlab ki savings account ka kitna hai?"*<br><br>The VU needle bounces with voice energy, then drops to silence. | `SPEECH_ONSET: DETECTED`<br>`INTERIM: "Mera account balance"`<br>`VAD_SILENCE: 0ms -> 300ms`<br>`P_ACOUSTIC: 0.68 (Rising Pitch)` | **Host (Live to Vartalaap):** *"Mera account balance..."*<br>*(Host pauses deliberately for 850ms, looking at screen)* |
| `1:15 - 1:24` | Close-up on the telemetry HUD. At 250ms silence, instead of cutting off, the ETS bar expands to 1250ms! Word *"matlab"* triggers $+0.95$ penalty. State badge glows: `HOLDING_FOR_THOUGHT`. Reels maintain floor. | `VAD_SILENCE: 650ms`<br>`TOKEN: "matlab ki"`<br>`LEXICAL_PENALTY: +0.95`<br>`DYNAMIC_ETS: 1250ms`<br>`STATE: HOLDING_FOR_THOUGHT` | **Host (Continuing smoothly):** *"...matlab ki savings account ka kitna hai?"* |
| `1:24 - 1:35` | Terminal verb *"hai"* drops pitch. In 293ms, the turn resolves cleanly. Speculative prefetch fires. Gnani Timbre TTS responds instantly. | `TERMINAL_TOKEN: "hai"`<br>`P_COMPLETE: 0.89`<br>`ETS: COLLAPSED 293ms`<br>`STATE: TURN_GRANTED`<br>`TTS_STREAMING: GNANI_TIMBRE` | **Vartalaap Bot (Smooth, natural Hindi audio via Gnani Timbre):** *"Aapke savings account mein kul barah hazaar char sau pachas rupaye hain."*<br><br>**Host:** *"Look at that! It held the floor through an 850ms pause because it recognized the rising pitch and the filler 'matlab'. Zero premature cutoff!"* |

---

### SCENE 4: Live Demo 2 — The Cultural "Haanji" Backchannel
- **Timecode:** `1:35 — 2:05` (30 seconds)
- **Layout:** Split-Screen. Left: Host listening intently. Right: Nagra Console playing back agent explanation.
- **Visuals:** Bot is delivering a longer sentence. Host interjects an affirmative cultural backchannel: *"Haanji..."*
- **Telemetry Action:** Stage 1 instantly soft-ducks bot playback by **-18 dB** at $t=2\text{ms}$. Stage 2 identifies *"haanji"* ($220\text{ms}$ duration), keeps bot turn, and restores volume to **0 dB**. Bot never stops speaking!

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `1:35 - 1:45` | Bot begins explaining a transaction. Nagra reels spin at 15 IPS playback speed.<br><br>**Bot Voice:** *"Aapka electricity bill kal raat ko generate hua tha, aur payment due date pandrah November hai..."* | `BOT_SPEAKING: TRUE`<br>`PLAYBACK_GAIN: 0 dB`<br>`REEL_SPEED: 15 IPS` | **Vartalaap Bot:** *"Aapka electricity bill kal raat ko generate hua tha, aur payment due date pandrah November..."* |
| `1:45 - 1:55` | Right in the middle of the bot's speech, Host naturally nods and interjects: *"Haanji..."*<br><br>The VU meter shows host energy. Bot audio dips smoothly by -18 dB for a fraction of a second, then restores to full volume as bot finishes. | `USER_VOICE_ONSET: t=2ms`<br>`STAGE_1_ACTION: DUCK (-18 dB)`<br>`TOKEN_PARSED: "haanji" (220ms)`<br>`STAGE_2: BACKCHANNEL_CONFIRMED`<br>`ACTION: RESTORE_GAIN (0 dB)`<br>`STATE: BACKCHANNEL_IGNORED` | **Host (Interjecting mid-speech):** *"Haanji..."*<br><br>*(Bot audio softly dips to -18 dB for 300ms, then smoothly swells back to 0 dB)*<br><br>**Vartalaap Bot (Continuing seamlessly):** *"...hai. Kya aap abhi UPI se payment karna chahte hain?"* |
| `1:55 - 2:05` | Host points to the live Telemetry Event Log showing `BACKCHANNEL_IGNORED: "haanji" -> GAIN RESTORED 0 dB`. | `METRIC: FALSE_BARGE_IN_PREVENTED`<br>`PIPELINE_RESET: FALSE` | **Host:** *"In standard bots, saying 'haanji' kills the bot's audio dead. In Vartalaap, the Two-Stage Soft-Ducking Gate attenuated playback by -18 dB, verified the cultural backchannel in 220 milliseconds, and restored full volume without stopping!"* |

---

### SCENE 5: Live Demo 3 — Urgent "Ruko" Imperative Barge-In
- **Timecode:** `2:05 — 2:25` (20 seconds)
- **Layout:** Split-Screen. High-speed action demonstration.
- **Visuals:** Bot starts speaking again. Host issues an urgent command: *"Ruko cancel karo!"*
- **Telemetry Action:** Hard barge-in triggers within **46.7 milliseconds**. Playback instantly flushes to **-100 dB**. Reels snap to a complete brake stop with an authentic magnetic tape screech. Floor shifts instantly to user.

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `2:05 - 2:12` | Bot starts detailing payment options:<br>**Bot:** *"Theek hai, main aapke account se transfer initiate..."* | `BOT_SPEAKING: TRUE`<br>`PLAYBACK_GAIN: 0 dB` | **Vartalaap Bot:** *"Theek hai, main aapke account se transfer initiate..."* |
| `2:12 - 2:18` | Host speaks with sudden urgency: *"Ruko cancel karo!"*<br><br>Nagra reels immediately brake and halt! Audio stops on a dime. Red telemetry pill flashes: `HARD_BARGE_IN_TRIGGERED (<50ms)`. | `IMPERATIVE_DETECTED: "ruko"`<br>`ACTION: HARD_FLUSH (-100 dB)`<br>`FLUSH_LATENCY: 46.7 ms`<br>`STATE: BARGE_IN_TRIGGERED`<br>`REELS: BRAKE_HALT` | **Host (Firmly):** *"Ruko cancel karo!"*<br><br>*(SFX: Instantaneous audio cut in 46.7ms, reels stop with mechanical brake click)* |
| `2:18 - 2:25` | Host smiles and gestures at the latency metric on screen: `46.7ms Hard Cancellation`. | `USER_FLOOR_GRANTED: TRUE`<br>`SLA_MET: <60ms` | **Host:** *"Instantaneous. In just 46.7 milliseconds—well under the 60ms human perception threshold—the audio buffer flushed, the reels halted, and the floor returned to the user."* |

---

### SCENE 6: Quantitative Benchmark Proof & RTX 5060 Telemetry
- **Timecode:** `2:25 — 2:45` (20 seconds)
- **Layout:** Full screen graphic transition with split telemetry panel displaying `benchmark/results_summary.json` alongside live terminal execution.
- **Visuals:** Clean, animated comparison tables with verified numbers from the automated benchmark suite.

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `2:25 - 2:35` | On-screen graphic displays **Tier A & Tier B Results Table**:<br>• 500 Segments on RTX 5060: **94.60% Accuracy**, **6.57ms Latency**.<br>• Premature Cutoffs: **100% $\to$ 0.0% (Eliminated)**.<br>• False Interruption: **100% $\to$ 0.0% (Eliminated)**. | `TIER_A: 500 SAMPLES`<br>`ACCURACY: 94.60%`<br>`GPU_LATENCY: 6.57ms`<br>`TIER_B: 30 INDIC SCENARIOS`<br>`CUTOFF_RATE: 0.0%` | **Host:** *"This isn't just a prototype demo—it’s verified by our rigorous automated benchmark suite. Across 500 segments on our local RTX 5060 GPU, Vartalaap achieved 94.6% accuracy at 6.57ms latency. Across 30 Indic stress scenarios, premature cut-offs and false backchannel interruptions were completely eliminated to 0%."* |
| `2:35 - 2:45` | Graphic shows dead-air turnaround: **500ms $\to$ 293.8ms (-41.2% dead air)**. Brief 3-second rapid montage of the other Design Studio archetypes (Bloomberg, Braun, Sonar, Apple Sequoia). | `DEAD_AIR: 293.8ms (-41.2%)`<br>`BARGE_IN: 46.7ms`<br>`TEMPLATES: 10 ARCHETYPES` | **Host:** *"We reduced conversational dead air by over 41%, saving 200 milliseconds per turn. And we wrapped it in 10 world-class Design Studio archetypes."* |

---

### SCENE 7: Sovereign AI for India & Outro
- **Timecode:** `2:45 — 3:00` (15 seconds)
- **Layout:** Camera A on Host, transitioning to Title Card with GitHub repository link, Gnani.ai logo, and Sovereign AI for India emblem.
- **Visuals:** Warm lighting, high-contrast typography, clean GitHub URL and QR code.

| Timecode | Visual Cues & On-Screen Actions | Telemetry HUD State | Exact Voiceover / Audio Script |
|---|---|---|---|
| `2:45 - 2:54` | Host looks into camera with sincere, visionary delivery. Subtitles appear in Hindi and English. | `MISSION: SOVEREIGN_AI`<br>`COMPLIANCE: DPDP_ACT_2023`<br>`AUDIO_RETENTION: ZERO_RAM` | **Host:** *"Project Vartalaap proves that Sovereign AI for India is not about copying Western defaults. It’s about building foundational systems that understand our rhythms, our languages, and our culture—with 100% DPDP Act zero-retention privacy."* |
| `2:54 - 3:00` | Fullscreen Closing Card:<br>**VARTALAAP (वार्तालाप)**<br>Built for Gnani.ai The Great Indian AI Challenge<br>`github.com/iSHAANphaye/Gnani-Vartalaap` | `STATUS: COMPLETE`<br>`LICENSE: APACHE 2.0`<br>`WARM_OUTRO_FADE` | **Host:** *"The code, benchmarks, and interactive console are open source. Thank you, Gnani.ai. Jai Hind!"*<br><br>*(Music swells with modern Indian ambient synthesis, fading smoothly to black at 03:00)* |

---

## 📋 Production Checklist for Recording

- [ ] **Environment Setup:** Launch `uv run uvicorn app.main:app --port 8000` and open `/static/templates/5-nagra.html` on Google Chrome.
- [ ] **Audio Routing:** Route microphone into Chrome with echo cancellation enabled; route system audio through OBS Virtual Audio Cable to capture clean Timbre TTS.
- [ ] **OBS Studio Canvas:** Configure 1920×1080 split-screen scene (Host Webcam 50% left, Browser Capture 50% right).
- [ ] **Telemetry HUD Overlay:** Ensure dynamic ETS bar, state badge, and decibel ducking readout are clearly legible in 1080p.
- [ ] **Test Script Rehearsal:** Practice the 850ms pause after *"Mera account balance..."* and the natural *"Haanji"* interjection to ensure precise telemetry alignment.
- [ ] **Total Run Time Verification:** Ensure final edit clocks in between 02:55 and 03:00.
