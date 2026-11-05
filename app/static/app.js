/**
 * Project Vartalaap — Real-Time Diagnostic Telemetry HUD Cockpit Controller
 *
 * Implements:
 * 1. WebRTC & Web Audio capture pipeline with Acoustic Echo Cancellation (AEC).
 * 2. Real-time 16kHz 16-bit mono linear PCM chunking (512 samples / 1024 bytes) over WebSocket.
 * 3. 60 FPS Dual-Waveform Canvas Oscilloscope with dynamic color shifts.
 * 4. Bot Audio Player with queue scheduling and soft-ducking (-18 dB / HARD_FLUSH) automation.
 * 5. High-throughput telemetry state synchronization with tabular metric updates.
 * 6. Signature interactive A/B mode toggle (Standard 500ms Fixed vs Vartalaap v2.0 Dynamic).
 * 7. One-click judge benchmark simulations for Hinglish pause, Haanji backchannel, and Ruko interrupt.
 */

(() => {
  'use strict';

  // --------------------------------------------------------------------------
  // Configuration & Constants
  // --------------------------------------------------------------------------
  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_SAMPLES = 512;
  const CHUNK_BYTES = CHUNK_SAMPLES * 2; // 16-bit = 2 bytes per sample (1024 bytes)

  const CATEGORY_PENALTIES = {
    COGNITIVE_HESITATION: '+0.95 (Delay)',
    POSTPOSITION: '+0.85 (Delay)',
    HANGING_CONJUNCTION: '+0.80 (Delay)',
    TERMINAL_CLOSURE: '-0.65 (Fast Release)',
    SHORT_CONFIRMATION: '-0.75 (Fast Release)',
    NEUTRAL: '0.00 (Baseline)',
  };

  const KNOWN_MARKERS = [
    'matlab', 'matlab ki', 'ke liye', 'keliye', 'ke', 'ka', 'ki', 'ko', 'se', 'mein', 'par',
    'aur', 'lekin', 'magar', 'kyunki', 'ya', 'warna',
    'haanji', 'haan', 'achha', 'accha', 'theek hai', 'ji', 'seri',
    'ruko', 'rukiye', 'stop', 'chup', 'cancel', 'wait', 'bas',
  ];

  // --------------------------------------------------------------------------
  // Application State
  // --------------------------------------------------------------------------
  const state = {
    isConnected: false,
    isRecording: false,
    activeMode: 'VARTALAAP', // 'VARTALAAP' or 'STANDARD'
    turnState: 'LISTENING',
    elapsedSilenceMs: 0,
    dynamicEtsMs: 250,
    pComplete: 0,
    pIncomplete: 0,
    category: 'NEUTRAL',
    duckingGainDb: 0.0,
    matchedMarkers: [],
    cutoffsPrevented: 0,
    backchannelsPreserved: 0,
    lastCutoffFlagged: false,
    turnGrantedTimestamp: 0,
    ttfaMeasured: null,
  };

  // --------------------------------------------------------------------------
  // Audio Pipeline References
  // --------------------------------------------------------------------------
  let audioCtx = null;
  let micStream = null;
  let micSourceNode = null;
  let micProcessorNode = null;
  let micAnalyserNode = null;

  let botGainNode = null;
  let botAnalyserNode = null;
  let nextPlayTime = 0;
  const activeBotSources = [];

  // Downsampling / Chunking buffer accumulator
  let pcmAccumulator = new Float32Array(0);

  // WebSocket Connection Reference
  let ws = null;
  let wsReconnectTimer = null;

  // Animation Loop Reference
  let animationFrameId = null;

  // --------------------------------------------------------------------------
  // DOM Elements Cache
  // --------------------------------------------------------------------------
  const dom = {
    // Header
    wsConnDot: document.getElementById('wsConnDot'),
    wsConnLabel: document.getElementById('wsConnLabel'),
    gpuStatusDot: document.getElementById('gpuStatusDot'),
    modeStandardBtn: document.getElementById('modeStandardBtn'),
    modeVartalaapBtn: document.getElementById('modeVartalaapBtn'),
    arbitrationModeBadge: document.getElementById('arbitrationModeBadge'),

    // Main Cockpit
    turnStateBadge: document.getElementById('turnStateBadge'),
    turnStateDot: document.getElementById('turnStateDot'),
    turnStateText: document.getElementById('turnStateText'),
    elapsedSilenceVal: document.getElementById('elapsedSilenceVal'),
    dynamicEtsVal: document.getElementById('dynamicEtsVal'),
    silenceBar: document.getElementById('silence-bar'),
    fixed500msMarker: document.getElementById('fixed500msMarker'),
    silenceReasonTrace: document.getElementById('silenceReasonTrace'),

    // Oscilloscope
    canvas: document.getElementById('oscilloscope'),

    // Transcripts
    transcriptBox: document.getElementById('transcript-box'),
    agentResponseBox: document.getElementById('agent-response-box'),
    botPlaybackStatus: document.getElementById('botPlaybackStatus'),

    // Telemetry Sidebar
    pIncompleteVal: document.getElementById('pIncompleteVal'),
    pIncompleteFill: document.getElementById('pIncompleteFill'),
    pCompleteVal: document.getElementById('pCompleteVal'),
    pCompleteFill: document.getElementById('pCompleteFill'),
    categoryBadge: document.getElementById('categoryBadge'),
    lexicalPenaltyBadge: document.getElementById('lexicalPenaltyBadge'),
    matchedMarkersList: document.getElementById('matchedMarkersList'),
    duckingStatusBadge: document.getElementById('duckingStatusBadge'),
    duckingBarFill: document.getElementById('duckingBarFill'),
    duckingGainVal: document.getElementById('duckingGainVal'),
    cutoffsPreventedVal: document.getElementById('cutoffsPreventedVal'),
    backchannelsPreservedVal: document.getElementById('backchannelsPreservedVal'),
    ttfaMetricVal: document.getElementById('ttfaMetricVal'),

    // Controls
    sessionBtn: document.getElementById('sessionBtn'),
    sessionBtnText: document.getElementById('sessionBtnText'),
    sessionBtnIcon: document.getElementById('sessionBtnIcon'),
    resetBtn: document.getElementById('resetBtn'),
    simHinglishPauseBtn: document.getElementById('simHinglishPauseBtn'),
    simHaanjiBtn: document.getElementById('simHaanjiBtn'),
    simRukoBtn: document.getElementById('simRukoBtn'),
  };

  // --------------------------------------------------------------------------
  // WebSocket Communication Layer
  // --------------------------------------------------------------------------
  function getWsUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    return `${protocol}//${host}/ws/audio`;
  }

  function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      ws = new WebSocket(getWsUrl());
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        state.isConnected = true;
        updateConnectionStatus('ONLINE', 'green');
        sendJsonMessage({ action: 'SET_MODE', mode: state.activeMode });
      };

      ws.onclose = () => {
        state.isConnected = false;
        updateConnectionStatus('OFFLINE', 'amber');
        scheduleWsReconnect();
      };

      ws.onerror = () => {
        state.isConnected = false;
        updateConnectionStatus('ERROR', 'red');
      };

      ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          handleIncomingJsonMessage(event.data);
        }
      };
    } catch (err) {
      console.error('Failed to initialize WebSocket:', err);
      scheduleWsReconnect();
    }
  }

  function scheduleWsReconnect() {
    if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
    wsReconnectTimer = setTimeout(() => {
      initWebSocket();
    }, 2000);
  }

  function sendJsonMessage(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function sendBinaryAudioFrame(arrayBuffer) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(arrayBuffer);
    }
  }

  function updateConnectionStatus(text, color) {
    if (dom.wsConnLabel) dom.wsConnLabel.textContent = text;
    if (dom.wsConnDot) {
      dom.wsConnDot.className = `status-dot ${color}`;
    }
  }

  // --------------------------------------------------------------------------
  // Incoming Message Dispatcher
  // --------------------------------------------------------------------------
  function handleIncomingJsonMessage(rawJson) {
    try {
      const msg = JSON.parse(rawJson);
      const type = msg.type || '';

      switch (type) {
        case 'TELEMETRY':
          handleTelemetry(msg);
          break;
        case 'CONTROL':
          handleControlAction(msg);
          break;
        case 'TRANSCRIPT_STREAM':
          handleTranscriptStream(msg);
          break;
        case 'BOT_TURN_START':
          handleBotTurnStart(msg);
          break;
        case 'AUDIO_CHUNK':
          handleIncomingAudioChunk(msg);
          break;
        case 'BOT_TURN_END':
          handleBotTurnEnd(msg);
          break;
        case 'CONTROL_ACK':
          // Control acknowledged
          break;
        case 'PONG':
          break;
        default:
          break;
      }
    } catch (err) {
      console.warn('Failed to parse incoming message:', err);
    }
  }

  // --------------------------------------------------------------------------
  // Telemetry Synchronization
  // --------------------------------------------------------------------------
  function handleTelemetry(msg) {
    state.turnState = msg.state || 'LISTENING';
    state.elapsedSilenceMs = Number(msg.elapsed_silence_ms) || 0;
    state.dynamicEtsMs = Number(msg.dynamic_ets_ms) || 250;
    state.pComplete = Number(msg.p_complete) || 0;
    state.pIncomplete = msg.p_incomplete !== undefined
      ? Number(msg.p_incomplete)
      : Math.max(0, 1.0 - state.pComplete);
    state.category = msg.category || 'NEUTRAL';
    state.duckingGainDb = Number(msg.ducking_gain_db) || 0.0;
    const reason = msg.reason || '';

    // Update Turn State Badge
    updateTurnStateDisplay(state.turnState);

    // Update Silence Timeout Progress Bar
    if (dom.elapsedSilenceVal) dom.elapsedSilenceVal.textContent = Math.round(state.elapsedSilenceMs);
    if (dom.dynamicEtsVal) dom.dynamicEtsVal.textContent = Math.round(state.dynamicEtsMs);

    const progressPct = Math.min(100, Math.max(0, (state.elapsedSilenceMs / Math.max(state.dynamicEtsMs, 1)) * 100));
    if (dom.silenceBar) {
      dom.silenceBar.style.width = `${progressPct.toFixed(1)}%`;
    }

    // Update Fixed 500ms Threshold Marker on the dynamic track
    if (dom.fixed500msMarker) {
      const fixedPct = Math.min(100, (500 / Math.max(state.dynamicEtsMs, 500)) * 100);
      dom.fixed500msMarker.style.left = `${fixedPct.toFixed(1)}%`;
    }

    if (dom.silenceReasonTrace && reason) {
      dom.silenceReasonTrace.textContent = reason;
    }

    // Update Multimodal Boundary Probabilities
    const pIncPct = (state.pIncomplete * 100).toFixed(1);
    const pCompPct = (state.pComplete * 100).toFixed(1);

    if (dom.pIncompleteVal) dom.pIncompleteVal.textContent = `${pIncPct}%`;
    if (dom.pIncompleteFill) dom.pIncompleteFill.style.width = `${pIncPct}%`;
    if (dom.pCompleteVal) dom.pCompleteVal.textContent = `${pCompPct}%`;
    if (dom.pCompleteFill) dom.pCompleteFill.style.width = `${pCompPct}%`;

    // Update Linguistic Category & Matched Markers
    if (dom.categoryBadge) {
      dom.categoryBadge.textContent = state.category;
    }
    if (dom.lexicalPenaltyBadge) {
      const penaltyText = CATEGORY_PENALTIES[state.category] || '0.00';
      dom.lexicalPenaltyBadge.textContent = `PENALTY: ${penaltyText}`;
    }

    // Update Matched Markers tags if any are mentioned in reason or current transcript
    updateMatchedMarkers(reason);

    // Evaluate Impact Counter: Cut-Offs Prevented
    // In Vartalaap dynamic mode, if elapsed silence surpasses 500ms (the standard cutoff)
    // while ETS was dynamically extended, a cutoff was actively prevented!
    if (state.activeMode === 'VARTALAAP' && state.dynamicEtsMs > 500 && state.elapsedSilenceMs > 500) {
      if (!state.lastCutoffFlagged) {
        state.cutoffsPrevented += 1;
        state.lastCutoffFlagged = true;
        if (dom.cutoffsPreventedVal) {
          dom.cutoffsPreventedVal.textContent = String(state.cutoffsPrevented);
        }
      }
    } else if (state.elapsedSilenceMs < 100) {
      state.lastCutoffFlagged = false;
    }

    // Evaluate Impact Counter: Cultural Backchannels Preserved
    if (state.turnState === 'BACKCHANNEL_IGNORED' || state.category === 'SHORT_CONFIRMATION') {
      if (!state.lastBackchannelFlagged) {
        state.backchannelsPreserved += 1;
        state.lastBackchannelFlagged = true;
        if (dom.backchannelsPreservedVal) {
          dom.backchannelsPreservedVal.textContent = String(state.backchannelsPreserved);
        }
      }
    } else {
      state.lastBackchannelFlagged = false;
    }

    // Turn Granted timestamp capture for TTFA
    if (state.turnState === 'TURN_GRANTED') {
      state.turnGrantedTimestamp = performance.now();
    }
  }

  function updateTurnStateDisplay(turnState) {
    if (!dom.turnStateBadge || !dom.turnStateText || !dom.turnStateDot) return;

    dom.turnStateBadge.className = 'turn-state-badge';
    dom.turnStateDot.className = 'status-dot';

    switch (turnState) {
      case 'LISTENING':
        dom.turnStateBadge.classList.add('state-listening');
        dom.turnStateDot.classList.add('green', 'pulse');
        dom.turnStateText.textContent = 'LISTENING';
        break;
      case 'EVALUATING_PAUSE':
      case 'HOLDING_FOR_THOUGHT':
      case 'SPECULATIVE_PREFETCHING':
        dom.turnStateBadge.classList.add('state-holding');
        dom.turnStateDot.classList.add('amber', 'pulse');
        dom.turnStateText.textContent = 'HOLDING_FOR_THOUGHT';
        break;
      case 'BACKCHANNEL_IGNORED':
        dom.turnStateBadge.classList.add('state-backchannel');
        dom.turnStateDot.classList.add('cyan');
        dom.turnStateText.textContent = 'BACKCHANNEL_IGNORED';
        break;
      case 'BOT_SPEAKING':
        dom.turnStateBadge.classList.add('state-bot-speaking');
        dom.turnStateDot.classList.add('violet', 'pulse');
        dom.turnStateText.textContent = 'BOT_SPEAKING';
        break;
      case 'BOT_DUCKED':
        dom.turnStateBadge.classList.add('state-bot-ducked');
        dom.turnStateDot.classList.add('amber', 'pulse');
        dom.turnStateText.textContent = 'BOT_DUCKED';
        break;
      case 'TURN_GRANTED':
        dom.turnStateBadge.classList.add('state-turn-granted');
        dom.turnStateDot.classList.add('cyan', 'pulse');
        dom.turnStateText.textContent = 'TURN_GRANTED';
        break;
      default:
        dom.turnStateBadge.classList.add('state-listening');
        dom.turnStateDot.classList.add('green');
        dom.turnStateText.textContent = turnState;
        break;
    }
  }

  function updateMatchedMarkers(reasonText) {
    if (!dom.matchedMarkersList) return;

    const transcript = dom.transcriptBox ? dom.transcriptBox.innerText.toLowerCase() : '';
    const combined = `${reasonText.toLowerCase()} ${transcript}`;

    const matched = [];
    for (const marker of KNOWN_MARKERS) {
      if (combined.includes(marker)) {
        matched.push(marker);
      }
    }

    if (matched.length === 0) {
      dom.matchedMarkersList.innerHTML = '<span class="marker-tag">none</span>';
    } else {
      const unique = Array.from(new Set(matched));
      dom.matchedMarkersList.innerHTML = unique
        .map((m) => `<span class="marker-tag">${escapeHtml(m)}</span>`)
        .join('');
    }
  }

  // --------------------------------------------------------------------------
  // Soft-Ducking & Gain Automation
  // --------------------------------------------------------------------------
  function handleControlAction(msg) {
    const action = msg.action || '';
    const gainDb = Number(msg.gain_db) || 0.0;

    applyGainDucking(action, gainDb);
  }

  function applyGainDucking(action, gainDb) {
    state.duckingGainDb = gainDb;

    // Decibel to linear conversion: 10^(dB / 20)
    let linearGain = 1.0;
    if (gainDb <= -90.0) {
      linearGain = 0.0;
    } else {
      linearGain = Math.pow(10, gainDb / 20);
    }

    // Apply Web Audio gain transition with zero pop
    if (botGainNode && audioCtx) {
      const now = audioCtx.currentTime;
      botGainNode.gain.cancelScheduledValues(now);

      if (action === 'HARD_FLUSH') {
        botGainNode.gain.setValueAtTime(0.0, now);
        // Abort all playing bot sources immediately
        activeBotSources.forEach((src) => {
          try {
            src.stop();
          } catch (_) {}
        });
        activeBotSources.length = 0;
        nextPlayTime = now;
      } else {
        botGainNode.gain.setTargetAtTime(linearGain, now, 0.04);
      }
    }

    // Update Ducking Telemetry Meter
    updateDuckingMeterDisplay(action, gainDb);
  }

  function updateDuckingMeterDisplay(action, gainDb) {
    if (dom.duckingGainVal) {
      dom.duckingGainVal.textContent = `${gainDb.toFixed(1)} dB`;
    }

    // Meter scale: -100 dB (0%) to 0 dB (100%)
    let pct = 100;
    if (gainDb <= -90) {
      pct = 0;
    } else if (gainDb < 0) {
      pct = Math.max(0, Math.min(100, ((gainDb + 100) / 100) * 100));
    }

    if (dom.duckingBarFill) {
      dom.duckingBarFill.style.width = `${pct.toFixed(1)}%`;
    }

    if (dom.duckingStatusBadge) {
      dom.duckingStatusBadge.className = 'ducking-badge';
      if (action === 'HARD_FLUSH' || gainDb <= -90) {
        dom.duckingStatusBadge.classList.add('flushed');
        dom.duckingStatusBadge.textContent = 'MUTED (-100 dB)';
      } else if (action === 'DUCK' || gainDb < -2) {
        dom.duckingStatusBadge.classList.add('ducked');
        dom.duckingStatusBadge.textContent = 'DUCKED (-18 dB)';
      } else {
        dom.duckingStatusBadge.classList.add('normal');
        dom.duckingStatusBadge.textContent = '0.0 dB NORMAL';
      }
    }
  }

  // --------------------------------------------------------------------------
  // Streaming Transcription Display
  // --------------------------------------------------------------------------
  function handleTranscriptStream(msg) {
    const rawText = msg.text || '';
    if (!dom.transcriptBox || !rawText) return;

    dom.transcriptBox.classList.remove('empty-state');

    // Highlight recognized grammatical markers
    let highlighted = escapeHtml(rawText);
    for (const marker of KNOWN_MARKERS) {
      const regex = new RegExp(`\\b(${marker})\\b`, 'gi');
      highlighted = highlighted.replace(regex, '<span class="marker-highlight">$1</span>');
    }

    dom.transcriptBox.innerHTML = `${highlighted}<span class="streaming-cursor"></span>`;
    dom.transcriptBox.scrollTop = dom.transcriptBox.scrollHeight;
  }

  function handleBotTurnStart(msg) {
    if (dom.botPlaybackStatus) {
      dom.botPlaybackStatus.textContent = 'SYNTHESIZING';
      dom.botPlaybackStatus.style.color = 'var(--accent-violet)';
    }
    if (dom.agentResponseBox) {
      dom.agentResponseBox.classList.remove('empty-state');
      dom.agentResponseBox.textContent = msg.text || '...';
    }
  }

  function handleBotTurnEnd(msg) {
    if (dom.botPlaybackStatus) {
      dom.botPlaybackStatus.textContent = 'IDLE';
      dom.botPlaybackStatus.style.color = 'var(--text-muted)';
    }
    if (dom.agentResponseBox && msg.text) {
      dom.agentResponseBox.textContent = msg.text;
    }
  }

  // --------------------------------------------------------------------------
  // Bot Audio Playback Engine
  // --------------------------------------------------------------------------
  async function handleIncomingAudioChunk(msg) {
    const b64 = msg.audio_base64 || '';
    if (!b64) return;

    // Record TTFA (Time to First Audio) if turn was just granted
    if (state.turnGrantedTimestamp > 0) {
      const elapsed = Math.round(performance.now() - state.turnGrantedTimestamp);
      state.ttfaMeasured = elapsed;
      state.turnGrantedTimestamp = 0;
      if (dom.ttfaMetricVal) {
        dom.ttfaMetricVal.textContent = `${elapsed} ms`;
      }
    }

    ensureAudioContext();

    try {
      const rawBinary = base64ToArrayBuffer(b64);
      const audioBuffer = await audioCtx.decodeAudioData(rawBinary.slice(0));

      const sourceNode = audioCtx.createBufferSource();
      sourceNode.buffer = audioBuffer;
      sourceNode.connect(botGainNode);

      const startTime = Math.max(audioCtx.currentTime, nextPlayTime);
      sourceNode.start(startTime);
      nextPlayTime = startTime + audioBuffer.duration;

      activeBotSources.push(sourceNode);
      sourceNode.onended = () => {
        const idx = activeBotSources.indexOf(sourceNode);
        if (idx !== -1) activeBotSources.splice(idx, 1);
      };

      if (dom.botPlaybackStatus) {
        dom.botPlaybackStatus.textContent = 'STREAMING';
        dom.botPlaybackStatus.style.color = 'var(--accent-violet)';
      }
    } catch (err) {
      console.warn('Could not decode or play bot audio chunk:', err);
    }
  }

  function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  // --------------------------------------------------------------------------
  // WebRTC Audio Ingestion & Resampling Pipeline
  // --------------------------------------------------------------------------
  async function startConversation() {
    ensureAudioContext();
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }

    initWebSocket();

    try {
      // WebRTC getUserMedia with standard acoustic echo cancellation constraints
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: TARGET_SAMPLE_RATE,
          channelCount: 1,
        },
      });

      micSourceNode = audioCtx.createMediaStreamSource(micStream);

      // Create Mic Analyser for Oscilloscope
      micAnalyserNode = audioCtx.createAnalyser();
      micAnalyserNode.fftSize = 1024;
      micAnalyserNode.smoothingTimeConstant = 0.3;
      micSourceNode.connect(micAnalyserNode);

      // ScriptProcessorNode downsampler & chunking buffer
      // (Using 4096 buffer size for smooth downsampling)
      micProcessorNode = audioCtx.createScriptProcessor(4096, 1, 1);
      micProcessorNode.onaudioprocess = handleAudioProcess;

      micSourceNode.connect(micProcessorNode);
      // Connect to silent dummy node to ensure onaudioprocess fires across all browsers
      const dummyGain = audioCtx.createGain();
      dummyGain.gain.value = 0.0;
      micProcessorNode.connect(dummyGain);
      dummyGain.connect(audioCtx.destination);

      state.isRecording = true;
      updateSessionButton(true);

      if (dom.transcriptBox && dom.transcriptBox.classList.contains('empty-state')) {
        dom.transcriptBox.classList.remove('empty-state');
        dom.transcriptBox.innerHTML = '<span style="color: var(--text-muted);">Listening... Speak in Hindi, Hinglish, or English.</span>';
      }
    } catch (err) {
      console.error('Microphone capture error:', err);
      alert(`Microphone access error: ${err.message}. You can still run the judge simulation benchmarks below.`);
    }
  }

  function stopConversation() {
    state.isRecording = false;

    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (micProcessorNode) {
      micProcessorNode.disconnect();
      micProcessorNode = null;
    }
    if (micSourceNode) {
      micSourceNode.disconnect();
      micSourceNode = null;
    }

    sendJsonMessage({ action: 'STOP_CALL' });
    updateSessionButton(false);
  }

  function handleAudioProcess(e) {
    if (!state.isRecording) return;

    const inputData = e.inputBuffer.getChannelData(0);
    const sourceRate = audioCtx.sampleRate;

    // Resample input samples to 16kHz
    const resampled = resampleLinear(inputData, sourceRate, TARGET_SAMPLE_RATE);

    // Append to accumulator
    const newAccumulator = new Float32Array(pcmAccumulator.length + resampled.length);
    newAccumulator.set(pcmAccumulator, 0);
    newAccumulator.set(resampled, pcmAccumulator.length);
    pcmAccumulator = newAccumulator;

    // Extract complete 512-sample (1024-byte) frames
    while (pcmAccumulator.length >= CHUNK_SAMPLES) {
      const slice = pcmAccumulator.subarray(0, CHUNK_SAMPLES);
      pcmAccumulator = pcmAccumulator.subarray(CHUNK_SAMPLES);

      const pcm16Buffer = convertFloat32ToInt16(slice);
      sendBinaryAudioFrame(pcm16Buffer);
    }
  }

  function resampleLinear(inputBuffer, inRate, outRate) {
    if (inRate === outRate) {
      return inputBuffer;
    }
    const ratio = inRate / outRate;
    const outLength = Math.round(inputBuffer.length / ratio);
    const result = new Float32Array(outLength);

    for (let i = 0; i < outLength; i++) {
      const srcIdx = i * ratio;
      const idxFloor = Math.floor(srcIdx);
      const idxCeil = Math.min(inputBuffer.length - 1, idxFloor + 1);
      const weight = srcIdx - idxFloor;
      result[i] = inputBuffer[idxFloor] * (1 - weight) + inputBuffer[idxCeil] * weight;
    }
    return result;
  }

  function convertFloat32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1.0, Math.min(1.0, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16Array.buffer;
  }

  function ensureAudioContext() {
    if (!audioCtx) {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioCtxClass();

      // Configure Bot Gain & Analyser
      botGainNode = audioCtx.createGain();
      botGainNode.gain.setValueAtTime(1.0, audioCtx.currentTime);

      botAnalyserNode = audioCtx.createAnalyser();
      botAnalyserNode.fftSize = 1024;
      botAnalyserNode.smoothingTimeConstant = 0.3;

      botGainNode.connect(botAnalyserNode);
      botAnalyserNode.connect(audioCtx.destination);
    }
  }

  function updateSessionButton(isRecording) {
    if (!dom.sessionBtn || !dom.sessionBtnText) return;

    if (isRecording) {
      dom.sessionBtn.classList.add('recording');
      dom.sessionBtnText.textContent = 'End Conversation';
      if (dom.sessionBtnIcon) {
        dom.sessionBtnIcon.innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2"></rect>';
      }
    } else {
      dom.sessionBtn.classList.remove('recording');
      dom.sessionBtnText.textContent = 'Start Conversation';
      if (dom.sessionBtnIcon) {
        dom.sessionBtnIcon.innerHTML = '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>';
      }
    }
  }

  // --------------------------------------------------------------------------
  // High-DPI Canvas Oscilloscope (60 FPS)
  // --------------------------------------------------------------------------
  function initOscilloscope() {
    if (!dom.canvas) return;

    function renderLoop() {
      drawOscilloscopeFrame();
      animationFrameId = requestAnimationFrame(renderLoop);
    }
    animationFrameId = requestAnimationFrame(renderLoop);
  }

  function drawOscilloscopeFrame() {
    const canvas = dom.canvas;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;

    if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
      canvas.width = displayWidth * dpr;
      canvas.height = displayHeight * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;
    const midY = height / 2;

    // Clear background
    ctx.fillStyle = '#0b0f17';
    ctx.fillRect(0, 0, width, height);

    // Draw Oscilloscope Graticule
    ctx.strokeStyle = '#182234';
    ctx.lineWidth = 1;

    // Horizontal grid divisions
    const hSteps = 4;
    for (let i = 1; i < hSteps; i++) {
      const y = (height / hSteps) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Vertical grid divisions
    const vSteps = 8;
    for (let i = 1; i < vSteps; i++) {
      const x = (width / vSteps) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Center Reference Zero-Crossing Line
    ctx.strokeStyle = 'rgba(55, 65, 81, 0.8)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // 1. Draw User Voice Waveform
    const micData = new Uint8Array(512);
    if (micAnalyserNode) {
      micAnalyserNode.getByteTimeDomainData(micData);
    } else {
      micData.fill(128);
    }

    // Determine User Waveform Color based on turn state
    let strokeColor = '#10b981'; // Green (default speaking)
    if (state.turnState === 'HOLDING_FOR_THOUGHT' || state.turnState === 'EVALUATING_PAUSE') {
      strokeColor = '#f59e0b'; // Amber (pause hold)
    } else if (state.turnState === 'BACKCHANNEL_IGNORED') {
      strokeColor = '#06b6d4'; // Cyan (backchannel)
    }

    ctx.lineWidth = 1.75;
    ctx.strokeStyle = strokeColor;
    ctx.beginPath();

    const sliceWidth = width / micData.length;
    let x = 0;

    for (let i = 0; i < micData.length; i++) {
      const v = micData[i] / 128.0;
      const y = v * midY;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    ctx.stroke();

    // 2. Draw Bot Synthesis Waveform Trace (Violet)
    const botData = new Uint8Array(512);
    if (botAnalyserNode) {
      botAnalyserNode.getByteTimeDomainData(botData);
    } else {
      botData.fill(128);
    }

    // Check if bot has non-trivial signal
    let hasBotAudio = false;
    for (let i = 0; i < botData.length; i += 8) {
      if (Math.abs(botData[i] - 128) > 2) {
        hasBotAudio = true;
        break;
      }
    }

    if (hasBotAudio) {
      ctx.lineWidth = 2.0;
      ctx.strokeStyle = '#8b5cf6'; // Violet
      ctx.beginPath();

      let bx = 0;
      for (let i = 0; i < botData.length; i++) {
        const v = botData[i] / 128.0;
        const by = v * midY;

        if (i === 0) {
          ctx.moveTo(bx, by);
        } else {
          ctx.lineTo(bx, by);
        }
        bx += sliceWidth;
      }
      ctx.stroke();
    }

    ctx.restore();
  }

  // --------------------------------------------------------------------------
  // Signature Mode Toggle
  // --------------------------------------------------------------------------
  function setMode(newMode) {
    state.activeMode = newMode;

    if (newMode === 'STANDARD') {
      if (dom.modeStandardBtn) {
        dom.modeStandardBtn.className = 'mode-toggle-btn active standard-mode';
      }
      if (dom.modeVartalaapBtn) {
        dom.modeVartalaapBtn.className = 'mode-toggle-btn';
      }
      if (dom.arbitrationModeBadge) {
        dom.arbitrationModeBadge.textContent = 'MODE: 500MS FIXED';
      }
    } else {
      if (dom.modeStandardBtn) {
        dom.modeStandardBtn.className = 'mode-toggle-btn';
      }
      if (dom.modeVartalaapBtn) {
        dom.modeVartalaapBtn.className = 'mode-toggle-btn active vartalaap-mode';
      }
      if (dom.arbitrationModeBadge) {
        dom.arbitrationModeBadge.textContent = 'MODE: DYNAMIC ETS';
      }
    }

    sendJsonMessage({ action: 'SET_MODE', mode: newMode });
  }

  // --------------------------------------------------------------------------
  // One-Click Judge Benchmark Simulations
  // --------------------------------------------------------------------------
  async function simulateHinglishPause() {
    initWebSocket();
    ensureAudioContext();

    const text = 'Mera balance... matlab';
    if (dom.transcriptBox) {
      dom.transcriptBox.classList.remove('empty-state');
      dom.transcriptBox.innerHTML = 'Mera balance... <span class="marker-highlight">matlab</span><span class="streaming-cursor"></span>';
    }

    sendJsonMessage({ action: 'SET_TRANSCRIPT', text });

    // Stream 700ms of low-level synthetic PCM frames (22 frames @ 32ms)
    const framesCount = 22;
    for (let i = 0; i < framesCount; i++) {
      const frameBuffer = createSyntheticNoiseFrame(CHUNK_SAMPLES, 50.0);
      sendBinaryAudioFrame(frameBuffer);
      await delay(32);
    }
  }

  async function simulateHaanjiBackchannel() {
    initWebSocket();
    ensureAudioContext();

    if (dom.agentResponseBox) {
      dom.agentResponseBox.classList.remove('empty-state');
      dom.agentResponseBox.textContent = 'Aapka current account balance verify kar raha hoon...';
    }
    if (dom.botPlaybackStatus) {
      dom.botPlaybackStatus.textContent = 'STREAMING';
      dom.botPlaybackStatus.style.color = 'var(--accent-violet)';
    }

    // Step 1: User says brief "haanji"
    sendJsonMessage({ action: 'SET_TRANSCRIPT', text: 'haanji' });
    if (dom.transcriptBox) {
      dom.transcriptBox.classList.remove('empty-state');
      dom.transcriptBox.innerHTML = '<span class="marker-highlight backchannel">haanji</span>';
    }

    // Step 2: Send brief speech frames (~200ms = 6 frames)
    for (let i = 0; i < 6; i++) {
      const frame = createSyntheticSpeechToneFrame(CHUNK_SAMPLES, 240.0, 8000.0);
      sendBinaryAudioFrame(frame);
      await delay(32);
    }
  }

  async function simulateRukoInterrupt() {
    initWebSocket();
    ensureAudioContext();

    if (dom.agentResponseBox) {
      dom.agentResponseBox.classList.remove('empty-state');
      dom.agentResponseBox.textContent = 'Ji, aapka EMI statement download kar raha hoon...';
    }

    // User says imperative interrupt "ruko"
    sendJsonMessage({ action: 'SET_TRANSCRIPT', text: 'ruko' });
    if (dom.transcriptBox) {
      dom.transcriptBox.classList.remove('empty-state');
      dom.transcriptBox.innerHTML = '<span class="marker-highlight interrupt">ruko</span>';
    }

    // Send interrupt speech frames (~300ms = 9 frames)
    for (let i = 0; i < 9; i++) {
      const frame = createSyntheticSpeechToneFrame(CHUNK_SAMPLES, 280.0, 14000.0);
      sendBinaryAudioFrame(frame);
      await delay(32);
    }
  }

  function resetConsole() {
    sendJsonMessage({ action: 'RESET' });
    state.cutoffsPrevented = 0;
    state.backchannelsPreserved = 0;
    state.lastCutoffFlagged = false;
    state.lastBackchannelFlagged = false;
    state.turnGrantedTimestamp = 0;
    state.ttfaMeasured = null;

    if (dom.cutoffsPreventedVal) dom.cutoffsPreventedVal.textContent = '0';
    if (dom.backchannelsPreservedVal) dom.backchannelsPreservedVal.textContent = '0';
    if (dom.ttfaMetricVal) dom.ttfaMetricVal.textContent = '--';
    if (dom.silenceBar) dom.silenceBar.style.width = '0%';
    if (dom.elapsedSilenceVal) dom.elapsedSilenceVal.textContent = '0';
    if (dom.pIncompleteVal) dom.pIncompleteVal.textContent = '0.0%';
    if (dom.pCompleteVal) dom.pCompleteVal.textContent = '0.0%';
    if (dom.pIncompleteFill) dom.pIncompleteFill.style.width = '0%';
    if (dom.pCompleteFill) dom.pCompleteFill.style.width = '0%';
    if (dom.categoryBadge) dom.categoryBadge.textContent = 'NEUTRAL';
    if (dom.matchedMarkersList) dom.matchedMarkersList.innerHTML = '<span class="marker-tag">none</span>';
    if (dom.transcriptBox) {
      dom.transcriptBox.className = 'transcript-stream-box empty-state';
      dom.transcriptBox.textContent = 'Press "Start Conversation" or click a simulation preset...';
    }
    if (dom.agentResponseBox) {
      dom.agentResponseBox.className = 'transcript-stream-box empty-state';
      dom.agentResponseBox.textContent = 'Awaiting conversational turn completion...';
    }

    updateTurnStateDisplay('LISTENING');
    applyGainDucking('RESTORE', 0.0);
  }

  // --------------------------------------------------------------------------
  // Audio Generation Helpers for Judge Demos
  // --------------------------------------------------------------------------
  function createSyntheticNoiseFrame(numSamples, amplitude) {
    const buffer = new ArrayBuffer(numSamples * 2);
    const view = new Int16Array(buffer);
    for (let i = 0; i < numSamples; i++) {
      view[i] = Math.round((Math.random() * 2 - 1) * amplitude);
    }
    return buffer;
  }

  function createSyntheticSpeechToneFrame(numSamples, freq, amplitude) {
    const buffer = new ArrayBuffer(numSamples * 2);
    const view = new Int16Array(buffer);
    for (let i = 0; i < numSamples; i++) {
      const t = i / TARGET_SAMPLE_RATE;
      const sample = Math.sin(2 * Math.PI * freq * t) * amplitude;
      view[i] = Math.round(Math.max(-32767, Math.min(32767, sample)));
    }
    return buffer;
  }

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // --------------------------------------------------------------------------
  // Event Listeners Binding
  // --------------------------------------------------------------------------
  function initEventBindings() {
    if (dom.sessionBtn) {
      dom.sessionBtn.addEventListener('click', () => {
        if (!state.isRecording) {
          startConversation();
        } else {
          stopConversation();
        }
      });
    }

    if (dom.resetBtn) {
      dom.resetBtn.addEventListener('click', resetConsole);
    }

    if (dom.modeStandardBtn) {
      dom.modeStandardBtn.addEventListener('click', () => setMode('STANDARD'));
    }

    if (dom.modeVartalaapBtn) {
      dom.modeVartalaapBtn.addEventListener('click', () => setMode('VARTALAAP'));
    }

    if (dom.simHinglishPauseBtn) {
      dom.simHinglishPauseBtn.addEventListener('click', simulateHinglishPause);
    }

    if (dom.simHaanjiBtn) {
      dom.simHaanjiBtn.addEventListener('click', simulateHaanjiBackchannel);
    }

    if (dom.simRukoBtn) {
      dom.simRukoBtn.addEventListener('click', simulateRukoInterrupt);
    }
  }

  // --------------------------------------------------------------------------
  // System Health Probe
  // --------------------------------------------------------------------------
  async function probeSystemHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (dom.gpuStatusDot) {
          if (data.torch_cuda) {
            dom.gpuStatusDot.className = 'status-dot green pulse';
          } else {
            dom.gpuStatusDot.className = 'status-dot amber';
          }
        }
      }
    } catch (_) {
      // Offline or starting up
    }
  }

  // --------------------------------------------------------------------------
  // Bootstrap Application
  // --------------------------------------------------------------------------
  window.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    initOscilloscope();
    initEventBindings();
    probeSystemHealth();
  });
})();
