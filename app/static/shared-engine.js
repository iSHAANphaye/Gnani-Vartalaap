/**
 * Project Vartalaap - Shared Telemetry & Simulation Engine
 * Powers all 10 visual templates with real-time WebSocket telemetry,
 * offline simulation fallbacks, realistic Indic conversational turns,
 * and audio synthesis for demos.
 */

(function (window) {
  'use strict';

  // Highlight active nav item based on current URL
  function initNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.template-nav a.nav-item');
    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href && (currentPath.endsWith(href) || currentPath.endsWith(href.split('/').pop()))) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNav);
  } else {
    initNav();
  }

  class VartalaapEngineClass {
    constructor() {
      this.ws = null;
      this.isConnected = false;
      this.isSimulating = false;
      this.isMicActive = false;
      this.audioContext = null;
      this.analyser = null;
      this.micStream = null;
      this.scriptNode = null;

      // Current State Data
      this.mode = 'VARTALAAP'; // Active turn-taking mode: 'VARTALAAP' or 'STANDARD'
      this.state = 'IDLE'; // IDLE, LISTENING, USER_PAUSE, BACKCHANNEL, BOT_SPEAKING, HARD_INTERRUPT, TURN_GRANTED
      this.category = 'NEUTRAL';
      this.pComplete = 0.0;
      this.silenceMs = 0;
      this.dynamicEtsMs = 600;
      this.gainDb = 0.0;
      this.botIsSpeaking = false;
      this.transcript = '';
      this.botResponse = '';
      this.audioLevel = 0; // 0.0 to 1.0

      // Callbacks
      this.listeners = {
        stateChange: [],
        telemetry: [],
        audioLevel: [],
        transcript: [],
        control: [],
        log: []
      };

      // Sound synthesis oscillator for simulation cues
      this.synthAudio = true;

      // Auto-connect on load
      this.connectWebSocket();
    }

    on(event, callback) {
      if (this.listeners[event]) {
        this.listeners[event].push(callback);
      }
    }

    emit(event, data) {
      if (this.listeners[event]) {
        this.listeners[event].forEach(cb => {
          try { cb(data); } catch (e) { console.error('Listener error:', e); }
        });
      }
    }

    log(msg, level = 'info') {
      const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false }) + '.' + Math.floor(Date.now() % 1000).toString().padStart(3, '0');
      this.emit('log', { timestamp, msg, level });
    }

    connectWebSocket() {
      if (window.location.protocol === 'file:') {
        this.log('Running in local file mode — simulation engine active', 'warn');
        this.updateConnectionStatus(false);
        return;
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/audio`;

      try {
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          this.isConnected = true;
          this.log('Connected to Vartalaap Real-Time WebSocket', 'success');
          this.updateConnectionStatus(true);
          // Ping to verify
          this.sendJson({ action: 'PING' });
        };

        this.ws.onmessage = (event) => {
          if (typeof event.data === 'string') {
            try {
              const msg = JSON.parse(event.data);
              this.handleWsMessage(msg);
            } catch (err) {
              console.error('Failed to parse WS JSON:', err);
            }
          } else if (event.data instanceof ArrayBuffer) {
            // Binary audio chunk from bot
            this.handleAudioFrame(event.data);
          }
        };

        this.ws.onclose = () => {
          this.isConnected = false;
          this.log('WebSocket disconnected. Operating in offline simulation mode.', 'warn');
          this.updateConnectionStatus(false);
          // Retry after delay
          setTimeout(() => this.connectWebSocket(), 5000);
        };

        this.ws.onerror = () => {
          this.isConnected = false;
          this.updateConnectionStatus(false);
        };
      } catch (err) {
        this.log('WebSocket init notice: ' + err.message, 'warn');
        this.updateConnectionStatus(false);
      }
    }

    updateConnectionStatus(connected) {
      const dots = document.querySelectorAll('.nav-status-dot');
      dots.forEach(dot => {
        if (connected) {
          dot.classList.remove('offline');
          dot.title = 'Vartalaap WebSocket Server: Online';
        } else {
          dot.classList.add('offline');
          dot.title = 'Vartalaap WebSocket Server: Simulation Mode';
        }
      });
    }

    sendJson(obj) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(obj));
      }
    }

    setMode(mode) {
      this.mode = mode;
      this.sendJson({ type: 'CONTROL', action: 'SET_MODE', mode: mode });
      this.log(`Arbitration mode switched to: ${mode}`, 'info');
      this.emit('modeChange', mode);
    }

    handleWsMessage(msg) {
      switch (msg.type) {
        case 'TELEMETRY':
          this.state = msg.state;
          this.silenceMs = msg.elapsed_silence_ms;
          this.dynamicEtsMs = msg.dynamic_ets_ms;
          this.pComplete = msg.p_complete;
          this.category = msg.category;
          this.gainDb = msg.ducking_gain_db;
          this.emit('telemetry', msg);
          this.emit('stateChange', this.state);
          break;

        case 'TRANSCRIPT_STREAM':
          this.transcript = msg.text;
          this.emit('transcript', msg);
          break;

        case 'AUDIO_CHUNK':
          if (msg.audio_base64) {
            this.playAudioChunk(msg.audio_base64);
          }
          break;

        case 'BOT_TURN_START':
          this.botIsSpeaking = true;
          this.botResponse = msg.text || '';
          this.state = 'BOT_SPEAKING';
          this.stopBotPlayback();
          this.emit('stateChange', this.state);
          this.log(`Bot began turn: "${this.botResponse}"`, 'info');
          break;

        case 'BOT_TURN_END':
          this.botIsSpeaking = false;
          this.state = 'LISTENING';
          this.emit('stateChange', this.state);
          this.log('Bot ended turn', 'info');
          break;

        case 'CONTROL':
          this.gainDb = msg.gain_db;
          this.setDuckingGain(msg.gain_db);
          if (msg.action === 'HARD_FLUSH') {
            this.stopBotPlayback();
          }
          this.emit('control', msg);
          this.log(`Control action: ${msg.action} (${msg.gain_db} dB) - ${msg.reason || ''}`, 'warn');
          break;
      }
    }

    playAudioChunk(base64Data) {
      try {
        const binaryString = window.atob(base64Data);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        if (!this.playAudioCtx) {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          this.playAudioCtx = new AudioCtx({ sampleRate: 16000 });
          this.botGainNode = this.playAudioCtx.createGain();
          this.botGainNode.connect(this.playAudioCtx.destination);
          this.nextChunkTime = this.playAudioCtx.currentTime;
          this.activeSources = [];
        }

        if (this.playAudioCtx.state === 'suspended') {
          this.playAudioCtx.resume();
        }

        // Check for 44-byte RIFF header or raw PCM
        let byteOffset = 0;
        if (bytes.length > 44 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
          byteOffset = 44;
        }

        const pcm16 = new Int16Array(bytes.buffer, byteOffset, Math.floor((bytes.byteLength - byteOffset) / 2));
        if (pcm16.length === 0) return;

        const audioBuffer = this.playAudioCtx.createBuffer(1, pcm16.length, 16000);
        const channelData = audioBuffer.getChannelData(0);
        for (let i = 0; i < pcm16.length; i++) {
          channelData[i] = pcm16[i] / 32768.0;
        }

        const source = this.playAudioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.botGainNode);

        const startTime = Math.max(this.playAudioCtx.currentTime, this.nextChunkTime || 0);
        source.start(startTime);
        this.nextChunkTime = startTime + audioBuffer.duration;
        this.activeSources = this.activeSources || [];
        this.activeSources.push(source);
        source.onended = () => {
          if (this.activeSources) {
            const idx = this.activeSources.indexOf(source);
            if (idx !== -1) this.activeSources.splice(idx, 1);
          }
        };

        // Also stimulate audio level meter during bot speech
        let sum = 0;
        for (let i = 0; i < pcm16.length; i++) {
          sum += (pcm16[i] / 32768) * (pcm16[i] / 32768);
        }
        const rms = Math.sqrt(sum / pcm16.length);
        this.audioLevel = Math.min(1.0, rms * 3.5);
        this.emit('audioLevel', this.audioLevel);
      } catch (err) {
        console.warn('Error playing audio chunk:', err);
      }
    }

    stopBotPlayback() {
      if (this.activeSources && this.activeSources.length > 0) {
        this.activeSources.forEach(s => {
          try { s.stop(); } catch (e) {}
        });
        this.activeSources = [];
      }
      if (this.playAudioCtx) {
        this.nextChunkTime = this.playAudioCtx.currentTime;
      }
    }

    setDuckingGain(gainDb) {
      if (this.botGainNode && this.playAudioCtx) {
        const linearGain = Math.max(0.0001, Math.pow(10, gainDb / 20));
        this.botGainNode.gain.setTargetAtTime(linearGain, this.playAudioCtx.currentTime, 0.05);
      }
    }

    handleAudioFrame(buffer) {
      // Calculate RMS for meter
      const pcm16 = new Int16Array(buffer);
      let sum = 0;
      for (let i = 0; i < pcm16.length; i++) {
        sum += (pcm16[i] / 32768) * (pcm16[i] / 32768);
      }
      const rms = Math.sqrt(sum / pcm16.length);
      this.audioLevel = Math.min(1.0, rms * 3.5);
      this.emit('audioLevel', this.audioLevel);
    }

    // ==========================================
    // Real Microphone Stream Support
    // ==========================================
    async toggleMic() {
      if (this.isMicActive) {
        this.stopMic();
        return false;
      } else {
        return await this.startMic();
      }
    }

    async startMic() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true
          }
        });

        this.micStream = stream;
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        source.connect(this.analyser);

        // ScriptProcessor to stream 16kHz PCM if WS is open
        this.scriptNode = this.audioContext.createScriptProcessor(512, 1, 1);
        this.scriptNode.onaudioprocess = (e) => {
          const input = e.inputBuffer.getChannelData(0);
          let sum = 0;
          const pcm16 = new Int16Array(input.length);
          for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            sum += s * s;
          }
          const rms = Math.sqrt(sum / input.length);
          this.audioLevel = Math.min(1.0, rms * 4.0);
          this.emit('audioLevel', this.audioLevel);

          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(pcm16.buffer);
          }
        };

        source.connect(this.scriptNode);
        this.scriptNode.connect(this.audioContext.destination);

        this.isMicActive = true;
        this.log('Microphone active (16kHz 16-bit PCM streaming)', 'success');
        return true;
      } catch (err) {
        this.log('Mic access error: ' + err.message, 'error');
        return false;
      }
    }

    stopMic() {
      if (this.micStream) {
        this.micStream.getTracks().forEach(t => t.stop());
        this.micStream = null;
      }
      if (this.scriptNode) {
        this.scriptNode.disconnect();
        this.scriptNode = null;
      }
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
      this.isMicActive = false;
      this.log('Microphone stopped', 'info');
    }

    // ==========================================
    // Interactive Simulation Scenarios
    // ==========================================
    triggerSound(freq = 440, type = 'sine', duration = 0.15, vol = 0.1) {
      if (!this.synthAudio) return;
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(vol, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + duration);
      } catch (e) {
        // AudioContext not unlocked yet
      }
    }

    /**
     * Scenario 1: Hinglish Mid-Utterance Pause
     * User speaks with an authentic Hindi/Hinglish pause ("kal Bangalore jaana hai toh... [pause] ...subah flight hai?")
     * Demonstrates dynamic ETS expansion to 1250ms preventing false interrupt.
     */
    simulateHinglishPause() {
      if (this.isSimulating) return;
      this.isSimulating = true;
      this.triggerSound(520, 'sine', 0.1, 0.08);

      this.log('▶ [SIMULATION] Hinglish Mid-Sentence Pause Scenario initiated', 'info');

      // Send to backend if connected
      this.sendJson({ action: 'INJECT_TRANSCRIPT', text: 'Actually mujhe kal Bangalore jaana hai toh...' });

      // Step 1: User Speaking Part 1
      this.state = 'LISTENING';
      this.transcript = 'Actually mujhe kal Bangalore jaana hai toh...';
      this.pComplete = 0.14;
      this.silenceMs = 0;
      this.dynamicEtsMs = 600;
      this.category = 'HINDI_MID_CONJUNCTION';
      this.audioLevel = 0.65;
      this.emitTelemetry();

      let pauseElapsed = 0;
      const pauseInterval = setInterval(() => {
        pauseElapsed += 100;
        this.silenceMs = pauseElapsed;
        this.audioLevel = Math.max(0.02, this.audioLevel * 0.7);

        // Smart-Turn expands ETS dynamically!
        this.dynamicEtsMs = Math.min(1280, 600 + pauseElapsed * 0.95);
        this.state = 'USER_PAUSE';
        this.category = 'HINDI_MID_CONJUNCTION';
        this.pComplete = 0.18;
        this.emitTelemetry();

        if (pauseElapsed >= 700) {
          clearInterval(pauseInterval);

          // Step 2: User resumes and completes question
          setTimeout(() => {
            this.log('User resumes after 700ms pause: "...subah ki flight mil sakti hai?"', 'info');
            this.transcript = 'Actually mujhe kal Bangalore jaana hai toh... kya subah ki flight mil sakti hai?';
            this.audioLevel = 0.78;
            this.silenceMs = 40;
            this.dynamicEtsMs = 450;
            this.pComplete = 0.96;
            this.state = 'LISTENING';
            this.emitTelemetry();

            // Step 3: Turn Granted to Bot
            setTimeout(() => {
              this.state = 'TURN_GRANTED';
              this.silenceMs = 450;
              this.pComplete = 0.98;
              this.log('✓ Turn Granted: Smart-Turn recognized sentence completion (ETS reached)', 'success');
              this.emitTelemetry();

              // Step 4: Bot begins speaking
              setTimeout(() => {
                this.botIsSpeaking = true;
                this.state = 'BOT_SPEAKING';
                this.botResponse = 'Ji bilkul! Bangalore ke liye subah 6:15 AM aur 8:40 AM ki do non-stop flights uplabdh hain.';
                this.audioLevel = 0.72;
                this.emitTelemetry();
                this.emit('transcript', { text: this.botResponse, bot: true });
                this.triggerSound(660, 'triangle', 0.25, 0.1);

                setTimeout(() => {
                  this.botIsSpeaking = false;
                  this.state = 'LISTENING';
                  this.audioLevel = 0.05;
                  this.isSimulating = false;
                  this.emitTelemetry();
                  this.log('Conversation ready for next turn', 'info');
                }, 2800);
              }, 600);
            }, 600);
          }, 300);
        }
      }, 100);
    }

    /**
     * Scenario 2: Cultural "Haanji" Backchannel
     * Bot is delivering flight info. User says "Haanji... haanji...".
     * Gate applies soft-ducking (-14 dB) without dropping the bot's turn.
     */
    simulateHaanjiBackchannel() {
      if (this.isSimulating) return;
      this.isSimulating = true;
      this.triggerSound(440, 'triangle', 0.1, 0.08);

      this.log('▶ [SIMULATION] Cultural "Haanji" Backchannel Scenario initiated', 'info');

      // Bot is currently speaking
      this.botIsSpeaking = true;
      this.state = 'BOT_SPEAKING';
      this.botResponse = 'Aapka booking confirm ho gaya hai, boarding pass 48 ghante pehle auto-generate ho jayega...';
      this.gainDb = 0.0;
      this.audioLevel = 0.68;
      this.emitTelemetry();
      this.emit('transcript', { text: this.botResponse, bot: true });

      // After 800ms, user murmurs "Haanji..."
      setTimeout(() => {
        this.log('User interjects cultural backchannel: "Haanji..."', 'warn');
        this.transcript = 'Haanji...';
        this.state = 'USER_BACKCHANNEL';
        this.category = 'INDIC_BACKCHANNEL';
        this.pComplete = 0.08;

        // Trigger Soft Ducking
        this.gainDb = -14.0;
        this.emit('control', { action: 'DUCK', gain_db: -14.0, reason: 'Cultural backchannel acknowledgment' });
        this.emitTelemetry();
        this.triggerSound(350, 'sine', 0.08, 0.05);

        // After 600ms, restore volume cleanly
        setTimeout(() => {
          this.log('Restoring bot audio to 0 dB — Bot speech uninterrupted', 'success');
          this.gainDb = 0.0;
          this.state = 'BOT_SPEAKING';
          this.category = 'NEUTRAL';
          this.emit('control', { action: 'RESTORE', gain_db: 0.0, reason: 'Backchannel window expired' });
          this.emitTelemetry();

          setTimeout(() => {
            this.botIsSpeaking = false;
            this.state = 'LISTENING';
            this.audioLevel = 0.04;
            this.isSimulating = false;
            this.emitTelemetry();
          }, 1500);
        }, 800);
      }, 900);
    }

    /**
     * Scenario 3: "Ruko" Hard Interrupt (Barge-in)
     * Bot is reading out details. User commands "Ruko ruko! Cancel karo!".
     * Hard flush cut-off (-100 dB) within <15ms. Bot yields turn immediately.
     */
    simulateRukoInterrupt() {
      if (this.isSimulating) return;
      this.isSimulating = true;
      this.triggerSound(880, 'sawtooth', 0.15, 0.12);

      this.log('▶ [SIMULATION] "Ruko" Hard Interrupt / Barge-In initiated', 'info');

      // Bot is speaking
      this.botIsSpeaking = true;
      this.state = 'BOT_SPEAKING';
      this.botResponse = 'Aapki transaction process ho rahi hai, hum aapke registered account se deduction kar...';
      this.gainDb = 0.0;
      this.audioLevel = 0.75;
      this.emitTelemetry();

      // After 700ms, user shouts "Ruko ruko!"
      setTimeout(() => {
        this.log('⚡ BARGE-IN DETECTED: "Ruko ruko, pehle cancel karo!"', 'error');
        this.transcript = 'Ruko ruko, pehle cancel karo!';
        this.state = 'HARD_INTERRUPT';
        this.gainDb = -100.0;
        this.botIsSpeaking = false;
        this.audioLevel = 0.95; // Loud user burst
        this.pComplete = 0.92;

        this.emit('control', { action: 'HARD_FLUSH', gain_db: -100.0, reason: 'Imperative Indic interrupt (Ruko)' });
        this.emitTelemetry();
        this.triggerSound(220, 'square', 0.1, 0.15);

        // Immediate turn shift to User
        setTimeout(() => {
          this.state = 'TURN_GRANTED';
          this.pComplete = 0.99;
          this.audioLevel = 0.3;
          this.emitTelemetry();
          this.log('Bot yielded immediately within 12ms. User holds floor.', 'success');

          setTimeout(() => {
            this.state = 'LISTENING';
            this.audioLevel = 0.05;
            this.isSimulating = false;
            this.emitTelemetry();
          }, 1200);
        }, 500);
      }, 800);
    }

    simulateBackchannel() {
      return this.simulateHaanjiBackchannel();
    }

    simulateHardInterrupt() {
      return this.simulateRukoInterrupt();
    }

    emitTelemetry() {
      const data = {
        type: 'TELEMETRY',
        state: this.state,
        elapsed_silence_ms: this.silenceMs,
        dynamic_ets_ms: this.dynamicEtsMs,
        p_complete: this.pComplete,
        category: this.category,
        ducking_gain_db: this.gainDb,
        reason: this.state
      };
      this.emit('telemetry', data);
      this.emit('stateChange', this.state);
      this.emit('audioLevel', this.audioLevel);
    }
  }

  window.VartalaapEngine = new VartalaapEngineClass();
})(window);
