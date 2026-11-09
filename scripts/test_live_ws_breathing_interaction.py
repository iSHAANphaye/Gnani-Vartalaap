"""End-to-End WebSocket Live Interaction Verification for Breathing & Repetition."""

import asyncio
import base64
import json
import sys
from pathlib import Path
import websockets
import numpy as np


def generate_formant_speech_chunk(sr: int = 16000, n_samples: int = 512) -> bytes:
    pulses = np.zeros(n_samples, dtype=np.float32)
    period = int(sr / 120)
    pulses[::period] = 1.0

    def biquad(samples: np.ndarray, freq: float, bw: float) -> np.ndarray:
        r = np.exp(-np.pi * bw / sr)
        theta = 2.0 * np.pi * freq / sr
        a1 = -2.0 * r * np.cos(theta)
        a2 = r * r
        b0 = 1.0 - r

        y = np.zeros_like(samples)
        y1, y2 = 0.0, 0.0
        for i in range(len(samples)):
            val = b0 * samples[i] - a1 * y1 - a2 * y2
            y[i] = val
            y2 = y1
            y1 = val
        return y

    filtered = biquad(pulses, 730.0, 90.0)
    filtered = biquad(filtered, 1090.0, 110.0)
    filtered = biquad(filtered, 2440.0, 150.0)
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = 0.85 * (filtered / max_val)
    return (filtered * 32767.0).astype(np.int16).tobytes()

def generate_synthetic_breath_chunk(n_samples: int = 512, target_rms: float = 0.008) -> bytes:
    rng = np.random.default_rng(42)
    white = rng.standard_normal(n_samples).astype(np.float32)
    filtered = np.convolve(white, [0.3, 0.4, 0.3], mode="same")
    current_rms = np.sqrt(np.mean(filtered**2))
    if current_rms > 0:
        filtered = filtered * (target_rms / current_rms)
    pcm = (np.clip(filtered, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()

async def run_live_test():
    uri = "ws://localhost:7860/ws/audio"
    print(f"Connecting to live WebSocket at {uri}...")

    async with websockets.connect(uri) as ws:
        # 1. PING verification
        await ws.send(json.dumps({"action": "PING"}))
        pong = await ws.recv()
        print("  ✓ PONG received:", pong)

        # 2. Inject question
        prompt = "Mera electricity bill kitna hai?"
        print(f"  Injecting prompt: '{prompt}'")
        await ws.send(json.dumps({"action": "INJECT_TRANSCRIPT", "text": prompt}))

        # 3. Stream silence to trigger turn grant
        print("  Streaming silence frames to trigger turn grant...")
        for _ in range(12):
            await ws.send(b"\x00" * 1024)
            await asyncio.sleep(0.03)

        # 4. Wait for BOT_TURN_START
        bot_started = False
        received_chunks = 0
        hard_flushed = False

        while not bot_started:
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(msg_raw)
            if msg.get("type") == "BOT_TURN_START":
                bot_started = True
                print(f"  ✓ BOT_TURN_START received: '{msg.get('text')}'")

        # 5. While bot is speaking, stream 15 breathing frames (480ms of pink noise)
        print("  Simulating user breathing into mic while bot is speaking (15 frames @ RMS 0.008)...")
        breath_frame = generate_synthetic_breath_chunk(target_rms=0.008)

        for _ in range(15):
            await ws.send(breath_frame)
            await asyncio.sleep(0.03)

        # Read incoming messages during/after breathing: verify NO HARD_FLUSH occurs
        bot_turn_ended = False
        second_turn_started = False

        while not bot_turn_ended:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
                msg = json.loads(msg_raw)
                mtype = msg.get("type")

                if mtype == "AUDIO_CHUNK":
                    received_chunks += 1
                elif mtype == "CONTROL":
                    print(f"  Received CONTROL event: {msg}")
                    if msg.get("action") == "HARD_FLUSH":
                        hard_flushed = True
                elif mtype == "BOT_TURN_END":
                    bot_turn_ended = True
                    print(f"  ✓ BOT_TURN_END received! (Total audio chunks: {received_chunks})")
            except asyncio.TimeoutError:
                break

        assert not hard_flushed, "FAIL: Breathing falsely triggered HARD_FLUSH!"
        assert received_chunks > 10, f"FAIL: Expected bot audio chunks, got {received_chunks}"
        print("  ✓ PASS: Bot spoke completely without being interrupted by breathing!")

        # 6. Now simulate 30 silent frames after bot finishes speaking
        print("  Simulating silence after turn (user does not speak)...")
        for _ in range(30):
            await ws.send(b"\x00" * 1024)
            await asyncio.sleep(0.03)

        # Check if any new BOT_TURN_START arrives in the next 1.5 seconds
        try:
            while True:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                msg = json.loads(msg_raw)
                if msg.get("type") == "BOT_TURN_START":
                    second_turn_started = True
                    print(f"  ❌ UNEXPECTED BOT_TURN_START: '{msg.get('text')}'")
        except asyncio.TimeoutError:
            pass

        assert not second_turn_started, "FAIL: Bot repeated the answer from the start!"
        print("  ✓ PASS: Zero answer repetition! Silence cleanly maintained in LISTENING state.")

        # 7. Verification: Real Barge-In ("ruko") STILL stops bot immediately
        print("\n  Testing genuine barge-in with 'ruko' during bot playback...")
        prompt2 = "Tell me about this project and architecture"
        await ws.send(json.dumps({"action": "INJECT_TRANSCRIPT", "text": prompt2}))
        for _ in range(12):
            await ws.send(b"\x00" * 1024)
            await asyncio.sleep(0.03)

        bot_started2 = False
        while not bot_started2:
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(msg_raw)
            if msg.get("type") == "BOT_TURN_START":
                bot_started2 = True
                print(f"  ✓ Bot began answering: '{msg.get('text')[:35]}...'")

        # Now send speech frame with 'ruko'
        print("  Injecting barge-in command: 'ruko' + vocal speech frame...")
        await ws.send(json.dumps({"action": "INJECT_TRANSCRIPT", "text": "ruko"}))
        vocal_frame = generate_formant_speech_chunk()  # RMS 0.08 is clearly vocal energy
        for _ in range(8):
            await ws.send(vocal_frame)
            await asyncio.sleep(0.03)

        barge_in_hard_flush = False
        for _ in range(25):
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                msg = json.loads(msg_raw)
                if msg.get("type") == "CONTROL" and msg.get("action") == "HARD_FLUSH":
                    barge_in_hard_flush = True
                    print(f"  ✓ HARD_FLUSH confirmed on 'ruko': {msg.get('reason')}")
                    break
            except asyncio.TimeoutError:
                break

        assert barge_in_hard_flush, "FAIL: 'ruko' did not trigger HARD_FLUSH!"
        print("  ✓ PASS: Genuine barge-in reliably stops the bot immediately!")

        print("\n" + "=" * 70)
        print(" 🎉 ALL LIVE WEBSOCKET FULL-DUPLEX CHECKS PASSED PERFECTLY!")
        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_live_test())
