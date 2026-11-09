"""Integration Test Suite for Project Vartalaap Server.

Validates the full-duplex FastAPI WebSocket server and REST endpoints:
1. Healthcheck Endpoint (GET /api/health)
2. WebSocket Connection Lifecycle and Ping-Pong
3. Control Message Processing (SET_MODE, RESET)
4. Full-Duplex Audio Streaming & Real-Time Telemetry Packets
5. Conversational Turn Grant, Dynamic ETS Expiry, and TTS Audio Chunk Streaming
6. Clean WebSocket Disconnection and Resource Teardown
7. Conversational Barge-In Interruption and Playback Hard-Flush
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.main import app

# Silence verbose loggers for clean test output
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def receive_until_type(
    ws: Any,
    expected_type: str,
    max_reads: int = 40,
) -> Optional[Dict[str, Any]]:
    """Helper to poll WebSocket messages until a specific type arrives."""
    for _ in range(max_reads):
        try:
            msg = ws.receive_json()
            if msg.get("type") == expected_type:
                return msg
        except Exception:
            break
    return None


def test_health_endpoint(client: TestClient) -> bool:
    """Validate GET /api/health endpoint structure, GPU acceleration status, and credentials flags."""
    print("\n" + "=" * 75)
    print(" [TEST 1] Server Healthcheck Endpoint (GET /api/health)")
    print("=" * 75)

    response = client.get("/api/health")
    print(f"  HTTP Status Code : {response.status_code}")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    print(f"  Payload Response : {data}")

    assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
    assert "gpu" in data, "Missing 'gpu' device info in response"
    assert "torch_cuda" in data and isinstance(data["torch_cuda"], bool), "'torch_cuda' must be boolean"
    assert "gnani_stt" in data and isinstance(data["gnani_stt"], bool), "'gnani_stt' must be boolean"
    assert "gnani_tts" in data and isinstance(data["gnani_tts"], bool), "'gnani_tts' must be boolean"

    print(f"  GPU Hardware     : {data['gpu']} (CUDA Active: {data['torch_cuda']})")
    print(f"  Gnani Services   : STT={data['gnani_stt']}, TTS={data['gnani_tts']}")
    print("  ✅ PASS: Healthcheck endpoint verified successfully.")
    return True


def test_root_redirect(client: TestClient) -> bool:
    """Validate root URL serves or redirects to static conversation console."""
    print("\n" + "=" * 75)
    print(" [TEST 2] Root Path Serving (GET /)")
    print("=" * 75)

    response = client.get("/", follow_redirects=False)
    print(f"  HTTP Status Code : {response.status_code}")
    assert response.status_code in (200, 301, 302, 307, 308), f"Expected 200 or redirect status, got {response.status_code}"

    if response.status_code in (301, 302, 307, 308):
        location = response.headers.get("location", "")
        print(f"  Redirect Target  : {location}")
        assert "index.html" in location, f"Expected redirect to index.html, got {location}"
    else:
        assert len(response.content) > 1000, "Expected non-empty HTML at root"

    # Verify static file content is accessible
    static_resp = client.get("/static/index.html")
    assert static_resp.status_code == 200, f"Expected 200 OK for /static/index.html, got {static_resp.status_code}"
    print(f"  Static Console Size : {len(static_resp.content)} bytes")
    print("  ✅ PASS: Root endpoint and static console verified.")
    return True
    return True


def test_websocket_ping_pong(client: TestClient) -> bool:
    """Validate WebSocket connection lifecycle and control PING / PONG protocol."""
    print("\n" + "=" * 75)
    print(" [TEST 3] WebSocket Connection Lifecycle & Ping-Pong")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        t0 = time.perf_counter()
        ws.send_json({"action": "PING"})
        msg = ws.receive_json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Ping Sent -> Received: {msg}")
        print(f"  Round-Trip Latency   : {latency_ms:.2f} ms")

        assert msg.get("type") == "PONG", f"Expected message type 'PONG', got {msg.get('type')}"
        assert "timestamp" in msg, "Expected 'timestamp' field in PONG message"

    print("  ✅ PASS: WebSocket connected, verified PING/PONG and disconnected cleanly.")
    return True


def test_control_mode_switch(client: TestClient) -> bool:
    """Validate runtime arbitration mode switching via SET_MODE control frame."""
    print("\n" + "=" * 75)
    print(" [TEST 4] Control Message Processing (SET_MODE: STANDARD)")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        # Switch to STANDARD mode
        ws.send_json({"action": "SET_MODE", "mode": "STANDARD"})
        ack = ws.receive_json()
        print(f"  Mode Change ACK : {ack}")

        assert ack.get("type") == "CONTROL_ACK", f"Expected 'CONTROL_ACK', got {ack.get('type')}"
        assert ack.get("action") == "SET_MODE", f"Expected action 'SET_MODE', got {ack.get('action')}"
        assert ack.get("mode") == "STANDARD", f"Expected mode 'STANDARD', got {ack.get('mode')}"
        assert ack.get("status") == "ok", f"Expected status 'ok', got {ack.get('status')}"

        # Test RESET command
        ws.send_json({"action": "RESET"})
        reset_ack = ws.receive_json()
        print(f"  Reset Command ACK: {reset_ack}")
        assert reset_ack.get("type") == "CONTROL_ACK"
        assert reset_ack.get("action") == "RESET"
        assert reset_ack.get("status") == "ok"

    print("  ✅ PASS: Control messages executed and acknowledged properly.")
    return True


def test_audio_streaming_telemetry(client: TestClient) -> bool:
    """Validate full-duplex binary PCM audio chunk streaming and real-time telemetry emission."""
    print("\n" + "=" * 75)
    print(" [TEST 5] Audio Streaming & Real-Time Telemetry Emission")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        # Set mode to STANDARD (fixed 500ms baseline)
        ws.send_json({"action": "SET_MODE", "mode": "STANDARD"})
        _ = ws.receive_json()

        # Stream 20 silent 16kHz 16-bit mono PCM chunks (512 samples = 1024 bytes each)
        chunk_bytes = b"\x00" * 1024
        num_chunks = 20

        print(f"  Streaming {num_chunks} raw 1024-byte PCM audio chunks (640ms total duration)...")
        for _ in range(num_chunks):
            ws.send_bytes(chunk_bytes)

        # Collect emitted telemetry packets
        telemetry_packets: List[Dict[str, Any]] = []
        for _ in range(num_chunks + 5):
            try:
                msg = ws.receive_json()
                if msg.get("type") == "TELEMETRY":
                    telemetry_packets.append(msg)
            except Exception:
                break

        print(f"  Telemetry Packets Received: {len(telemetry_packets)}")
        assert len(telemetry_packets) >= 15, f"Expected at least 15 telemetry packets, got {len(telemetry_packets)}"

        # Validate packet fields
        sample_packet = telemetry_packets[-1]
        print(f"  Sample Telemetry Record   : {sample_packet}")

        assert "state" in sample_packet, "Missing 'state' in telemetry"
        assert "elapsed_silence_ms" in sample_packet, "Missing 'elapsed_silence_ms'"
        assert "dynamic_ets_ms" in sample_packet, "Missing 'dynamic_ets_ms'"
        assert "p_complete" in sample_packet, "Missing 'p_complete'"
        assert "category" in sample_packet, "Missing 'category'"
        assert "ducking_gain_db" in sample_packet, "Missing 'ducking_gain_db'"
        assert "reason" in sample_packet, "Missing 'reason'"

        # In STANDARD mode, dynamic ETS must equal fixed 500ms
        assert sample_packet["dynamic_ets_ms"] == 500.0, (
            f"Expected fixed 500.0ms ETS in STANDARD mode, got {sample_packet['dynamic_ets_ms']}"
        )

    print("  ✅ PASS: Full-duplex audio chunk ingestion and structured telemetry verified.")
    return True


def test_turn_grant_and_tts_streaming(client: TestClient) -> bool:
    """Validate complete phrase evaluation, dynamic ETS turn grant, and TTS audio chunk streaming."""
    print("\n" + "=" * 75)
    print(" [TEST 6] Conversational Turn Grant & TTS Response Audio Streaming")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        # Ensure VARTALAAP dynamic mode is active
        ws.send_json({"action": "SET_MODE", "mode": "VARTALAAP"})
        _ = ws.receive_json()

        # Inject terminal closure phrase ("bhar diya hai" -> -0.65 reward, ETS ~250ms)
        phrase = "Mera electricity bill bhar diya hai"
        ws.send_json({"action": "INJECT_TRANSCRIPT", "text": phrase})
        tx_ack = ws.receive_json()
        print(f"  Injected Utterance : '{tx_ack.get('text')}'")
        assert tx_ack.get("type") == "TRANSCRIPT_STREAM"

        # Stream silent chunks (each 32ms) until silence reaches dynamic ETS (250ms)
        print("  Streaming pause frames to trigger turn grant...")
        for _ in range(12):
            ws.send_bytes(b"\x00" * 1024)

        # Collect response events
        bot_turn_start: Optional[Dict[str, Any]] = None
        audio_chunks: List[Dict[str, Any]] = []

        for _ in range(35):
            try:
                msg = ws.receive_json()
                msg_type = msg.get("type")
                if msg_type == "BOT_TURN_START":
                    bot_turn_start = msg
                elif msg_type == "AUDIO_CHUNK":
                    audio_chunks.append(msg)
                    if bot_turn_start and len(audio_chunks) >= 2:
                        break
            except Exception:
                break

        print(f"  BOT_TURN_START Event : {bot_turn_start}")
        print(f"  AUDIO_CHUNK Packets  : {len(audio_chunks)} received")

        assert bot_turn_start is not None, "BOT_TURN_START event was not emitted"
        assert bot_turn_start.get("text"), "BOT_TURN_START text must not be empty"
        print(f"  Bot Synthesis Text   : \"{bot_turn_start.get('text')}\"")

        assert len(audio_chunks) > 0, "No AUDIO_CHUNK packets were streamed"
        first_chunk = audio_chunks[0]
        assert first_chunk.get("format") == "audio/wav", f"Expected format 'audio/wav', got {first_chunk.get('format')}"
        assert len(first_chunk.get("audio_base64", "")) > 0, "audio_base64 chunk data is empty"
        print(f"  First Audio Chunk Size: {len(first_chunk['audio_base64'])} base64 chars")

    print("  ✅ PASS: Conversational turn grant triggered and TTS audio chunks streamed.")
    return True


def generate_formant_speech_frame(sr: int = 16000, n_samples: int = 512) -> bytes:
    """Synthesize a 512-sample human formant speech frame that triggers neural VAD."""
    pulses = np.zeros(n_samples, dtype=np.float32)
    period = int(sr / 120)  # 120Hz fundamental pitch
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

    pcm_int16 = (filtered * 32767.0).astype(np.int16)
    return pcm_int16.tobytes()


def test_barge_in_interruption(client: TestClient) -> bool:
    """Validate conversational barge-in: cancel in-flight TTS playback and emit HARD_FLUSH."""
    print("\n" + "=" * 75)
    print(" [TEST 7] Conversational Barge-In Interruption & Playback Hard-Flush")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        ws.send_json({"action": "SET_MODE", "mode": "VARTALAAP"})
        _ = ws.receive_json()

        # Trigger turn grant to start bot speaking
        ws.send_json({"action": "INJECT_TRANSCRIPT", "text": "Mera electricity bill bhar diya hai"})
        _ = ws.receive_json()

        for _ in range(12):
            ws.send_bytes(b"\x00" * 1024)

        # Wait for bot playback to start
        turn_started = receive_until_type(ws, "BOT_TURN_START", max_reads=25)
        assert turn_started is not None, "Failed to initiate bot playback for barge-in test"
        print("  Bot playback active. Simulating user barge-in with imperative interrupt 'ruko'...")

        # Inject imperative interrupt "ruko" while bot is actively speaking
        ws.send_json({"action": "INJECT_TRANSCRIPT", "text": "ruko"})

        # Stream vocal speech chunk into server while bot is speaking
        speech_pcm = generate_formant_speech_frame()
        ws.send_bytes(speech_pcm)

        # Look for CONTROL: HARD_FLUSH packet
        hard_flush_detected = False
        for _ in range(40):
            try:
                msg = ws.receive_json()
                if msg.get("type") == "CONTROL" and msg.get("action") == "HARD_FLUSH":
                    hard_flush_detected = True
                    print(f"  Received Control Directive: {msg}")
                    break
            except Exception:
                break

        assert hard_flush_detected, "Server did not emit CONTROL HARD_FLUSH on user barge-in"

    print("  ✅ PASS: In-flight playback cancelled cleanly and HARD_FLUSH emitted.")
    return True


def main() -> int:
    """Execute complete server integration test suite."""
    print("=" * 75)
    print(" 🚀 PROJECT VARTALAAP - FULL-DUPLEX WEBSOCKET SERVER INTEGRATION SUITE")
    print("=" * 75)

    tests = [
        ("Healthcheck Endpoint", test_health_endpoint),
        ("Root Redirect & Static HUD", test_root_redirect),
        ("WebSocket Lifecycle & Ping-Pong", test_websocket_ping_pong),
        ("Control Mode Switch", test_control_mode_switch),
        ("Audio Streaming & Telemetry", test_audio_streaming_telemetry),
        ("Turn Grant & TTS Audio Streaming", test_turn_grant_and_tts_streaming),
        ("Barge-In Interruption & Hard-Flush", test_barge_in_interruption),
    ]

    passed_count = 0
    with TestClient(app) as client:
        for name, test_fn in tests:
            try:
                if test_fn(client):
                    passed_count += 1
            except Exception as exc:
                print(f"  ❌ FAILED [{name}]: {exc}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 75)
    print(f" SUMMARY: {passed_count}/{len(tests)} Tests Passed Successfully")
    print("=" * 75)

    if passed_count == len(tests):
        print(" 🎉 ALL SERVER INTEGRATION TESTS COMPLETED SUCCESSFULLY!\n")
        return 0
    else:
        print(" ❌ SOME TESTS ENCOUNTERED FAILURES.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
