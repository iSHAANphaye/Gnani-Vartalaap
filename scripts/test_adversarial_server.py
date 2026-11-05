"""Adversarial and Edge Case Verification Test Suite for Project Vartalaap Server.

Tests:
1. Concurrent WebSocket connections: Isolated session state, simultaneous streaming without crosstalk.
2. Malformed data handling: Invalid JSON strings and odd-length binary audio buffers.
3. Abrupt client disconnect during active bot TTS streaming.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.main import app

# Silence verbose loggers for clean adversarial test output
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def test_concurrent_sessions(client: TestClient) -> bool:
    """Verify concurrent WebSocket connections maintain isolated session states and receive independent telemetry."""
    print("\n" + "=" * 75)
    print(" [ADVERSARIAL TEST 1] Concurrent WebSocket Sessions & State Isolation")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws_a, client.websocket_connect("/ws/audio") as ws_b:
        print("  Client A and Client B connected simultaneously.")

        # Client A: Set mode to STANDARD (fixed 500ms ETS)
        ws_a.send_json({"action": "SET_MODE", "mode": "STANDARD"})
        ack_a = ws_a.receive_json()
        assert ack_a.get("type") == "CONTROL_ACK" and ack_a.get("mode") == "STANDARD"

        # Client B: Keep mode at VARTALAAP (dynamic ETS) and inject complete closure phrase
        ws_b.send_json({"action": "SET_MODE", "mode": "VARTALAAP"})
        ack_b = ws_b.receive_json()
        assert ack_b.get("type") == "CONTROL_ACK" and ack_b.get("mode") == "VARTALAAP"

        ws_b.send_json({"action": "INJECT_TRANSCRIPT", "text": "Mera electricity bill bhar diya hai"})
        tx_ack_b = ws_b.receive_json()
        assert tx_ack_b.get("type") == "TRANSCRIPT_STREAM"

        # Stream audio to both clients
        chunk = b"\x00" * 1024
        ws_a.send_bytes(chunk)
        ws_b.send_bytes(chunk)

        # Receive telemetry from Client A
        telemetry_a: Optional[Dict[str, Any]] = None
        for _ in range(5):
            msg = ws_a.receive_json()
            if msg.get("type") == "TELEMETRY":
                telemetry_a = msg
                break

        # Receive telemetry from Client B
        telemetry_b: Optional[Dict[str, Any]] = None
        for _ in range(5):
            msg = ws_b.receive_json()
            if msg.get("type") == "TELEMETRY":
                telemetry_b = msg
                break

        assert telemetry_a is not None, "Client A did not receive telemetry"
        assert telemetry_b is not None, "Client B did not receive telemetry"

        print(f"  Client A Telemetry: dynamic_ets_ms={telemetry_a['dynamic_ets_ms']} (Expected: 500.0)")
        print(f"  Client B Telemetry: dynamic_ets_ms={telemetry_b['dynamic_ets_ms']} (Expected: 250.0)")

        assert telemetry_a["dynamic_ets_ms"] == 500.0, "Client A dynamic_ets_ms corrupted by Client B"
        assert telemetry_b["dynamic_ets_ms"] == 250.0, "Client B dynamic_ets_ms corrupted by Client A"

    print("  ✅ PASS: Concurrent sessions demonstrated perfect state isolation without crosstalk.")
    return True


def test_malformed_data(client: TestClient) -> bool:
    """Verify server gracefully tolerates invalid JSON and partial/odd-length audio frames."""
    print("\n" + "=" * 75)
    print(" [ADVERSARIAL TEST 2] Malformed Data & Odd-Length Audio Chunk Handling")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        print("  1. Sending malformed JSON string: '{bad json, not valid...'")
        ws.send_text("{bad json, not valid...")

        # Connection must remain intact and functional
        ws.send_json({"action": "PING"})
        pong = ws.receive_json()
        assert pong.get("type") == "PONG", f"Expected PONG after malformed JSON, got {pong}"
        print("  -> Server ignored malformed JSON gracefully; PING/PONG succeeded.")

        print("  2. Sending odd-length audio chunk: 333 bytes (below 1024 frame boundary)...")
        odd_chunk = b"\x12" * 333
        ws.send_bytes(odd_chunk)

        # Send remaining 691 bytes to complete the 1024-byte boundary
        print("  -> Sending remaining 691 bytes to complete the 1024 frame...")
        remaining_chunk = b"\x12" * 691
        ws.send_bytes(remaining_chunk)

        telemetry_received = False
        for _ in range(5):
            msg = ws.receive_json()
            if msg.get("type") == "TELEMETRY":
                telemetry_received = True
                print(f"  -> Telemetry received after boundary completion: state={msg.get('state')}")
                break

        assert telemetry_received, "Server failed to process reconstructed 1024-byte frame from partial buffers"

    print("  ✅ PASS: Server handled malformed JSON and odd-length binary frames robustly.")
    return True


def test_abrupt_disconnect_during_tts(client: TestClient) -> bool:
    """Verify abrupt client disconnection during active bot TTS streaming causes no uncaught server crashes."""
    print("\n" + "=" * 75)
    print(" [ADVERSARIAL TEST 3] Abrupt Client Disconnection During Active TTS Streaming")
    print("=" * 75)

    # Open connection, trigger turn grant, and wait for BOT_TURN_START
    try:
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_json({"action": "SET_MODE", "mode": "VARTALAAP"})
            _ = ws.receive_json()

            ws.send_json({"action": "INJECT_TRANSCRIPT", "text": "Mera electricity bill bhar diya hai"})
            _ = ws.receive_json()

            for _ in range(12):
                ws.send_bytes(b"\x00" * 1024)

            bot_turn_start = None
            for _ in range(25):
                msg = ws.receive_json()
                if msg.get("type") == "BOT_TURN_START":
                    bot_turn_start = msg
                    break

            assert bot_turn_start is not None, "Failed to start bot turn playback"
            print("  Bot playback actively streaming. Abruptly disconnecting socket context...")
            # Exiting the 'with' block here causes abrupt WebSocket disconnect while TTS task is running
    except Exception as exc:
        print(f"  Unexpected exception during client close: {exc}")
        return False

    print("  WebSocket context exited. Performing post-disconnect health check on server...")
    # Health check must return 200 immediately, verifying server process didn't crash
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200, f"Expected 200 after abrupt disconnect, got {health_resp.status_code}"
    print(f"  Post-Disconnect Healthcheck: {health_resp.json()}")

    print("  ✅ PASS: Server handled abrupt client disconnect during active TTS streaming without crashing.")
    return True


def main() -> int:
    """Run all adversarial edge case tests."""
    print("=" * 75)
    print(" 🛡️ PROJECT VARTALAAP - ADVERSARIAL EDGE CASE VERIFICATION SUITE")
    print("=" * 75)

    tests = [
        ("Concurrent Sessions", test_concurrent_sessions),
        ("Malformed Data Handling", test_malformed_data),
        ("Abrupt Disconnect During TTS", test_abrupt_disconnect_during_tts),
    ]

    passed = 0
    with TestClient(app) as client:
        for name, test_fn in tests:
            try:
                if test_fn(client):
                    passed += 1
            except Exception as exc:
                print(f"  ❌ FAILED [{name}]: {exc}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 75)
    print(f" ADVERSARIAL SUMMARY: {passed}/{len(tests)} Tests Passed Successfully")
    print("=" * 75)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
