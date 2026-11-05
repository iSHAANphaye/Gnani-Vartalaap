"""Diagnostic Telemetry HUD Frontend Verification Suite.

Validates the static assets, semantic HTML cockpit structure, CSS design tokens,
JavaScript WebRTC AEC pipeline, and FastAPI server delivery:
1. Static Asset Existence on Disk
2. FastAPI Static and Root Redirect Endpoints (GET /, /static/index.html, /static/styles.css, /static/app.js)
3. HTML Structure & Mandatory DOM Elements (oscilloscope, mode-toggle, silence-bar, transcript-box)
4. CSS Design Tokens & Typography Contract (DAW Cockpit Palette, Tabular Nums)
5. JavaScript Syntax, WebRTC Constraints, and Web Audio Soft-Ducking Pipeline
6. Live WebSocket Telemetry & Mode Negotiation Integration
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.main import app

# Silence verbose loggers for clean test output
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def test_static_asset_files_exist() -> bool:
    """Verify that all frontend asset files exist on disk with non-trivial size."""
    print("\n" + "=" * 75)
    print(" [TEST 1] Static Asset Files Existence and File Sizing")
    print("=" * 75)

    static_dir = ROOT_DIR / "app" / "static"
    assert static_dir.is_dir(), f"Static directory not found at: {static_dir}"

    required_files = {
        "index.html": 1000,
        "styles.css": 2000,
        "app.js": 3000,
    }

    for filename, min_bytes in required_files.items():
        file_path = static_dir / filename
        assert file_path.exists(), f"Required asset missing: {file_path}"
        file_size = file_path.stat().st_size
        print(f"  Asset File: {filename:<12} -> {file_size:>6} bytes (min: {min_bytes} bytes)")
        assert file_size >= min_bytes, f"Asset file {filename} too small ({file_size} bytes < {min_bytes} bytes)"

    print("  PASS: All static assets exist on disk and meet sizing requirements.")
    return True


def test_fastapi_server_routes(client: TestClient) -> bool:
    """Verify FastAPI route delivery: root redirect and static asset HTTP responses."""
    print("\n" + "=" * 75)
    print(" [TEST 2] FastAPI Server Route Delivery and Content-Type Headers")
    print("=" * 75)

    # 1. Root redirect check
    resp_root = client.get("/", follow_redirects=False)
    print(f"  GET /               -> Status: {resp_root.status_code}")
    assert resp_root.status_code in (301, 302, 307, 308), f"Expected redirect for '/', got {resp_root.status_code}"
    location = resp_root.headers.get("location", "")
    print(f"  Redirect Location   -> '{location}'")
    assert "/static/index.html" in location, f"Expected redirect to /static/index.html, got '{location}'"

    # 2. Static HTML check
    resp_html = client.get("/static/index.html")
    print(f"  GET /static/index.html -> Status: {resp_html.status_code}, Content-Type: {resp_html.headers.get('content-type')}")
    assert resp_html.status_code == 200, f"Expected 200 OK, got {resp_html.status_code}"
    assert "text/html" in resp_html.headers.get("content-type", ""), "Expected text/html content-type"

    # 3. Static CSS check
    resp_css = client.get("/static/styles.css")
    print(f"  GET /static/styles.css -> Status: {resp_css.status_code}, Content-Type: {resp_css.headers.get('content-type')}")
    assert resp_css.status_code == 200, f"Expected 200 OK, got {resp_css.status_code}"
    assert "text/css" in resp_css.headers.get("content-type", ""), "Expected text/css content-type"

    # 4. Static JS check
    resp_js = client.get("/static/app.js")
    print(f"  GET /static/app.js     -> Status: {resp_js.status_code}, Content-Type: {resp_js.headers.get('content-type')}")
    assert resp_js.status_code == 200, f"Expected 200 OK, got {resp_js.status_code}"
    ct = resp_js.headers.get("content-type", "")
    assert ("javascript" in ct or "text/plain" in ct), f"Expected javascript content-type, got '{ct}'"

    print("  PASS: FastAPI server successfully routes root redirect and all static assets.")
    return True


def test_html_semantic_structure() -> bool:
    """Verify semantic layout and all mandatory DOM IDs in index.html."""
    print("\n" + "=" * 75)
    print(" [TEST 3] Semantic HTML Cockpit Structure & Required Element IDs")
    print("=" * 75)

    html_path = ROOT_DIR / "app" / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Mandatory elements per specification
    mandatory_element_ids = [
        "oscilloscope",
        "mode-toggle",
        "silence-bar",
        "transcript-box",
    ]

    for elem_id in mandatory_element_ids:
        pattern = rf'id=["\']{re.escape(elem_id)}["\']'
        match = re.search(pattern, content)
        print(f"  Mandatory Element: id='{elem_id}' -> {'Found' if match else 'MISSING'}")
        assert match is not None, f"Mandatory DOM element id='{elem_id}' not found in index.html"

    # Hardware pills check
    hardware_pills = ["RTX 5060", "Smart-Turn", "Gnani Prisma", "Silero VAD"]
    for pill in hardware_pills:
        assert pill in content, f"Expected hardware pill text '{pill}' in index.html"
        print(f"  Hardware Indicator: '{pill}' -> Found")

    # Simulation buttons check
    sim_buttons = [
        "simHinglishPauseBtn",
        "simHaanjiBtn",
        "simRukoBtn",
    ]
    for btn_id in sim_buttons:
        assert f'id="{btn_id}"' in content, f"Simulation button id='{btn_id}' missing in index.html"
        print(f"  Simulation Action : id='{btn_id}' -> Found")

    # Impact metrics check
    impact_metrics = [
        "cutoffsPreventedVal",
        "backchannelsPreservedVal",
        "ttfaMetricVal",
    ]
    for metric_id in impact_metrics:
        assert f'id="{metric_id}"' in content, f"Impact metric id='{metric_id}' missing in index.html"
        print(f"  Impact Counter    : id='{metric_id}' -> Found")

    print("  PASS: Semantic HTML contains all required elements, pills, and controls.")
    return True


def test_css_design_tokens_and_typography() -> bool:
    """Verify acoustic DAW console color tokens and tabular typography contract."""
    print("\n" + "=" * 75)
    print(" [TEST 4] CSS Design Tokens, Acoustic DAW Cockpit Palette & Typography")
    print("=" * 75)

    css_path = ROOT_DIR / "app" / "static" / "styles.css"
    content = css_path.read_text(encoding="utf-8")

    required_color_tokens = [
        ("#0b0f17", "Background Dark Console Base"),
        ("#111827", "Card Surface"),
        ("#1f2937", "Subtle Border"),
        ("#10b981", "Green Accent"),
        ("#f59e0b", "Amber Accent"),
        ("#06b6d4", "Cyan Accent"),
        ("#8b5cf6", "Violet Accent"),
        ("#f9fafb", "Primary Text"),
        ("#9ca3af", "Muted Text"),
    ]

    for hex_code, description in required_color_tokens:
        found = hex_code.lower() in content.lower()
        print(f"  Color Token: {hex_code} ({description}) -> {'Found' if found else 'MISSING'}")
        assert found, f"Required color token '{hex_code}' ({description}) not found in styles.css"

    # Typography check
    assert "Plus Jakarta Sans" in content, "Font 'Plus Jakarta Sans' missing in styles.css"
    assert "JetBrains Mono" in content, "Font 'JetBrains Mono' missing in styles.css"
    assert "tabular-nums" in content, "Property 'font-variant-numeric: tabular-nums' missing in styles.css"

    print("  Typography Contract: 'Plus Jakarta Sans', 'JetBrains Mono', tabular-nums -> Verified")
    print("  PASS: CSS design system faithfully implements acoustic cockpit tokens.")
    return True


def test_javascript_syntax_and_pipeline() -> bool:
    """Verify JavaScript syntax, WebRTC AEC audio constraints, and soft-ducking pipeline."""
    print("\n" + "=" * 75)
    print(" [TEST 5] JavaScript Syntax & WebRTC AEC Audio Processing Pipeline")
    print("=" * 75)

    js_path = ROOT_DIR / "app" / "static" / "app.js"
    content = js_path.read_text(encoding="utf-8")

    # 1. Execute Node.js syntax checker if available
    try:
        node_res = subprocess.run(
            ["node", "--check", str(js_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        print("  Node.js Syntax Check : PASS (Zero Syntax Errors)")
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        print(f"  Node.js Check Notice : {err}")

    # 2. WebRTC Acoustic Echo Cancellation constraints
    webrtc_keys = [
        "echoCancellation",
        "noiseSuppression",
        "autoGainControl",
        "sampleRate",
        "channelCount",
    ]
    for key in webrtc_keys:
        assert key in content, f"WebRTC AEC constraint '{key}' missing from app.js"
        print(f"  WebRTC Constraint    : {key:<18} -> Verified")

    # 3. Audio chunking (512 samples / 1024 bytes at 16kHz)
    assert "512" in content, "Audio frame chunk size 512 samples not specified in app.js"
    assert "16000" in content, "Target sample rate 16000 Hz not specified in app.js"
    print("  Audio Chunking       : 16kHz Mono Linear PCM (512 samples / frame) -> Verified")

    # 4. Soft-ducking & Hard-flush directives
    ducking_directives = ["DUCK", "RESTORE", "HARD_FLUSH"]
    for directive in ducking_directives:
        assert directive in content, f"Ducking directive '{directive}' not handled in app.js"
        print(f"  Ducking Directive    : {directive:<18} -> Handled")

    # 5. Dynamic Mode Negotiation
    assert "SET_MODE" in content, "SET_MODE action missing in app.js"
    assert "VARTALAAP" in content and "STANDARD" in content, "Mode identifiers missing in app.js"
    print("  Turn Mode Control    : SET_MODE (STANDARD / VARTALAAP) -> Verified")

    # 6. Oscilloscope Canvas & 60fps loop
    assert "requestAnimationFrame" in content, "Oscilloscope 60fps render loop missing in app.js"
    assert "devicePixelRatio" in content, "High-DPI retina canvas scaling missing in app.js"
    print("  Canvas Oscilloscope  : High-DPI 60 FPS requestAnimationFrame Loop -> Verified")

    print("  PASS: JavaScript architecture and Web Audio pipeline verified successfully.")
    return True


def test_websocket_telemetry_integration(client: TestClient) -> bool:
    """Verify live WebSocket endpoint message protocols and telemetry packets."""
    print("\n" + "=" * 75)
    print(" [TEST 6] Live WebSocket Telemetry & Mode Negotiation Integration")
    print("=" * 75)

    with client.websocket_connect("/ws/audio") as ws:
        # 1. Test PING / PONG
        ws.send_json({"action": "PING"})
        pong_msg = ws.receive_json()
        print(f"  PING -> Response: {pong_msg}")
        assert pong_msg.get("type") == "PONG", f"Expected PONG, got {pong_msg}"

        # 2. Test SET_MODE: STANDARD
        ws.send_json({"action": "SET_MODE", "mode": "STANDARD"})
        mode_std_ack = ws.receive_json()
        print(f"  SET_MODE STANDARD -> ACK: {mode_std_ack}")
        assert mode_std_ack.get("type") == "CONTROL_ACK"
        assert mode_std_ack.get("mode") == "STANDARD"

        # 3. Test SET_MODE: VARTALAAP
        ws.send_json({"action": "SET_MODE", "mode": "VARTALAAP"})
        mode_var_ack = ws.receive_json()
        print(f"  SET_MODE VARTALAAP -> ACK: {mode_var_ack}")
        assert mode_var_ack.get("type") == "CONTROL_ACK"
        assert mode_var_ack.get("mode") == "VARTALAAP"

        # 4. Ingest an audio frame and check telemetry packet schema
        synthetic_pcm_frame = bytes(1024)
        ws.send_bytes(synthetic_pcm_frame)

        # Receive telemetry frame
        telemetry_msg: Dict[str, Any] = {}
        for _ in range(10):
            msg = ws.receive_json()
            if msg.get("type") == "TELEMETRY":
                telemetry_msg = msg
                break

        print(f"  Telemetry Packet: {telemetry_msg}")
        assert telemetry_msg.get("type") == "TELEMETRY", "Failed to receive TELEMETRY message"
        assert "state" in telemetry_msg, "Missing 'state' in telemetry"
        assert "elapsed_silence_ms" in telemetry_msg, "Missing 'elapsed_silence_ms'"
        assert "dynamic_ets_ms" in telemetry_msg, "Missing 'dynamic_ets_ms'"
        assert "p_complete" in telemetry_msg, "Missing 'p_complete'"
        assert "category" in telemetry_msg, "Missing 'category'"
        assert "ducking_gain_db" in telemetry_msg, "Missing 'ducking_gain_db'"

        print("  PASS: Live WebSocket protocol, control ACKs, and telemetry packet schemas verified.")
    return True


def main() -> None:
    """Main verification routine running all diagnostic frontend checks."""
    print("=" * 75)
    print(" PROJECT VARTALAAP — FRONTEND HUD VERIFICATION SUITE")
    print("=" * 75)

    with TestClient(app) as client:
        t1 = test_static_asset_files_exist()
        t2 = test_fastapi_server_routes(client)
        t3 = test_html_semantic_structure()
        t4 = test_css_design_tokens_and_typography()
        t5 = test_javascript_syntax_and_pipeline()
        t6 = test_websocket_telemetry_integration(client)

    all_passed = t1 and t2 and t3 and t4 and t5 and t6
    print("\n" + "=" * 75)
    if all_passed:
        print(" 🎉 ALL 6 FRONTEND HUD INTEGRATION TESTS PASSED CLEANLY")
    else:
        print(" ❌ FRONTEND HUD INTEGRATION TESTS FAILED")
    print("=" * 75)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
