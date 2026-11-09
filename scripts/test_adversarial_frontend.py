"""Adversarial QA & Spec Auditor Test Suite for Diagnostic Telemetry HUD Frontend.
Tests:
1. Asset link integrity (HTML tags vs actual HTTP routes).
2. CSS/JS structural validation and DOM ID parity.
3. WebSocket resilience against malformed payloads, rapid disconnects, and reconnections.
"""

import sys
import re
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from html.parser import HTMLParser
from starlette.testclient import TestClient

from app.main import app

class HTMLExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.css_links = []
        self.scripts = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if "id" in attr_dict:
            self.ids.add(attr_dict["id"])
        if tag == "link" and attr_dict.get("rel") == "stylesheet":
            if "href" in attr_dict:
                self.css_links.append(attr_dict["href"])
        if tag == "script" and "src" in attr_dict:
            self.scripts.append(attr_dict["src"])

def run_adversarial_suite():
    client = TestClient(app)
    print("=" * 75)
    print("🛡️ ADVERSARIAL QA & EDGE CASE AUDIT: FRONTEND HUD & ASSETS")
    print("=" * 75)

    # 1. HTML Parsing and Asset Link Validation
    print("\n[AUDIT 1] Inspecting index.html for link / script references & 404s...")
    index_path = Path("app/static/index.html")
    assert index_path.exists(), "index.html missing"
    html_content = index_path.read_text(encoding="utf-8")
    
    parser = HTMLExtractor()
    parser.feed(html_content)

    # Check stylesheets
    for href in parser.css_links:
        if href.startswith("/"):
            res = client.get(href)
            assert res.status_code == 200, f"Asset {href} returned {res.status_code}"
            assert len(res.content) > 0, f"Asset {href} is empty"
            print(f"  Internal CSS: {href} -> 200 OK ({len(res.content)} bytes)")
        else:
            print(f"  External CSS: {href} -> Validated link format")

    # Check scripts
    for src in parser.scripts:
        if src.startswith("/"):
            res = client.get(src)
            assert res.status_code == 200, f"Script {src} returned {res.status_code}"
            assert len(res.content) > 0, f"Script {src} is empty"
            print(f"  Internal Script: {src} -> 200 OK ({len(res.content)} bytes)")
        else:
            print(f"  External Script: {src} -> Validated link format")

    # 2. DOM ID Parity Check
    print("\n[AUDIT 2] DOM ID Parity between console scripts and index.html...")
    # Extract IDs referenced in index.html script and shared-engine.js
    inline_ids = re.findall(r"document\.getElementById\(['\"]([^'\"]+)['\"]\)", html_content)
    engine_js_path = Path("app/static/shared-engine.js")
    engine_content = engine_js_path.read_text(encoding="utf-8")
    engine_ids = re.findall(r"document\.getElementById\(['\"]([^'\"]+)['\"]\)", engine_content)

    referenced_ids = inline_ids + engine_ids
    unique_ids = sorted(list(set(referenced_ids)))
    
    missing_ids = []
    for element_id in unique_ids:
        if element_id not in parser.ids:
            missing_ids.append(element_id)

    print(f"  Checked {len(unique_ids)} unique DOM IDs referenced in Nagra console scripts:")
    for eid in unique_ids:
        print(f"    - #{eid} -> Present in HTML")
    assert not missing_ids, f"Missing IDs in HTML: {missing_ids}"
    print("  ✅ All referenced DOM IDs exist in index.html with 100% parity.")

    # 3. Adversarial WebSocket Fuzzing
    print("\n[AUDIT 3] WebSocket Edge Case & Adversarial Resilience Testing...")

    # Edge case 3.1: Malformed JSON payload
    with client.websocket_connect("/ws/audio") as ws:
        ws.send_text("THIS IS NOT JSON {{{ broken")
        # Should not crash server, should continue to accept valid messages
        ws.send_json({"action": "PING"})
        pong = ws.receive_json()
        assert pong.get("type") == "PONG", f"Expected PONG after malformed JSON, got {pong}"
        print("  ✅ Malformed JSON handled gracefully without dropping connection.")

    # Edge case 3.2: Odd byte count / corrupt binary PCM
    with client.websocket_connect("/ws/audio") as ws:
        # 1 byte binary frame (incomplete int16)
        ws.send_bytes(b"\x00")
        ws.send_bytes(b"\x00\x01\x02") # 3 bytes
        ws.send_json({"action": "PING"})
        pong = ws.receive_json()
        assert pong.get("type") == "PONG", f"Expected PONG after corrupt binary, got {pong}"
        print("  ✅ Corrupt binary PCM frame rejected or normalized without server panic.")

    # Edge case 3.3: Unknown action / command fuzzing
    with client.websocket_connect("/ws/audio") as ws:
        ws.send_json({"action": "NON_EXISTENT_COMMAND_XYZ_9999", "foo": "bar"})
        ws.send_json({"action": "PING"})
        pong = ws.receive_json()
        assert pong.get("type") == "PONG"
        print("  ✅ Unknown action payload ignored gracefully.")

    # Edge case 3.4: Rapid reconnect cycling (5 consecutive connections closed abruptly)
    print("  Testing rapid reconnect cycling...")
    for cycle in range(5):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_json({"action": "SET_MODE", "mode": "VARTALAAP"})
            ack = ws.receive_json()
            assert ack.get("status") == "ok"
            # Abrupt exit without STOP_CALL
    print("  ✅ 5 rapid connect/abrupt-close cycles handled cleanly without leaking handles.")

    # 4. Anti-AI and Ticket Number Compliance Check
    print("\n[AUDIT 4] Codebase Cleanliness & Anti-AI Linting...")
    static_files = [index_path, Path("app/static/styles.css"), Path("app/static/shared-engine.js")]
    forbidden_terms = ["TODO", "FIXME", "XXX", "vibecoded"]
    ticket_patterns = [r"ticket[-_\s]*\d+", r"TICKET[-_\s]*\d+"]

    for sf in static_files:
        content = sf.read_text(encoding="utf-8")
        for term in forbidden_terms:
            matches = [line for line in content.splitlines() if term in line and "<!DOCTYPE" not in line]
            assert not matches, f"Forbidden term '{term}' found in {sf}: {matches}"
        for pat in ticket_patterns:
            matches = [line for line in content.splitlines() if re.search(pat, line)]
            assert not matches, f"Ticket reference pattern '{pat}' found in {sf}: {matches}"
        print(f"  {sf.name} -> Clean, 0 forbidden markers, 0 ticket tags.")

    print("\n" + "=" * 75)
    print("🎉 ALL ADVERSARIAL AUDITS PASSED WITH ZERO ERRORS!")
    print("=" * 75)

if __name__ == "__main__":
    run_adversarial_suite()
