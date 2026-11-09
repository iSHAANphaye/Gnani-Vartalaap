"""Test live handshake with Gnani.ai API endpoints."""

import asyncio
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import websockets
from app.config import get_settings

async def test_live_handshake():
    settings = get_settings()
    api_key = settings.gnani_api_key.get_secret_value()
    print("Testing Live Gnani AI Connection...")
    print(f"Key Prefix: {api_key[:10]}...{api_key[-8:]}")

    # 1. Test REST endpoint reachability
    print("\n--- 1. Testing REST Endpoint: api.vachana.ai ---")
    headers = {
        "X-API-Key-ID": api_key,
        "token": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    try:
        r = requests.get("https://api.vachana.ai", headers=headers, timeout=5)
        print(f"GET https://api.vachana.ai -> Status Code: {r.status_code}")
    except Exception as e:
        print(f"GET https://api.vachana.ai failed: {e}")

    # 2. Test TTS Endpoint
    print("\n--- 2. Testing Timbre TTS Endpoint ---")
    tts_url = settings.gnani_tts_rest_url
    print(f"URL: {tts_url}")
    try:
        r_tts = requests.post(
            tts_url,
            headers={
                "X-API-Key-ID": api_key,
                "token": api_key,
                "Content-Type": "application/json"
            },
            json={
                "text": "Namaste, Vartalaap mein aapka swagat hai.",
                "voice": "Shivani",
                "model": "timbre-v3.0",
                "language": "hi-IN",
                "audio_format": "wav"
            },
            timeout=8
        )
        print(f"TTS Response Status: {r_tts.status_code}")
        print(f"TTS Response Headers: {r_tts.headers.get('content-type')}")
        if r_tts.status_code == 200:
            print(f"✓ TTS Success! Generated {len(r_tts.content)} bytes of audio")
        else:
            print(f"TTS Response Text: {r_tts.text[:300]}")
    except Exception as e:
        print(f"TTS Request error: {e}")

    # 3. Test STT Streaming WebSocket
    print("\n--- 3. Testing Prisma STT WebSocket ---")
    ws_url = settings.gnani_stt_ws_url
    print(f"URL: {ws_url}")
    try:
        ws_headers = {
            "token": api_key,
            "X-API-Key-ID": api_key,
            "language_code": "hi-IN",
            "sample_rate": "16000",
            "audio_format": "pcm",
            "partials": "true",
        }
        test_ws_url = ws_url
        try:
            connect_kwargs = {"additional_headers": ws_headers}
            ws_conn = websockets.connect(test_ws_url, **connect_kwargs)
        except TypeError:
            ws_conn = websockets.connect(test_ws_url, extra_headers=ws_headers)
        
        async with ws_conn as ws:
            print("✓ WebSocket connection opened successfully to Gnani Prisma STT!")
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print("Server handshake message:", msg)

            # Stream sample_27_terminal_balance_batao.wav
            import wave
            wav_path = "benchmark/test_samples/sample_27_terminal_balance_batao.wav"
            with wave.open(wav_path, "rb") as wf:
                audio_data = wf.readframes(wf.getnframes())

            print(f"Streaming {len(audio_data)} bytes of audio ({wav_path})...")
            chunk_size = 1024  # 512 samples * 2 bytes = 1024 bytes (32ms)

            async def send_audio():
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await ws.send(chunk)
                    await asyncio.sleep(0.032)
                # Send silence or EOS if needed
                print("Finished streaming audio.")

            async def receive_transcripts():
                try:
                    while True:
                        res = await asyncio.wait_for(ws.recv(), timeout=4.0)
                        print(f"-> STT response: {res}")
                except asyncio.TimeoutError:
                    print("Receive timeout (utterance ended or no more frames).")

            await asyncio.gather(send_audio(), receive_transcripts())
    except Exception as e:
        print(f"WebSocket connection notice: {e}")

if __name__ == "__main__":
    asyncio.run(test_live_handshake())
