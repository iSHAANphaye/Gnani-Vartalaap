"""Diagnose Gnani Prisma STT streaming with trailing silence frames."""

import asyncio
import json
import logging
import sys
import time
import wave
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import websockets
from app.config import get_settings

async def test_single_sample(wav_file: str):
    settings = get_settings()
    api_key = settings.gnani_api_key.get_secret_value()
    ws_url = settings.gnani_stt_ws_url

    headers = {
        "token": api_key,
        "X-API-Key-ID": api_key,
        "language_code": "hi-IN",
        "sample_rate": "16000",
        "audio_format": "pcm",
    }

    try:
        connect_kwargs = {"additional_headers": headers}
        ws_conn = websockets.connect(ws_url, **connect_kwargs)
    except TypeError:
        ws_conn = websockets.connect(ws_url, extra_headers=headers)

    async with ws_conn as ws:
        handshake = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print("✓ Connected. Handshake:", handshake)

        wav_path = ROOT_DIR / "benchmark" / "test_samples" / wav_file
        with wave.open(str(wav_path), "rb") as wf:
            audio_bytes = wf.readframes(wf.getnframes())

        chunk_size = 1024  # 512 samples = 32ms

        async def send_audio():
            print(f"Streaming {len(audio_bytes)} bytes of {wav_file}...")
            # 1. Stream the audio file chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = chunk + b"\x00" * (chunk_size - len(chunk))
                await ws.send(chunk)
                await asyncio.sleep(0.032) # real-time 32ms pace

            print("Utterance finished. Now sending 1.0s of trailing silence for server VAD...")
            # 2. Stream 1 second of silence (31 chunks of 1024 bytes)
            silence_chunk = b"\x00" * chunk_size
            for _ in range(32):
                await ws.send(silence_chunk)
                await asyncio.sleep(0.032)

            print("Done streaming.")

        async def receive_all():
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    print(f"  -> RECEIVED [{time.strftime('%X')}]: {msg}")
            except asyncio.TimeoutError:
                print("Receive completed (timeout).")

        await asyncio.gather(send_audio(), receive_all())

if __name__ == "__main__":
    wav_file = "sample_01_hinglish_ke_postposition.wav"
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    asyncio.run(test_single_sample(wav_file))
