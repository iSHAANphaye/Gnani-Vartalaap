"""Synthesize authentic Indic speech with Gnani Timbre TTS and stream to Gnani Prisma STT."""

import asyncio
import io
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
from app.services.gnani_tts import GnaniTTSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def test_roundtrip():
    settings = get_settings()
    tts = GnaniTTSClient(settings=settings)

    part1_text = "मेरा अकाउंट बैलेंस के"
    part2_text = "बारे में बताओ"

    print("--- Step 1: Synthesizing Part 1 with Gnani Timbre TTS ---")
    wav1_bytes = tts.synthesize(part1_text)
    print(f"✓ Part 1 WAV: {len(wav1_bytes)} bytes")

    print("--- Step 2: Synthesizing Part 2 with Gnani Timbre TTS ---")
    wav2_bytes = tts.synthesize(part2_text)
    print(f"✓ Part 2 WAV: {len(wav2_bytes)} bytes")

    # Extract raw PCM from both WAVs
    with wave.open(io.BytesIO(wav1_bytes), "rb") as wf1:
        pcm1 = wf1.readframes(wf1.getnframes())
    with wave.open(io.BytesIO(wav2_bytes), "rb") as wf2:
        pcm2 = wf2.readframes(wf2.getnframes())

    # Create 750ms pause (0.75 * 16000 * 2 bytes = 24000 bytes of silence)
    pause_bytes = b"\x00" * int(0.75 * 16000 * 2)

    # 1.0s trailing silence for end of utterance
    trailing_bytes = b"\x00" * int(1.0 * 16000 * 2)

    combined_pcm = pcm1 + pause_bytes + pcm2 + trailing_bytes

    # Save to a test wav
    test_wav_path = ROOT_DIR / "benchmark" / "test_samples" / "real_gnani_sample_01.wav"
    with wave.open(str(test_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(combined_pcm)
    print(f"✓ Saved authentic spoken WAV: {test_wav_path} ({len(combined_pcm)} bytes PCM, {len(combined_pcm)/32000:.2f}s)")

    print("\n--- Step 3: Streaming Authentic Speech to Gnani Prisma STT ---")
    api_key = settings.gnani_api_key.get_secret_value()
    headers = {
        "token": api_key,
        "X-API-Key-ID": api_key,
        "language_code": "hi-IN",
        "sample_rate": "16000",
        "audio_format": "pcm",
    }
    try:
        connect_kwargs = {"additional_headers": headers}
        ws_conn = websockets.connect(settings.gnani_stt_ws_url, **connect_kwargs)
    except TypeError:
        ws_conn = websockets.connect(settings.gnani_stt_ws_url, extra_headers=headers)

    async with ws_conn as ws:
        handshake = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print("✓ STT Connected. Handshake:", handshake)

        chunk_size = 1024
        is_streaming_done = False

        async def stream_audio():
            nonlocal is_streaming_done
            t0 = time.time()
            for i in range(0, len(combined_pcm), chunk_size):
                chunk = combined_pcm[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b"\x00" * (chunk_size - len(chunk))
                await ws.send(chunk)
                await asyncio.sleep(0.032)
            is_streaming_done = True
            print(f"Finished streaming all {len(combined_pcm)} bytes in {time.time() - t0:.2f}s")

        async def listen_responses():
            while not is_streaming_done:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    mtype = data.get("type")
                    if mtype == "speech_start":
                        print(f"  [STT Event] SPEECH START at t_ms={data.get('t_ms')}")
                    elif mtype == "transcript":
                        txt = data.get("text") or data.get("output", {}).get("literal", "")
                        print(f"  [STT Transcript] is_final={data.get('is_final')} | Text: '{txt}' | Latency: {data.get('latency')}ms")
                    else:
                        print(f"  [STT Message] {msg}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"  [STT Exception during stream]: {type(e).__name__}: {e}")
                    break

            # Stream finished; wait up to 5s for trailing final transcripts
            print("Listening for trailing transcripts after stream completion...")
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=4.0)
                    data = json.loads(msg)
                    mtype = data.get("type")
                    if mtype == "transcript":
                        txt = data.get("text") or data.get("output", {}).get("literal", "")
                        print(f"  [STT Final Transcript] is_final={data.get('is_final')} | Text: '{txt}' | Latency: {data.get('latency')}ms")
                    else:
                        print(f"  [STT Trailing Message] {msg}")
            except asyncio.TimeoutError:
                print("All transcripts finished.")
            except Exception as e:
                print(f"  [STT Trailing Exception]: {type(e).__name__}: {e}")

        await asyncio.gather(stream_audio(), listen_responses())

if __name__ == "__main__":
    asyncio.run(test_roundtrip())
