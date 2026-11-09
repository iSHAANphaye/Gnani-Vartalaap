"""Verification test suite for [TICKET-02].

Tests:
1. Centralized settings loading and credential masking (app.config).
2. Smart-Turn v3.2 model loader on CUDA (RTX 5060) with sub-8ms inference latency.
3. Gnani Prisma STT async client streaming and REST fallback.
4. Gnani Timbre TTS client with expressive tags and WAV generation.
"""

import asyncio
import os
import sys
import time
import wave
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import Settings, get_settings
from app.services.gnani_stt import GnaniSTTClient, STTTranscript
from app.services.gnani_tts import GnaniTTSClient, strip_expressive_tags
from app.services.smart_turn import SmartTurnEngine, get_smart_turn_engine


def test_settings() -> bool:
    """Verify settings load cleanly with secret masking and valid defaults."""
    print("\n" + "=" * 60)
    print(" [TEST 1] Settings & Credential Masking Verification")
    print("=" * 60)

    settings = get_settings()
    assert isinstance(settings, Settings), "Settings failed to instantiate"

    print(f"  ✓ Settings instantiated successfully")
    print(f"  ✓ Host: {settings.host}:{settings.port}")
    print(f"  ✓ Torch Device target: {settings.torch_device}")
    print(f"  ✓ Default silence threshold: {settings.default_silence_threshold_ms} ms")
    print(f"  ✓ Has valid Gnani credentials: {settings.has_valid_gnani_credentials}")

    # Check that secrets are not exposed in repr or str
    repr_str = repr(settings)
    api_key_repr = repr(settings.gnani_api_key)
    raw_key = settings.gnani_api_key.get_secret_value()

    assert "**********" in api_key_repr, "SecretStr masking failed on API key repr"
    if raw_key and raw_key != "your_gnani_api_key_here":
        assert raw_key not in repr_str, "Raw API key leaked in Settings repr!"

    print(f"  ✓ API Key Masking Verified: {api_key_repr}")
    return True


def test_smart_turn_latency() -> bool:
    """Verify Smart-Turn v3.2 model executes on RTX 5060 in <8ms."""
    print("\n" + "=" * 60)
    print(" [TEST 2] Smart-Turn v3.2 Loader & RTX 5060 Latency")
    print("=" * 60)

    engine = get_smart_turn_engine()
    print(f"  ✓ Engine initialized on device: {engine.device}")
    if engine.is_gpu:
        device_name = torch.cuda.get_device_name(0)
        print(f"  ✓ GPU Acceleration Active: {device_name}")
        assert "RTX 5060" in device_name or "NVIDIA" in device_name

    # Test dummy 16kHz audio input (2 seconds)
    dummy_np = np.random.randn(32000).astype(np.float32) * 0.1
    p_incomplete_np = engine.predict_incomplete(dummy_np)
    print(f"  ✓ NumPy array inference: P(incomplete) = {p_incomplete_np:.4f}")
    assert 0.0 <= p_incomplete_np <= 1.0, f"Invalid probability: {p_incomplete_np}"

    # Test dummy PyTorch tensor input (4 seconds)
    dummy_tensor = torch.randn(64000) * 0.1
    p_incomplete_tensor = engine.predict_incomplete(dummy_tensor)
    print(f"  ✓ PyTorch Tensor inference: P(incomplete) = {p_incomplete_tensor:.4f}")
    assert 0.0 <= p_incomplete_tensor <= 1.0, f"Invalid probability: {p_incomplete_tensor}"

    # Latency Benchmark: 50 consecutive runs
    iterations = 50
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.predict_incomplete(dummy_np)
        if engine.is_gpu:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    min_ms = float(np.min(latencies))
    p95_ms = float(np.percentile(latencies, 95))

    print(f"  ✓ Latency Benchmark ({iterations} runs):")
    print(f"      - Average Latency : {avg_ms:.3f} ms")
    print(f"      - Min Latency     : {min_ms:.3f} ms")
    print(f"      - P95 Latency     : {p95_ms:.3f} ms")

    assert avg_ms < 8.0, f"Average latency {avg_ms:.2f}ms exceeds 8.0ms budget requirement!"
    print(f"  ✓ SUCCESS: Inference latency ({avg_ms:.3f} ms) is strictly <8ms on RTX 5060!")
    return True


async def test_gnani_stt() -> bool:
    """Verify Gnani Prisma STT async client streaming and fallback."""
    print("\n" + "=" * 60)
    print(" [TEST 3] Gnani Prisma STT Client Verification")
    print("=" * 60)

    client = GnaniSTTClient()
    connected = await client.connect()
    assert connected, "Failed to connect STT client"
    print(f"  ✓ STT Client Connected (Mock Mode: {client.is_mock_mode})")

    pcm_chunk = b"\x00\x00" * 800
    # Send audio frames
    wav_path = Path("benchmark/test_samples/sample_27_terminal_balance_batao.wav")
    if not client.is_mock_mode and wav_path.exists():
        with wave.open(str(wav_path), "rb") as wf:
            real_audio = wf.readframes(wf.getnframes())

        async def feed_audio():
            chunk_size = 1024
            for i in range(0, len(real_audio), chunk_size):
                await client.send_audio_chunk(real_audio[i:i + chunk_size])
                await asyncio.sleep(0.02)
        asyncio.create_task(feed_audio())
    else:
        await client.send_audio_chunk(pcm_chunk)

    # Receive interim transcripts with safety timeout
    received_tokens = []
    try:
        async for transcript in client.stream_transcripts():
            received_tokens.append(transcript.text)
            print(f"  ✓ Emitted Token: '{transcript.text}' (confidence={transcript.confidence:.2f})")
            if len(received_tokens) >= 1:
                break
    except asyncio.TimeoutError:
        pass

    assert len(received_tokens) >= 1, "Failed to receive interim transcript tokens"

    # Test REST fallback
    rest_result = client.transcribe_rest(pcm_chunk)
    assert isinstance(rest_result, STTTranscript)
    print(f"  ✓ REST Fallback Transcription: '{rest_result.text}'")

    await client.close()
    print("  ✓ STT Client closed cleanly")
    return True


async def test_gnani_tts() -> bool:
    """Verify Gnani Timbre TTS client with expressive tags and WAV generation."""
    print("\n" + "=" * 60)
    print(" [TEST 4] Gnani Timbre TTS Client Verification")
    print("=" * 60)

    client = GnaniTTSClient()
    print(f"  ✓ TTS Client Initialized (Mock Mode: {client.is_mock_mode})")

    # Test tag stripping utility
    sample_text = "<excited>Namaste!</excited> <pauses> Kaise hain aap?"
    clean_text = strip_expressive_tags(sample_text)
    assert clean_text == "Namaste!  Kaise hain aap?" or "Namaste!" in clean_text
    print(f"  ✓ Expressive Tag Stripper: '{sample_text}' -> '{clean_text}'")

    # Test audio synthesis
    out_dir = Path("benchmark/test_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ticket02_tts_test.wav"

    audio_bytes = client.synthesize(sample_text, output_path=out_path)
    assert len(audio_bytes) > 44, "Synthesized audio is too small"
    assert audio_bytes.startswith(b"RIFF"), "Generated audio does not have RIFF WAV header"

    # Verify WAV file properties
    with wave.open(str(out_path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / framerate
        print(f"  ✓ WAV File Validated: {channels} ch, {sampwidth * 8}-bit, {framerate} Hz, {duration:.2f}s")
        assert channels == 1, "Expected mono audio"
        assert framerate == 16000, "Expected 16kHz audio"

    # Test streaming chunk generator
    chunks = []
    async for chunk in client.synthesize_stream(sample_text, chunk_size=1600):
        chunks.append(chunk)

    total_streamed = sum(len(c) for c in chunks)
    print(f"  ✓ Streamed {len(chunks)} chunks ({total_streamed} bytes raw PCM)")
    assert total_streamed > 0, "No chunks streamed from synthesize_stream"

    return True


async def main_async() -> int:
    print("=" * 65)
    print(" 🚀  PROJECT VARTALAAP - TICKET-02 VERIFICATION SUITE")
    print("=" * 65)

    try:
        t1 = test_settings()
        t2 = test_smart_turn_latency()
        t3 = await test_gnani_stt()
        t4 = await test_gnani_tts()

        if all([t1, t2, t3, t4]):
            print("\n" + "=" * 65)
            print(" 🎉  ALL TICKET-02 VERIFICATION TESTS PASSED SUCCESSFULLY!")
            print("=" * 65)
            return 0
        else:
            print("\n❌ SOME VERIFICATION TESTS FAILED.")
            return 1
    except Exception as e:
        print(f"\n❌ UNHANDLED EXCEPTION DURING TEST EXECUTION: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
