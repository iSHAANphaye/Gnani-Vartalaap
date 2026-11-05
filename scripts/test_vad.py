"""Voice Activity Detection and Rolling Audio Ring Buffer Verification Suite.

Comprehensive functional and latency validation covering:
1. Ring buffer capacity bounds and circular overwrite behavior up to 20 seconds.
2. Left-padding verification matching Smart-Turn tensor input specifications.
3. VAD processing on silence and elapsed duration accumulation.
4. VAD transition on synthetic speech and duration counter reset.
5. High-precision CPU latency benchmark over 100 consecutive 512-sample chunks (<2.0ms budget).
6. Integrated AcousticDetector coordinator verification.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.vad import AcousticDetector, AudioRingBuffer, SileroVAD, VADResult


def generate_formant_speech_chunk(sr: int = 16000, n_samples: int = 512) -> bytes:
    """Synthesize a single 512-sample human vowel formant frame.

    Uses an impulse train modulated through second-order resonators tuned to the
    first three human formant frequencies (/a/ vowel: 730Hz, 1090Hz, 2440Hz).
    """
    pulses = np.zeros(n_samples, dtype=np.float32)
    period = int(sr / 120)  # ~120Hz fundamental pitch
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


def test_ring_buffer_capacity() -> bool:
    """Verify circular ring buffer capacity is capped at 8.0s (128,000 samples / 256,000 bytes)."""
    print("\n" + "=" * 65)
    print(" [TEST 1] AudioRingBuffer Capacity & Circular Bound Verification")
    print("=" * 65)

    buffer = AudioRingBuffer(capacity_sec=8.0, sample_rate=16000)
    assert buffer.capacity_samples == 128000, f"Expected 128000 capacity, got {buffer.capacity_samples}"
    assert buffer.capacity_bytes == 256000, f"Expected 256000 bytes capacity, got {buffer.capacity_bytes}"
    assert buffer.size == 0
    assert buffer.duration_sec == 0.0
    assert not buffer.is_full

    # Ingest 20.0 seconds of audio chunks (625 chunks of 512 samples)
    chunk_samples = 512
    total_sec = 20.0
    total_chunks = int(total_sec * 16000 / chunk_samples)  # 625 chunks

    print(f"  - Ingesting {total_sec:.1f}s of audio ({total_chunks} chunks of {chunk_samples} samples)...")
    for i in range(total_chunks):
        # Create chunk with sequential integer values to verify FIFO ordering
        chunk_data = np.full(chunk_samples, fill_value=i, dtype=np.int16).tobytes()
        written = buffer.append(chunk_data)
        assert written == chunk_samples
        assert buffer.size <= 128000, f"Buffer size {buffer.size} exceeded capacity 128000!"

    print(f"  ✓ Final buffer size        : {buffer.size} samples (capacity: {buffer.capacity_samples})")
    print(f"  ✓ Final buffer duration    : {buffer.duration_sec:.2f} seconds")
    print(f"  ✓ Final raw bytes length   : {len(buffer.get_raw_bytes())} bytes")
    print(f"  ✓ Buffer full status       : {buffer.is_full}")

    assert buffer.size == 128000, f"Expected size 128000, got {buffer.size}"
    assert len(buffer) == 128000, f"Expected len 128000, got {len(buffer)}"
    assert buffer.duration_sec == 8.0, f"Expected 8.0s, got {buffer.duration_sec}"
    assert len(buffer.get_raw_bytes()) == 256000, f"Expected 256000 bytes, got {len(buffer.get_raw_bytes())}"
    assert buffer.is_full is True

    # Verify FIFO retention: oldest preserved chunk index should be total_chunks - 250 = 375
    # (8.0s at 16kHz = 128,000 samples = 250 chunks of 512 samples)
    raw = buffer.get_raw_bytes()
    preserved_samples = np.frombuffer(raw, dtype=np.int16)
    expected_oldest_val = total_chunks - (128000 // chunk_samples)  # 625 - 250 = 375
    assert preserved_samples[0] == expected_oldest_val, (
        f"FIFO order violated: expected oldest chunk value {expected_oldest_val}, got {preserved_samples[0]}"
    )
    assert preserved_samples[-1] == total_chunks - 1, (
        f"FIFO order violated: expected newest chunk value {total_chunks - 1}, got {preserved_samples[-1]}"
    )
    print(f"  ✓ FIFO Retention Verified  : Oldest chunk={preserved_samples[0]}, Newest chunk={preserved_samples[-1]}")

    return True


def test_padding_verification() -> bool:
    """Verify get_recent_audio(8.0) returns exact shape (128000,) with proper zero-padding."""
    print("\n" + "=" * 65)
    print(" [TEST 2] Left-Padding Verification for Smart-Turn Acoustic Engine")
    print("=" * 65)

    buffer = AudioRingBuffer(capacity_sec=8.0, sample_rate=16000)

    # Ingest exactly 2.0s of audio (32,000 samples)
    target_samples = 32000
    test_value = 16384  # 0.5 amplitude when normalized
    audio_2s = np.full(target_samples, fill_value=test_value, dtype=np.int16).tobytes()
    buffer.append(audio_2s)

    assert buffer.size == target_samples
    assert buffer.duration_sec == 2.0
    print(f"  ✓ Buffer populated with {buffer.duration_sec:.1f}s of audio ({buffer.size} samples)")

    # Request 8.0 seconds from the 2.0s buffer
    audio_8s = buffer.get_recent_audio(duration_sec=8.0)
    print(f"  ✓ Returned audio tensor shape : {audio_8s.shape}")
    print(f"  ✓ Returned audio tensor dtype : {audio_8s.dtype}")

    # Validations
    assert audio_8s.shape == (128000,), f"Expected shape (128000,), got {audio_8s.shape}"
    assert audio_8s.dtype == np.float32, f"Expected float32, got {audio_8s.dtype}"

    # Verify zero-padding: 8.0s - 2.0s = 6.0s of left-padding = 96,000 zero samples
    expected_pad_len = 128000 - 32000  # 96,000
    zero_prefix = audio_8s[:expected_pad_len]
    audio_suffix = audio_8s[expected_pad_len:]

    assert np.all(zero_prefix == 0.0), "Left-padding corrupted: non-zero values found in prefix"
    expected_normalized_val = test_value / 32768.0  # 0.5
    assert np.allclose(audio_suffix, expected_normalized_val, atol=1e-4), (
        f"Audio suffix mismatch: expected {expected_normalized_val}, got {audio_suffix[0]}"
    )
    print(f"  ✓ Left-padding verified       : Exactly {expected_pad_len} zero samples prepended")
    print(f"  ✓ Audio waveform normalized   : {len(audio_suffix)} samples normalized to {audio_suffix[0]:.4f}")

    # Verify sub-duration requests (e.g. 1.0s) returns exactly recent samples without padding
    audio_1s = buffer.get_recent_audio(duration_sec=1.0)
    assert audio_1s.shape == (16000,), f"Expected shape (16000,), got {audio_1s.shape}"
    assert np.allclose(audio_1s, expected_normalized_val, atol=1e-4)
    print(f"  ✓ Sub-duration request (1.0s) : Exact shape (16000,) with no unnecessary padding")

    # Verify reset method
    buffer.reset()
    assert buffer.size == 0
    empty_8s = buffer.get_recent_audio(8.0)
    assert empty_8s.shape == (128000,)
    assert np.all(empty_8s == 0.0)
    print("  ✓ Buffer reset verified       : Successfully wiped all stored frames")

    return True


def test_vad_silence_processing() -> bool:
    """Verify Silero VAD correctly flags silence and tracks elapsed silence duration."""
    print("\n" + "=" * 65)
    print(" [TEST 3] Silero VAD Silence Processing & Elapsed Timer Verification")
    print("=" * 65)

    vad = SileroVAD(threshold=0.5, sample_rate=16000)
    chunk_samples = 512
    silent_chunk = bytes(chunk_samples * 2)  # 1024 zero bytes

    print("  - Streaming 10 consecutive silent chunks (32ms each = 320ms total)...")
    for step in range(1, 11):
        result = vad.process_chunk(silent_chunk)
        expected_silence_ms = step * 32.0

        assert not result.is_speech, f"Step {step}: False positive speech detected on silence!"
        assert result.confidence < 0.5, f"Step {step}: Confidence {result.confidence} >= threshold 0.5"
        assert abs(result.silence_duration_ms - expected_silence_ms) < 1e-3, (
            f"Step {step}: Expected silence {expected_silence_ms}ms, got {result.silence_duration_ms}ms"
        )
        assert result.speech_duration_ms == 0.0, f"Step {step}: Speech duration should be 0.0ms"

    print(f"  ✓ Silence Detection Passed    : 10/10 chunks classified as is_speech=False")
    print(f"  ✓ Final silence confidence    : {result.confidence:.4f} (threshold: {vad.threshold})")
    print(f"  ✓ Final silence duration      : {result.silence_duration_ms:.1f} ms (expected: 320.0 ms)")
    print(f"  ✓ Speech duration             : {result.speech_duration_ms:.1f} ms")

    return True


def test_vad_speech_transition() -> bool:
    """Verify Silero VAD detects synthetic speech and immediately resets silence duration."""
    print("\n" + "=" * 65)
    print(" [TEST 4] Silero VAD Speech Detection & Silence Timer Reset Verification")
    print("=" * 65)

    vad = SileroVAD(threshold=0.5, sample_rate=16000)
    silent_chunk = bytes(512 * 2)
    speech_chunk = generate_formant_speech_chunk(sr=16000, n_samples=512)

    # Accumulate 160ms of silence
    for _ in range(5):
        vad.process_chunk(silent_chunk)

    silence_before = vad.silence_duration_ms
    assert abs(silence_before - 160.0) < 1e-3
    print(f"  ✓ Established initial silence : {silence_before:.1f} ms")

    # Ingest speech chunk
    res_speech = vad.process_chunk(speech_chunk)
    print(f"  ✓ Speech Chunk Evaluation    : confidence={res_speech.confidence:.4f}, is_speech={res_speech.is_speech}")

    assert res_speech.is_speech, f"Failed to detect speech! Confidence was {res_speech.confidence:.4f}"
    assert res_speech.confidence >= 0.5, f"Confidence {res_speech.confidence:.4f} below threshold"
    assert res_speech.silence_duration_ms == 0.0, (
        f"Silence duration not reset upon speech! Found {res_speech.silence_duration_ms} ms"
    )
    assert res_speech.speech_duration_ms == 32.0, (
        f"Expected speech duration 32.0ms, got {res_speech.speech_duration_ms} ms"
    )
    print(f"  ✓ Silence Timer Reset         : silence_duration_ms successfully reset to 0.0 ms")
    print(f"  ✓ Speech Timer Increment      : speech_duration_ms = {res_speech.speech_duration_ms:.1f} ms")

    # Ingest second speech chunk
    res_speech2 = vad.process_chunk(speech_chunk)
    assert res_speech2.is_speech
    assert res_speech2.silence_duration_ms == 0.0
    assert res_speech2.speech_duration_ms == 64.0
    print(f"  ✓ Consecutive Speech Tracking : speech_duration_ms = {res_speech2.speech_duration_ms:.1f} ms")

    # Transition back to silence
    res_silence = vad.process_chunk(silent_chunk)
    assert not res_silence.is_speech
    assert res_silence.speech_duration_ms == 0.0
    assert res_silence.silence_duration_ms == 32.0
    print(f"  ✓ Return to Silence Tracking  : speech_duration_ms reset to 0.0 ms, silence_ms = 32.0 ms")

    return True


def test_vad_latency_benchmark() -> bool:
    """Verify Silero VAD executes 512-sample chunk evaluation in <2.0ms on CPU."""
    print("\n" + "=" * 65)
    print(" [TEST 5] Silero VAD CPU Latency Benchmark (100 Consecutive Runs)")
    print("=" * 65)

    vad = SileroVAD(threshold=0.5, sample_rate=16000)
    chunk = bytes(512 * 2)

    # Warmup
    for _ in range(20):
        vad.process_chunk(chunk)
    vad.reset()

    # Benchmark 100 consecutive chunks
    iterations = 100
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        vad.process_chunk(chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    avg_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))

    print(f"  ✓ Benchmark Statistics ({iterations} runs):")
    print(f"      - Average Latency : {avg_ms:.3f} ms")
    print(f"      - Median Latency  : {median_ms:.3f} ms")
    print(f"      - Min Latency     : {min_ms:.3f} ms")
    print(f"      - Max Latency     : {max_ms:.3f} ms")
    print(f"      - P95 Latency     : {p95_ms:.3f} ms")
    print(f"      - P99 Latency     : {p99_ms:.3f} ms")

    assert avg_ms < 2.0, f"Average latency {avg_ms:.3f}ms exceeds 2.0ms budget requirement!"
    assert p95_ms < 2.0, f"P95 latency {p95_ms:.3f}ms exceeds 2.0ms budget requirement!"

    print(f"  ✓ SUCCESS: Silero VAD execution strictly satisfied <2.0ms CPU budget!")
    return True


def test_acoustic_detector_coordinator() -> bool:
    """Verify integrated AcousticDetector coordinating buffer and VAD."""
    print("\n" + "=" * 65)
    print(" [TEST 6] AcousticDetector Coordinator Integration Verification")
    print("=" * 65)

    detector = AcousticDetector(threshold=0.5, ring_buffer_duration_sec=8.0, sample_rate=16000)
    speech_chunk = generate_formant_speech_chunk(sr=16000, n_samples=512)

    # Process speech chunk through coordinator
    result = detector.process_chunk(speech_chunk)
    assert isinstance(result, VADResult)
    assert result.is_speech
    assert detector.speech_duration_ms == 32.0
    assert detector.silence_duration_ms == 0.0

    # Verify audio buffer holds the chunk
    recent = detector.get_recent_audio(duration_sec=8.0)
    assert recent.shape == (128000,)
    assert len(detector.get_raw_bytes()) == 1024

    # Verify coordinator reset
    detector.reset()
    assert detector.speech_duration_ms == 0.0
    assert detector.silence_duration_ms == 0.0
    assert len(detector.get_raw_bytes()) == 0

    print("  ✓ AcousticDetector properly ingests audio, evaluates VAD, and exposes clean state")
    return True


def main() -> int:
    print("=" * 65)
    print(" 🚀  PROJECT VARTALAAP - VAD & AUDIO RING BUFFER VERIFICATION")
    print("=" * 65)

    try:
        t1 = test_ring_buffer_capacity()
        t2 = test_padding_verification()
        t3 = test_vad_silence_processing()
        t4 = test_vad_speech_transition()
        t5 = test_vad_latency_benchmark()
        t6 = test_acoustic_detector_coordinator()

        if all([t1, t2, t3, t4, t5, t6]):
            print("\n" + "=" * 65)
            print(" 🎉  ALL VAD & AUDIO BUFFER TESTS PASSED SUCCESSFULLY!")
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


if __name__ == "__main__":
    sys.exit(main())
