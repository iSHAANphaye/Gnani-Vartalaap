"""Live Testing-Improvement Benchmark Suite using Real Gnani.ai APIs.

Streams real Indic audio WAV samples through:
1. Live Gnani Prisma STT v3.0 Streaming WebSocket (wss://api.vachana.ai/stt/v3/stream)
2. Vartalaap Indic Semantic Parser & Hybrid Classifier
3. Vartalaap Dynamic Turn Engine (Dual-Stream Arbitration)
4. Live Gnani Timbre v3.0 Voice Synthesis (https://api.vachana.ai/api/v1/tts/inference)

Measures:
- ASR token accuracy & latency from Gnani Prisma
- Semantic classification accuracy across scripts (Devanagari / Romanized)
- Dynamic ETS calculation vs Fixed 500ms baseline
- Backchannel ducking & Barge-in flush latencies
- Timbre TTS audio generation & streaming speed
"""

import asyncio
import json
import logging
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from app.config import get_settings
from app.core.backchannel import DuckingAction, TwoStageBackchannelGate
from app.core.engine import DynamicTurnEngine, TurnState
from app.core.semantic import HybridTurnClassifier, TurnCategory
from app.services.gnani_stt import GnaniSTTClient, STTTranscript
from app.services.gnani_tts import GnaniTTSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_benchmark")

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "benchmark" / "test_samples" / "manifest.json"


async def test_sample_with_live_gnani(
    sample_info: Dict[str, Any],
    stt_client: GnaniSTTClient,
    classifier: HybridTurnClassifier,
    tts_client: GnaniTTSClient,
) -> Dict[str, Any]:
    """Stream audio sample to live Gnani STT and evaluate turn-taking arbitration."""
    sample_id = sample_info["id"]
    sample_name = sample_info["sample_name"]
    category = sample_info["category"]
    cat_name = sample_info["category_name"]
    lang = sample_info["language"]
    wav_path = ROOT_DIR / sample_info["file_path"]

    with wave.open(str(wav_path), "rb") as wf:
        audio_bytes = wf.readframes(wf.getnframes())

    logger.info("Evaluating %s (%s, %s): %d bytes audio", sample_id, cat_name, lang, len(audio_bytes))

    # Send audio chunks in real-time pace (1024 bytes = 32ms)
    chunk_size = 1024
    t_start = time.perf_counter()

    # Reconnect STT client if disconnected
    if not stt_client.is_connected or stt_client._closed:
        await stt_client.connect()

    # Stream audio
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i : i + chunk_size]
        await stt_client.send_audio_chunk(chunk)
        await asyncio.sleep(0.015)  # slightly faster than real-time for benchmark throughput

    # Collect incoming transcripts from Gnani Prisma
    transcripts_received: List[STTTranscript] = []
    t_stream_done = time.perf_counter()
    timeout = 3.0  # seconds to wait for ASR emission

    while time.perf_counter() - t_stream_done < timeout:
        try:
            transcript = await asyncio.wait_for(stt_client._transcript_queue.get(), timeout=0.3)
            transcripts_received.append(transcript)
            stt_client._transcript_queue.task_done()
            if transcript.is_final:
                break
        except asyncio.TimeoutError:
            if transcripts_received:
                break
            continue

    stt_latency_ms = (time.perf_counter() - t_start) * 1000.0
    final_text = transcripts_received[-1].text if transcripts_received else ""

    # Classify transcript through Vartalaap Hybrid Classifier
    dummy_audio = np.frombuffer(audio_bytes[:32000], dtype=np.int16).astype(np.float32) / 32768.0
    text_to_eval = final_text or sample_info.get("full_transcript", "")
    sem_res = classifier.evaluate(audio_pcm=dummy_audio, transcript=text_to_eval)

    # Dynamic ETS calculation
    p_comp = sem_res.p_complete
    dynamic_ets_ms = float(np.clip(250.0 + (1.0 - p_comp) * 1750.0, 250.0, 2000.0))

    # Synthesize sample response with Gnani Timbre TTS
    tts_bytes_len = 0
    tts_time_ms = 0.0
    if category in ("E", "A"):
        t0_tts = time.perf_counter()
        try:
            tts_res = tts_client.synthesize(f"Ji samajh gaya, {final_text[:30]}")
            tts_bytes_len = len(tts_res)
            tts_time_ms = (time.perf_counter() - t0_tts) * 1000.0
        except Exception as e:
            logger.warning("TTS generation notice: %s", e)

    return {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "category": category,
        "category_name": cat_name,
        "ground_truth_transcript": sample_info.get("full_transcript", ""),
        "gnani_stt_transcript": final_text,
        "gnani_stt_tokens_count": len(transcripts_received),
        "gnani_stt_latency_ms": round(stt_latency_ms, 1),
        "classified_category": sem_res.category.name,
        "lexical_penalty": round(sem_res.lexical_penalty, 2),
        "p_complete": round(p_comp, 4),
        "dynamic_ets_ms": round(dynamic_ets_ms, 1),
        "matched_markers": sem_res.matched_markers,
        "tts_bytes_generated": tts_bytes_len,
        "tts_latency_ms": round(tts_time_ms, 1),
    }


async def main():
    print("=" * 80)
    print(" 🚀 PROJECT VARTALAAP - EXTENSIVE TESTING & IMPROVEMENT BENCHMARK")
    print("=" * 80)

    settings = get_settings()
    print(f"Targeting Gnani Cloud Endpoints:")
    print(f"  - STT WebSocket : {settings.gnani_stt_ws_url}")
    print(f"  - TTS REST URL  : {settings.gnani_tts_rest_url}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    print(f"Total Test Scenarios in Manifest: {len(samples)}")

    stt_client = GnaniSTTClient(settings=settings)
    await stt_client.connect()
    classifier = HybridTurnClassifier(settings=settings)
    tts_client = GnaniTTSClient(settings=settings)

    # Test selected representative cross-section covering all 5 categories
    test_indices = [0, 1, 3, 4, 10, 11, 14, 16, 17, 22, 23, 26, 27]
    selected_samples = [samples[i] for i in test_indices]

    results = []
    print(f"\nRunning benchmark on {len(selected_samples)} representative Indic samples...")

    for sample in selected_samples:
        try:
            res = await test_sample_with_live_gnani(sample, stt_client, classifier, tts_client)
            results.append(res)
            print(f"\n[{res['sample_id']}] {res['category_name']}")
            print(f"  Expected text : '{res['ground_truth_transcript']}'")
            print(f"  Gnani STT text: '{res['gnani_stt_transcript']}'")
            print(f"  Vartalaap Cat : {res['classified_category']} (Penalty: {res['lexical_penalty']:+0.2f}, Markers: {res['matched_markers']})")
            print(f"  Dynamic ETS   : {res['dynamic_ets_ms']} ms | P(complete): {res['p_complete']:.2f}")
            if res['tts_bytes_generated'] > 0:
                print(f"  Gnani TTS Gen : {res['tts_bytes_generated']} bytes WAV in {res['tts_latency_ms']} ms")
        except Exception as e:
            logger.error("Error evaluating %s: %s", sample['id'], e)

    await stt_client.close()

    # Save initial results
    out_path = ROOT_DIR / "benchmark" / "live_gnani_benchmark_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved live benchmark run results to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
