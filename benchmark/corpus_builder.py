"""Offline Test Corpus Builder for Project Vartalaap (Stage 1).

Batch-synthesizes authentic Indic user utterances via Gnani Timbre TTS (timbre-v3.0)
across diverse speaker profiles (Shivani, Riyaa, Avinash) and generates procedural
acoustic noise samples. Implements permanent disk caching in benchmark/test_corpus/.
"""

from __future__ import annotations

import io
import json
import logging
import math
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.services.gnani_tts import GnaniTTSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("vartalaap.corpus_builder")

CORPUS_DIR = ROOT_DIR / "benchmark" / "test_corpus"
MANIFEST_YAML = ROOT_DIR / "benchmark" / "corpus_manifest.yaml"
INDEX_JSON = CORPUS_DIR / "corpus_manifest.json"


def generate_procedural_noise(noise_type: str, duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate deterministic procedural PCM noise."""
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / float(sample_rate)

    if noise_type == "silence":
        samples = np.zeros(n_samples, dtype=np.float32)

    elif noise_type == "breathing":
        # Simulate rhythmic chest breath intakes (respiratory cycle ~0.3 Hz)
        envelope = np.maximum(0.0, np.sin(2 * np.pi * 0.35 * t)) ** 3
        rng = np.random.default_rng(42)
        # Low-pass turbulent air breath
        noise = rng.standard_normal(n_samples).astype(np.float32)
        filtered = np.convolve(noise, np.ones(32) / 32.0, mode="same")
        samples = filtered * envelope * 0.012

    elif noise_type == "fan":
        # Realistic room fan hum: 50Hz fundamental + 100Hz/150Hz harmonics + white noise
        fundamental = 0.40 * np.sin(2 * np.pi * 50.0 * t)
        h2 = 0.25 * np.sin(2 * np.pi * 100.0 * t)
        h3 = 0.15 * np.sin(2 * np.pi * 150.0 * t)
        rng = np.random.default_rng(12345)
        white = rng.standard_normal(n_samples) * 0.15
        combined = (fundamental + h2 + h3 + white).astype(np.float32)
        rms = np.sqrt(np.mean(combined ** 2))
        target_rms = 0.015
        samples = (combined / (rms + 1e-9)) * target_rms

    else:
        samples = np.zeros(n_samples, dtype=np.float32)

    clipped = np.clip(samples, -1.0, 1.0)
    int16_pcm = (clipped * 32767.0).astype(np.int16).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_pcm)
    return buf.getvalue()


def scale_wav_amplitude(wav_bytes: bytes, scale: float) -> bytes:
    """Scale audio samples in WAV bytes by a float factor."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * scale
    clipped = np.clip(samples, -32768.0, 32767.0).astype(np.int16).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(clipped)
    return buf.getvalue()


def get_wav_duration_sec(wav_bytes: bytes) -> float:
    """Calculate duration of WAV bytes in seconds."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 3) if rate > 0 else 0.0
    except Exception:
        return 0.0


def build_corpus(force_rebuild: bool = False) -> Dict[str, Any]:
    """Build or verify complete multi-voice test corpus."""
    if not MANIFEST_YAML.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_YAML}")

    with open(MANIFEST_YAML, "r", encoding="utf-8") as f:
        manifest_data = yaml.safe_load(f)

    samples = manifest_data.get("samples", [])
    logger.info("Loaded manifest with %d sample specifications", len(samples))

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    tts_client = GnaniTTSClient(settings=settings)

    cached_count = 0
    synthesized_count = 0
    index_entries: List[Dict[str, Any]] = []

    t0 = time.time()

    for idx, item in enumerate(samples, 1):
        sample_id = item["id"]
        sample_type = item.get("type", "speech")
        voice = item.get("voice", "Shivani")
        text = item.get("text", "")
        out_path = CORPUS_DIR / f"{sample_id}.wav"

        # Check existing valid cache
        if not force_rebuild and out_path.exists() and out_path.stat().st_size > 1000:
            cached_count += 1
            wav_data = out_path.read_bytes()
            duration_sec = get_wav_duration_sec(wav_data)
            logger.debug("[%d/%d] Cache hit: %s (%.2fs)", idx, len(samples), sample_id, duration_sec)
        else:
            if sample_type == "synthetic_noise":
                noise_type = item.get("noise_type", "silence")
                duration = float(item.get("duration_sec", 3.0))
                wav_data = generate_procedural_noise(noise_type=noise_type, duration_sec=duration)
                out_path.write_bytes(wav_data)
                duration_sec = duration
                synthesized_count += 1
                logger.info("[%d/%d] Generated synthetic noise: %s (%s, %.1fs)", idx, len(samples), sample_id, noise_type, duration)
            else:
                logger.info(
                    "[%d/%d] Synthesizing [%s] via Gnani Timbre (%s): '%s'",
                    idx,
                    len(samples),
                    sample_id,
                    voice,
                    text[:40] + ("..." if len(text) > 40 else ""),
                )
                try:
                    wav_data = tts_client.synthesize(text=text, voice=voice, language="hi-IN")
                    amp_scale = item.get("amplitude_scale")
                    if amp_scale is not None and amp_scale > 0:
                        wav_data = scale_wav_amplitude(wav_data, float(amp_scale))
                    out_path.write_bytes(wav_data)
                    duration_sec = get_wav_duration_sec(wav_data)
                    synthesized_count += 1
                    # Polite rate-limit delay
                    time.sleep(0.08)
                except Exception as exc:
                    logger.error("Failed to synthesize sample %s: %s", sample_id, exc)
                    raise

        entry = dict(item)
        entry["file_path"] = str(out_path.relative_to(ROOT_DIR)).replace("\\", "/")
        entry["duration_sec"] = duration_sec
        entry["file_size_bytes"] = out_path.stat().st_size
        index_entries.append(entry)

    summary = {
        "version": manifest_data.get("version", "2.2.0"),
        "total_samples": len(samples),
        "cached_samples": cached_count,
        "synthesized_samples": synthesized_count,
        "build_duration_sec": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "samples": index_entries,
    }

    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Corpus build complete! Total: %d, Cached: %d, Synthesized: %d in %.2fs",
        len(samples),
        cached_count,
        synthesized_count,
        summary["build_duration_sec"],
    )
    logger.info("Corpus index saved to %s", INDEX_JSON)
    return summary


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_corpus(force_rebuild=force)
