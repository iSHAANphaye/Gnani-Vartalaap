"""Full-Duplex Automated WebSocket Test Runner for Project Vartalaap (Stage 2).

Executes comprehensive testing suites F through M against the live server endpoint
(ws://localhost:7860/ws/audio) using the multi-voice cached test corpus.
Validates domain accuracy, multi-turn coherence, script robustness, entity extraction,
turn-taking stress boundaries, concurrent session isolation, and STT round-trip fidelity.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.services.gnani_stt import GnaniSTTClient
from app.services.gnani_tts import GnaniTTSClient
from benchmark.report_generator import (
    ResponseValidator,
    TurnEvaluation,
    generate_reports,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("vartalaap.test_runner")

CORPUS_DIR = ROOT_DIR / "benchmark" / "test_corpus"
INDEX_JSON = CORPUS_DIR / "corpus_manifest.json"


def strip_wav_header(raw_bytes: bytes) -> bytes:
    """Safely strips variable-length RIFF/WAV headers (44 to 80+ bytes)."""
    if raw_bytes.startswith(b"RIFF"):
        data_idx = raw_bytes.find(b"data")
        if data_idx != -1:
            return raw_bytes[data_idx + 8 :]
        return raw_bytes[44:]
    return raw_bytes


def generate_noise_floor_pcm(duration_ms: int = 1600, rms: float = 0.002, sample_rate: int = 16000) -> bytes:
    """Generate subtle low-amplitude noise floor frames to simulate active microphone air."""
    n_samples = int((duration_ms / 1000.0) * sample_rate)
    rng = np.random.default_rng(999)
    noise = rng.standard_normal(n_samples).astype(np.float32)
    current_rms = np.sqrt(np.mean(noise ** 2))
    scaled = (noise / (current_rms + 1e-9)) * rms
    clipped = np.clip(scaled, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


async def run_full_duplex_turn(
    ws,
    speech_pcm: bytes,
    backchannel_pcm: Optional[bytes] = None,
    expect_no_turn_window_ms: int = 0,
) -> Dict[str, Any]:
    """Execute a single full-duplex conversational turn over WebSocket.

    Args:
        ws: Active WebSocket client connection.
        speech_pcm: 16kHz mono 16-bit PCM bytes for primary user utterance.
        backchannel_pcm: Optional user backchannel audio to inject mid-turn over bot speech.
        expect_no_turn_window_ms: Non-zero to exit gracefully on hold/silence tests.

    Returns:
        Dictionary containing timing timestamps, accumulated bot audio PCM, and received frames.
    """
    clean_speech = strip_wav_header(speech_pcm)
    t_last_speech_frame = 0.0
    bot_turn_start_time = None
    first_audio_chunk_time = None
    bot_turn_end_time = None
    bot_audio_bytes = bytearray()
    received_frames: List[Dict[str, Any]] = []
    sender_finished = asyncio.Event()

    async def receiver_loop() -> None:
        nonlocal bot_turn_start_time, first_audio_chunk_time, bot_turn_end_time
        try:
            async for message in ws:
                now = time.perf_counter()
                if isinstance(message, str):
                    data = json.loads(message)
                    data["_recv_perf_ts"] = now
                    received_frames.append(data)
                    mtype = data.get("type")

                    if mtype == "BOT_TURN_START" and bot_turn_start_time is None:
                        bot_turn_start_time = now

                    elif mtype == "AUDIO_CHUNK":
                        if first_audio_chunk_time is None:
                            first_audio_chunk_time = now
                        b64 = data.get("audio_base64", "")
                        if b64:
                            raw_chunk = base64.b64decode(b64)
                            bot_audio_bytes.extend(strip_wav_header(raw_chunk))

                    elif mtype == "BOT_TURN_END":
                        bot_turn_end_time = now
                        if backchannel_pcm and not sender_finished.is_set():
                            await sender_finished.wait()
                            await asyncio.sleep(0.35)  # Collect trailing CONTROL frames
                        break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Receiver loop ended with exception: %s", exc)

    async def sender_loop() -> None:
        nonlocal t_last_speech_frame
        # 1. Stream primary speech frames at 32ms cadence (1024 bytes per frame)
        for i in range(0, len(clean_speech), 1024):
            chunk = clean_speech[i : i + 1024]
            if len(chunk) < 1024:
                chunk += b"\x00" * (1024 - len(chunk))
            await ws.send(chunk)
            await asyncio.sleep(0.032)

        t_last_speech_frame = time.perf_counter()

        # 2. Stream trailing silence tail (break early if bot speaks & backchannel is queued)
        tail_ms = expect_no_turn_window_ms if expect_no_turn_window_ms > 0 else 1600
        silence_tail = generate_noise_floor_pcm(duration_ms=tail_ms, rms=0.002)
        for i in range(0, len(silence_tail), 1024):
            if backchannel_pcm and first_audio_chunk_time is not None:
                break
            await ws.send(silence_tail[i : i + 1024])
            await asyncio.sleep(0.032)

        # 3. Inject mid-turn backchannel / barge-in while bot is actively speaking
        if backchannel_pcm:
            bc_raw = strip_wav_header(backchannel_pcm)
            t_wait = 0.0
            while first_audio_chunk_time is None and t_wait < 3.0:
                await asyncio.sleep(0.02)
                t_wait += 0.02

            await asyncio.sleep(0.12)  # Wait 120ms into bot response
            for i in range(0, len(bc_raw), 1024):
                chunk = bc_raw[i : i + 1024]
                if len(chunk) < 1024:
                    chunk += b"\x00" * (1024 - len(chunk))
                await ws.send(chunk)
                await asyncio.sleep(0.032)

            # Stream 400ms post-backchannel silence so VAD closes the utterance
            post_bc_silence = generate_noise_floor_pcm(duration_ms=400, rms=0.002)
            for i in range(0, len(post_bc_silence), 1024):
                await ws.send(post_bc_silence[i : i + 1024])
                await asyncio.sleep(0.032)

        sender_finished.set()

    recv_task = asyncio.create_task(receiver_loop())
    send_task = asyncio.create_task(sender_loop())

    dynamic_timeout = (len(clean_speech) / 32000.0) + (2.5 if expect_no_turn_window_ms > 0 else 7.5)
    try:
        if expect_no_turn_window_ms > 0:
            await asyncio.wait_for(send_task, timeout=dynamic_timeout)
            recv_task.cancel()
        else:
            await asyncio.wait_for(asyncio.gather(send_task, recv_task), timeout=dynamic_timeout)
    except asyncio.TimeoutError:
        send_task.cancel()
        recv_task.cancel()

    return {
        "t_last_speech_frame": t_last_speech_frame,
        "bot_turn_start_time": bot_turn_start_time,
        "first_audio_chunk_time": first_audio_chunk_time,
        "bot_turn_end_time": bot_turn_end_time,
        "bot_audio_pcm": bytes(bot_audio_bytes),
        "control_messages": received_frames,
    }


class ComprehensiveTestRunner:
    """Executes all 8 test suites against the live Vartalaap WebSocket server."""

    def __init__(self, ws_url: str = "ws://localhost:7860/ws/audio") -> None:
        self.ws_url = ws_url
        self.validator = ResponseValidator()
        self.settings = get_settings()
        self.stt_client = GnaniSTTClient(settings=self.settings)
        self.samples_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_corpus_index()

    def _load_corpus_index(self) -> None:
        """Load corpus manifest index from disk."""
        if not INDEX_JSON.exists():
            raise FileNotFoundError(f"Corpus index not found at {INDEX_JSON}. Run corpus_builder.py first.")
        with open(INDEX_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("samples", []):
            self.samples_by_id[s["id"]] = s
        logger.info("Loaded %d corpus audio samples from %s", len(self.samples_by_id), INDEX_JSON)

    def get_sample_pcm(self, sample_id: str) -> bytes:
        """Load raw PCM bytes for a given sample ID."""
        item = self.samples_by_id.get(sample_id)
        if not item:
            raise KeyError(f"Sample '{sample_id}' not found in corpus manifest index")
        path = ROOT_DIR / item["file_path"]
        if not path.exists():
            raise FileNotFoundError(f"Audio file {path} missing on disk")
        with wave.open(str(path), "rb") as wf:
            return wf.readframes(wf.getnframes())

    # --------------------------------------------------------------------------
    # Suite F: Domain Recognition Accuracy (14 tests)
    # --------------------------------------------------------------------------
    async def run_suite_f(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite F: Domain Recognition Accuracy ===")
        results = []
        # 7 base domain queries + 7 multi-turn openers
        sample_ids = [
            "f_flights_01", "f_trains_01", "f_bill_01", "f_banking_01", "f_project_01", "f_greetings_01", "f_farewell_01",
            "g_flight_t1", "g_train_t1", "g_bill_t1", "g_bank_t1", "g_proj_t1", "g_greet_t1", "g_telecom_t1",
        ]
        for sid in sample_ids:
            meta = self.samples_by_id[sid]
            pcm = self.get_sample_pcm(sid)
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                raw = await run_full_duplex_turn(ws, speech_pcm=pcm)
            expectations = dict(meta)
            expectations["suite"] = "Suite F"
            eval_res = self.validator.evaluate_turn(raw, expectations)
            results.append(eval_res)
            logger.info("  [%s] %s: %s (E2E=%.0fms, Delta=%.0fms)", "PASS" if eval_res.passed else "FAIL", sid, eval_res.details, eval_res.t_e2e_ms, eval_res.delta_proc_ms)
        return results

    # --------------------------------------------------------------------------
    # Suite G: Multi-Turn Conversational Coherence (35 tests across 7 dialogues)
    # --------------------------------------------------------------------------
    async def run_suite_g(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite G: Multi-Turn Conversational Coherence ===")
        results = []
        dialogues = [
            ("flight_conv", ["g_flight_t1", "g_flight_t2", "g_flight_t3", "g_flight_t4", "g_flight_t5"]),
            ("train_conv", ["g_train_t1", "g_train_t2", "g_train_t3", "g_train_t4", "g_train_t5"]),
            ("bill_conv", ["g_bill_t1", "g_bill_t2", "g_bill_t3", "g_bill_t4", "g_bill_t5"]),
            ("bank_conv", ["g_bank_t1", "g_bank_t2", "g_bank_t3", "g_bank_t4", "g_bank_t5"]),
            ("proj_conv", ["g_proj_t1", "g_proj_t2", "g_proj_t3", "g_proj_t4", "g_proj_t5"]),
            ("greet_conv", ["g_greet_t1", "g_greet_t2", "g_greet_t3", "g_greet_t4", "g_greet_t5"]),
            ("telecom_conv", ["g_telecom_t1", "g_telecom_t2", "g_telecom_t3", "g_telecom_t4", "g_telecom_t5"]),
        ]

        for conv_id, turns in dialogues:
            logger.info("  Executing Dialogue: %s (5 turns)", conv_id)
            prior_replies: List[str] = []
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                for tid in turns:
                    meta = self.samples_by_id[tid]
                    pcm = self.get_sample_pcm(tid)
                    raw = await run_full_duplex_turn(ws, speech_pcm=pcm)
                    expectations = dict(meta)
                    expectations["suite"] = "Suite G"
                    expectations["prior_replies"] = list(prior_replies)
                    eval_res = self.validator.evaluate_turn(raw, expectations)
                    results.append(eval_res)
                    if eval_res.bot_reply:
                        prior_replies.append(eval_res.bot_reply)
                    logger.info("    [%s] Turn %s: %s (E2E=%.0fms)", "PASS" if eval_res.passed else "FAIL", tid, eval_res.details, eval_res.t_e2e_ms)
                    await asyncio.sleep(0.15)
        return results

    # --------------------------------------------------------------------------
    # Suite H: Devanagari Script & Hinglish Robustness (10 tests)
    # --------------------------------------------------------------------------
    async def run_suite_h(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite H: Devanagari Script & Hinglish Robustness ===")
        results = []
        sample_ids = [
            "dev_flight_01", "dev_bank_01", "dev_bill_01", "dev_train_01", "dev_project_01",
            "hing_flight_01", "hing_bank_01", "hing_bill_01", "hing_train_01", "hing_project_01",
        ]
        for sid in sample_ids:
            meta = self.samples_by_id[sid]
            pcm = self.get_sample_pcm(sid)
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                raw = await run_full_duplex_turn(ws, speech_pcm=pcm)
            expectations = dict(meta)
            expectations["suite"] = "Suite H"
            eval_res = self.validator.evaluate_turn(raw, expectations)
            results.append(eval_res)
            logger.info("  [%s] %s: %s (E2E=%.0fms)", "PASS" if eval_res.passed else "FAIL", sid, eval_res.details, eval_res.t_e2e_ms)
        return results

    # --------------------------------------------------------------------------
    # Suite I: City Pair Generalization (6 tests)
    # --------------------------------------------------------------------------
    async def run_suite_i(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite I: City Pair Generalization ===")
        results = []
        sample_ids = [
            "city_delhi_mumbai", "city_bangalore_chennai", "city_kolkata_hyderabad",
            "city_jaipur_pune", "city_ahmedabad_goa", "city_lucknow_delhi",
        ]
        for sid in sample_ids:
            meta = self.samples_by_id[sid]
            pcm = self.get_sample_pcm(sid)
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                raw = await run_full_duplex_turn(ws, speech_pcm=pcm)
            expectations = dict(meta)
            expectations["suite"] = "Suite I"
            eval_res = self.validator.evaluate_turn(raw, expectations)
            results.append(eval_res)
            logger.info("  [%s] %s: %s (Origin=%s, Dest=%s)", "PASS" if eval_res.passed else "FAIL", sid, eval_res.details, meta.get("origin"), meta.get("dest"))
        return results

    # --------------------------------------------------------------------------
    # Suite J: Turn-Taking Stress Tests (8 tests)
    # --------------------------------------------------------------------------
    async def run_suite_j(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite J: Turn-Taking Stress Tests ===")
        results = []

        # J1: Rapid Double Utterance
        pcm_rapid = self.get_sample_pcm("edge_rapid_01")
        pcm_cont = self.get_sample_pcm("edge_continuation")
        gap_silence = generate_noise_floor_pcm(duration_ms=200, rms=0.002)
        combined_rapid = pcm_rapid + gap_silence + pcm_cont

        async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
            raw_j1 = await run_full_duplex_turn(ws, speech_pcm=combined_rapid)
        meta_j1 = {
            "id": "j1_rapid_double",
            "suite": "Suite J",
            "text": "Rapid double utterance (Chandigarh flight + 200ms + mera budget thoda kam hai)",
            "domain": "FLIGHTS",
        }
        eval_j1 = self.validator.evaluate_turn(raw_j1, meta_j1)
        results.append(eval_j1)
        logger.info("  [%s] J1 Rapid Double Utterance: %s", "PASS" if eval_j1.passed else "FAIL", eval_j1.details)

        # J2: Ultra-Long Sentence (50+ words)
        pcm_long = self.get_sample_pcm("edge_ultra_long")
        async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
            raw_j2 = await run_full_duplex_turn(ws, speech_pcm=pcm_long)
        meta_j2 = {
            "id": "j2_ultra_long",
            "suite": "Suite J",
            "text": self.samples_by_id["edge_ultra_long"]["text"],
            "domain": "FLIGHTS",
        }
        eval_j2 = self.validator.evaluate_turn(raw_j2, meta_j2)
        results.append(eval_j2)
        logger.info("  [%s] J2 Ultra-Long Sentence: %s", "PASS" if eval_j2.passed else "FAIL", eval_j2.details)

        # J3: Single-Word Utterance
        pcm_single = self.get_sample_pcm("edge_single_word")
        async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
            raw_j3 = await run_full_duplex_turn(ws, speech_pcm=pcm_single)
        meta_j3 = {
            "id": "j3_single_word",
            "suite": "Suite J",
            "text": "Namaste",
            "domain": "GREETINGS",
        }
        eval_j3 = self.validator.evaluate_turn(raw_j3, meta_j3)
        results.append(eval_j3)
        logger.info("  [%s] J3 Single-Word Utterance: %s", "PASS" if eval_j3.passed else "FAIL", eval_j3.details)

        # J4: Empty Audio / Silence Rejection
        pcm_silence = self.get_sample_pcm("noise_silence")
        async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
            raw_j4 = await run_full_duplex_turn(ws, speech_pcm=pcm_silence, expect_no_turn_window_ms=3000)
        meta_j4 = {
            "id": "j4_silence_rejection",
            "suite": "Suite J",
            "text": "[SILENCE 3.0s]",
            "expect_no_turn": True,
            "hold_window_ms": 3000,
        }
        eval_j4 = self.validator.evaluate_turn(raw_j4, meta_j4)
        results.append(eval_j4)
        logger.info("  [%s] J4 Silence Rejection: %s", "PASS" if eval_j4.passed else "FAIL", eval_j4.details)

        # J5–J8: Incomplete Clauses (Trailing Conjunction Hold >= 1250ms)
        incomplete_ids = [("j5_lekin", "incomp_lekin"), ("j6_kyunki", "incomp_kyunki"), ("j7_aur", "incomp_aur"), ("j8_ya", "incomp_ya")]
        for test_name, incomp_id in incomplete_ids:
            pcm_incomp = self.get_sample_pcm(incomp_id)
            meta_incomp = self.samples_by_id[incomp_id]
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                raw_incomp = await run_full_duplex_turn(ws, speech_pcm=pcm_incomp, expect_no_turn_window_ms=1600)
            exp_incomp = {
                "id": test_name,
                "suite": "Suite J",
                "text": meta_incomp["text"],
                "has_trailing_conjunction": True,
            }
            eval_incomp = self.validator.evaluate_turn(raw_incomp, exp_incomp)
            results.append(eval_incomp)
            logger.info("  [%s] %s (%s): %s", "PASS" if eval_incomp.passed else "FAIL", test_name, incomp_id, eval_incomp.details)

        return results

    # --------------------------------------------------------------------------
    # Suite K: Concurrent Session Isolation (3 parallel sessions)
    # --------------------------------------------------------------------------
    async def run_suite_k(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite K: Concurrent Session Isolation (3 Parallel Sessions) ===")
        session_specs = [
            ("k_session_1", "f_flights_01", "FLIGHTS", ["flight", "flights", "direct", "chandigarh", "lucknow"]),
            ("k_session_2", "f_banking_01", "BANKING", ["balance", "khate", "savings", "rupaye", "account"]),
            ("k_session_3", "f_bill_01", "BILL", ["bill", "bijli", "rupaye", "due date", "pay"]),
        ]

        async def run_isolated_session(sid: str, samp_id: str) -> Dict[str, Any]:
            pcm = self.get_sample_pcm(samp_id)
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                return await run_full_duplex_turn(ws, speech_pcm=pcm)

        # Launch all 3 sessions simultaneously
        tasks = [run_isolated_session(sid, samp_id) for sid, samp_id, _, _ in session_specs]
        raw_outputs = await asyncio.gather(*tasks)

        results = []
        for (sid, samp_id, domain, kws), raw_res in zip(session_specs, raw_outputs):
            meta = {
                "id": sid,
                "suite": "Suite K",
                "name": f"Concurrent Session {sid} ({domain})",
                "text": self.samples_by_id[samp_id]["text"],
                "domain": domain,
                "expected_keywords": kws,
            }
            eval_res = self.validator.evaluate_turn(raw_res, meta)
            results.append(eval_res)
            logger.info("  [%s] %s: %s (Reply: '%s...')", "PASS" if eval_res.passed else "FAIL", sid, eval_res.details, eval_res.bot_reply[:40])
        return results

    # --------------------------------------------------------------------------
    # Suite M: Round-Trip Audio STT Verification (5 tests)
    # --------------------------------------------------------------------------
    async def run_suite_m(self) -> List[TurnEvaluation]:
        logger.info("=== Running Suite M: Round-Trip Audio STT Verification ===")
        sample_ids = ["f_flights_01", "f_trains_01", "f_bill_01", "f_banking_01", "f_project_01"]
        results = []

        for sid in sample_ids:
            meta = self.samples_by_id[sid]
            pcm = self.get_sample_pcm(sid)
            async with websockets.connect(self.ws_url, open_timeout=30.0) as ws:
                raw = await run_full_duplex_turn(ws, speech_pcm=pcm)

            bot_audio_pcm = raw.get("bot_audio_pcm", b"")
            stt_text = ""
            if len(bot_audio_pcm) > 3200:
                # Wrap accumulated PCM in standard WAV header for REST STT
                wav_bytes = GnaniTTSClient._wrap_pcm_as_wav(bot_audio_pcm, sample_rate=16000)
                try:
                    stt_res = self.stt_client.transcribe_rest(wav_bytes, audio_format="wav")
                    stt_text = stt_res.text
                    logger.info("  STT Decoded Bot Audio for %s: '%s'", sid, stt_text)
                except Exception as exc:
                    logger.warning("  STT Decode failed for %s: %s", sid, exc)
                    stt_text = ""
            else:
                logger.warning("  No sufficient bot audio PCM captured for %s (%d bytes)", sid, len(bot_audio_pcm))

            expectations = dict(meta)
            expectations["suite"] = "Suite M"
            expectations["id"] = f"m_{sid}"
            expectations["stt_decoded_text"] = stt_text
            eval_res = self.validator.evaluate_turn(raw, expectations)
            results.append(eval_res)
            logger.info("  [%s] Suite M %s: %s", "PASS" if eval_res.passed else "FAIL", sid, eval_res.details)

        return results

    async def warmup(self) -> None:
        """Perform initial pre-flight connection to warm up server GPU graph and STT buffers."""
        try:
            logger.info("Performing pre-flight warmup connection to %s...", self.ws_url)
            silence = generate_noise_floor_pcm(duration_ms=400, rms=0.001)
            async with websockets.connect(self.ws_url, open_timeout=20.0) as ws:
                await ws.send(silence)
                await asyncio.sleep(0.5)
            logger.info("Pre-flight warmup connection successful.")
        except Exception as exc:
            logger.debug("Pre-flight warmup completed with notice: %s", exc)

    # --------------------------------------------------------------------------
    # Master Execution Orchestrator
    # --------------------------------------------------------------------------
    async def run_all(self) -> Dict[str, Any]:
        """Execute all suites F through M and generate final reports."""
        all_evals: List[TurnEvaluation] = []
        t0 = time.time()

        await self.warmup()

        suite_f = await self.run_suite_f()
        all_evals.extend(suite_f)

        suite_g = await self.run_suite_g()
        all_evals.extend(suite_g)

        suite_h = await self.run_suite_h()
        all_evals.extend(suite_h)

        suite_i = await self.run_suite_i()
        all_evals.extend(suite_i)

        suite_j = await self.run_suite_j()
        all_evals.extend(suite_j)

        suite_k = await self.run_suite_k()
        all_evals.extend(suite_k)

        suite_m = await self.run_suite_m()
        all_evals.extend(suite_m)

        duration = round(time.time() - t0, 2)
        logger.info("All suites executed in %.2fs. Generating comprehensive reports...", duration)

        report = generate_reports(all_evals)

        print("\n" + "=" * 72)
        print(f" VARTALAAP ENTERPRISE QA COMPREHENSIVE REPORT: {report['passed_tests']}/{report['total_tests']} Passed ({report['pass_rate_pct']}%)")
        print("=" * 72)
        for s_name, st in report["suite_summary"].items():
            icon = "✓" if st["pass_rate_pct"] >= 95.0 else "✗"
            print(f" [{icon}] {s_name:<10}: {st['passed']}/{st['total']} Passed ({st['pass_rate_pct']}%)")
        print("-" * 72)
        lat = report["latency_percentiles"]
        print(f" Latency: p95 Delta_proc = {lat['p95_proc_ms']}ms (SLA <= 180ms) | p95 TTFA_proc = {lat['p95_ttfa_proc_ms']}ms (SLA <= 450ms)")
        print("=" * 72 + "\n")

        return report


if __name__ == "__main__":
    runner = ComprehensiveTestRunner()
    asyncio.run(runner.run_all())
