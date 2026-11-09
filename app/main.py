"""Full-Duplex FastAPI WebSocket Server for Project Vartalaap.

Coordinates full-duplex conversational audio streaming, acoustic Voice Activity
Detection, on-device Indic turn-boundary classification on NVIDIA RTX 5060 GPU,
two-stage cultural backchannel gating, speculative LLM dialogue prefetching,
and streaming speech synthesis.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import torch

from app.config import Settings, get_settings
from app.core.backchannel import DuckingAction, TwoStageBackchannelGate
from app.core.engine import DynamicTurnEngine, EngineTelemetry, TurnState
from app.core.semantic import HybridTurnClassifier
from app.core.vad import AcousticDetector
from app.services.gnani_stt import GnaniSTTClient
from app.services.gnani_tts import GnaniTTSClient
from app.services.llm_brain import LLMBrain
from app.services.smart_turn import get_smart_turn_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vartalaap.server")

# Path constants
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FRAME_BYTES = 1024  # 512 samples @ 16kHz 16-bit mono PCM


def get_system_hardware_info() -> Tuple[bool, str]:
    """Retrieve PyTorch CUDA hardware acceleration information."""
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        except Exception:
            return True, "CUDA Accelerate Active"
    return False, "None"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for model pre-warming and cleanup."""
    settings = get_settings()
    cuda_ok, gpu_name = get_system_hardware_info()
    logger.info("Initializing Project Vartalaap Server (Host: %s, Port: %d)", settings.host, settings.port)
    logger.info("Hardware Acceleration: CUDA=%s, Device='%s'", cuda_ok, gpu_name)

    # Pre-warm Smart-Turn acoustic engine on target device
    try:
        engine = get_smart_turn_engine(settings=settings)
        logger.info("Smart-Turn acoustic engine pre-warmed on device: %s", engine.device)
    except Exception as exc:
        logger.warning("Smart-Turn acoustic engine warmup encountered notice: %s", exc)

    yield

    logger.info("Shutting down Project Vartalaap Server")


# Instantiate FastAPI Application
app = FastAPI(
    title="Project Vartalaap Engine",
    description="Full-duplex Indic Conversational Turn-Taking and Speech Arbitration Server",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static UI HUD
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/", include_in_schema=False)
async def root_index() -> FileResponse:
    """Serve the finalized Vintage Nagra Tape Deck console directly at root."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/templates", include_in_schema=False)
async def templates_redirect() -> RedirectResponse:
    """Redirect templates endpoint to the Master Design Studio Gallery."""
    return RedirectResponse(url="/static/templates/gallery.html")


@app.get("/gallery", include_in_schema=False)
async def gallery_redirect() -> RedirectResponse:
    """Redirect gallery endpoint to the Master Design Studio Gallery."""
    return RedirectResponse(url="/static/templates/gallery.html")


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Healthcheck endpoint detailing runtime status, hardware, and service readiness."""
    cuda_ok, gpu_name = get_system_hardware_info()
    settings = get_settings()

    return JSONResponse(
        content={
            "status": "ok",
            "gpu": gpu_name,
            "torch_cuda": cuda_ok,
            "gnani_stt": settings.has_valid_gnani_credentials,
            "gnani_tts": settings.has_valid_gnani_credentials,
        }
    )


class ThreadSafeWebSocket:
    """Thread-safe WebSocket sender ensuring atomic frame delivery without interleaving."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._lock = asyncio.Lock()

    async def send_json(self, data: Dict[str, Any]) -> None:
        async with self._lock:
            await self._ws.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        async with self._lock:
            await self._ws.send_bytes(data)


@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket) -> None:
    """Full-duplex bi-directional audio streaming and state telemetry endpoint."""
    await websocket.accept()
    session_id = uuid.uuid4().hex[:8]
    safe_ws = ThreadSafeWebSocket(websocket)

    logger.info("WebSocket connected [session=%s, client=%s]", session_id, websocket.client)

    # Initialize per-connection conversational pipeline components
    settings = get_settings()
    detector = AcousticDetector(sample_rate=16000, ring_buffer_duration_sec=8.0)
    classifier = HybridTurnClassifier(settings=settings)
    gate = TwoStageBackchannelGate()
    llm = LLMBrain(settings=settings)
    engine = DynamicTurnEngine(
        detector=detector,
        classifier=classifier,
        gate=gate,
        llm=llm,
        mode="VARTALAAP",
        settings=settings,
    )
    stt = GnaniSTTClient(settings=settings)
    tts = GnaniTTSClient(settings=settings)

    # Connection local state
    session_state = {
        "transcript": "",
        "injected_transcript": "",
        "prev_ducking_action": DuckingAction.NONE,
    }
    pcm_buffer = bytearray()
    tts_task: Optional[asyncio.Task[None]] = None
    stt_task: Optional[asyncio.Task[None]] = None

    # Connect streaming speech-to-text service
    await stt.connect()

    async def _stt_receive_loop() -> None:
        """Asynchronously stream interim and final transcripts to client."""
        try:
            async for transcript_item in stt.stream_transcripts():
                if transcript_item.text:
                    if not session_state.get("injected_transcript"):
                        session_state["transcript"] = transcript_item.text
                    await safe_ws.send_json(
                        {
                            "type": "TRANSCRIPT_STREAM",
                            "text": transcript_item.text,
                            "is_final": transcript_item.is_final,
                            "language": transcript_item.language,
                        }
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Session %s STT receive loop terminated: %s", session_id, exc)

    stt_task = asyncio.create_task(_stt_receive_loop())

    async def _stream_tts_response(text: str) -> None:
        """Asynchronously synthesize and stream audio chunks to client with cooperative yielding."""
        try:
            engine.set_bot_speaking(True)
            await safe_ws.send_json(
                {
                    "type": "BOT_TURN_START",
                    "text": text,
                }
            )

            async for pcm_chunk in tts.synthesize_stream(text):
                b64_audio = base64.b64encode(pcm_chunk).decode("ascii")
                await safe_ws.send_json(
                    {
                        "type": "AUDIO_CHUNK",
                        "audio_base64": b64_audio,
                        "format": "audio/wav",
                        "is_final": False,
                        "is_final_chunk": False,
                    }
                )
                await asyncio.sleep(0.05)

            # Signal end of synthesized response stream
            await safe_ws.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "audio_base64": "",
                    "format": "audio/wav",
                    "is_final": True,
                    "is_final_chunk": True,
                }
            )
            await safe_ws.send_json(
                {
                    "type": "BOT_TURN_END",
                    "text": text,
                }
            )
        except asyncio.CancelledError:
            logger.info("Session %s: Bot TTS streaming cancelled by barge-in", session_id)
            raise
        except Exception as exc:
            logger.error("Session %s: Error streaming bot TTS audio: %s", session_id, exc)
        finally:
            engine.set_bot_speaking(False)

    try:
        while True:
            msg = await websocket.receive()
            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                logger.info("Session %s: Disconnect frame received", session_id)
                break

            # ------------------------------------------------------------------
            # 1. Incoming Binary Audio Frame Processing
            # ------------------------------------------------------------------
            if "bytes" in msg and msg["bytes"]:
                pcm_buffer.extend(msg["bytes"])

                while len(pcm_buffer) >= AUDIO_FRAME_BYTES:
                    chunk = bytes(pcm_buffer[:AUDIO_FRAME_BYTES])
                    del pcm_buffer[:AUDIO_FRAME_BYTES]

                    # Push audio to streaming ASR
                    await stt.send_audio_chunk(chunk)

                    # Step dynamic conversational turn-taking engine
                    current_transcript = session_state.get("injected_transcript") or session_state["transcript"]
                    telemetry: EngineTelemetry = await engine.step(
                        chunk_bytes=chunk,
                        transcript=current_transcript,
                        bot_is_speaking=engine.is_bot_speaking,
                    )

                    # Inspect cultural backchannel and soft-ducking gate directives
                    current_action = engine.gate.current_action
                    prev_action = session_state["prev_ducking_action"]

                    if current_action == DuckingAction.HARD_FLUSH:
                        if tts_task is not None and not tts_task.done():
                            tts_task.cancel()
                            try:
                                await tts_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            tts_task = None

                        # Clear transcript state on hard flush to avoid replaying interrupted response
                        session_state["transcript"] = ""
                        session_state["injected_transcript"] = ""
                        engine.clear_transcript()

                        if prev_action != DuckingAction.HARD_FLUSH:
                            await safe_ws.send_json(
                                {
                                    "type": "CONTROL",
                                    "action": "HARD_FLUSH",
                                    "gain_db": -100.0,
                                    "reason": telemetry.reason,
                                }
                            )
                            session_state["prev_ducking_action"] = DuckingAction.HARD_FLUSH

                    elif current_action == DuckingAction.DUCK:
                        if prev_action != DuckingAction.DUCK:
                            await safe_ws.send_json(
                                {
                                    "type": "CONTROL",
                                    "action": "DUCK",
                                    "gain_db": telemetry.ducking_gain_db,
                                    "reason": telemetry.reason,
                                }
                            )
                            session_state["prev_ducking_action"] = DuckingAction.DUCK

                    elif current_action == DuckingAction.RESTORE:
                        if prev_action != DuckingAction.RESTORE:
                            await safe_ws.send_json(
                                {
                                    "type": "CONTROL",
                                    "action": "RESTORE",
                                    "gain_db": 0.0,
                                    "reason": telemetry.reason,
                                }
                            )
                            session_state["prev_ducking_action"] = DuckingAction.RESTORE

                    elif current_action == DuckingAction.NONE:
                        if prev_action == DuckingAction.DUCK:
                            await safe_ws.send_json(
                                {
                                    "type": "CONTROL",
                                    "action": "RESTORE",
                                    "gain_db": 0.0,
                                    "reason": "SPEECH_ENDED_RESTORE",
                                }
                            )
                        session_state["prev_ducking_action"] = DuckingAction.NONE

                    # Send structured real-time telemetry packet to client HUD
                    await safe_ws.send_json(
                        {
                            "type": "TELEMETRY",
                            "state": telemetry.state.value,
                            "elapsed_silence_ms": round(float(telemetry.elapsed_silence_ms), 1),
                            "dynamic_ets_ms": round(float(telemetry.dynamic_ets_ms), 1),
                            "p_complete": round(float(telemetry.p_complete), 4),
                            "category": telemetry.category.value,
                            "ducking_gain_db": round(float(telemetry.ducking_gain_db), 1),
                            "reason": telemetry.reason,
                        }
                    )

                    # Handle turn granted state transition
                    if telemetry.state == TurnState.TURN_GRANTED:
                        # Guard: If bot is already speaking or TTS is actively generating, do not re-trigger
                        if engine.is_bot_speaking or (tts_task is not None and not tts_task.done()):
                            logger.debug("Session %s: Turn granted but bot already speaking; skipping", session_id)
                        else:
                            prompt_to_answer = current_transcript.strip()
                            if not prompt_to_answer and telemetry.response_text:
                                prompt_to_answer = telemetry.response_text.strip()

                            if not prompt_to_answer:
                                logger.debug("Session %s: Turn granted with empty transcript; ignoring", session_id)
                                engine.clear_transcript()
                                continue

                            response_text = telemetry.response_text
                            if not response_text:
                                response_text = await llm.generate_reply(prompt_to_answer)
                            if not response_text:
                                response_text = "Ji boliye, main sun raha hoon."

                            logger.info("Session %s: Turn granted. Replying: '%s'", session_id, response_text)
                            # Reset session transcripts for next turn
                            session_state["transcript"] = ""
                            session_state["injected_transcript"] = ""
                            engine.clear_transcript()

                            tts_task = asyncio.create_task(_stream_tts_response(response_text))

            # ------------------------------------------------------------------
            # 2. Incoming JSON Control Messages Processing
            # ------------------------------------------------------------------
            elif "text" in msg and msg["text"]:
                raw_text = msg["text"]
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    logger.warning("Session %s: Malformed JSON control message: '%s'", session_id, raw_text)
                    continue

                action = str(payload.get("action", "")).upper()

                if action == "PING":
                    await safe_ws.send_json(
                        {
                            "type": "PONG",
                            "timestamp": time.time(),
                        }
                    )

                elif action == "SET_MODE":
                    target_mode = str(payload.get("mode", "VARTALAAP")).upper()
                    engine.set_mode(target_mode)
                    logger.info("Session %s: Arbitration mode switched to %s", session_id, engine.mode)
                    await safe_ws.send_json(
                        {
                            "type": "CONTROL_ACK",
                            "action": "SET_MODE",
                            "mode": engine.mode,
                            "status": "ok",
                        }
                    )

                elif action == "RESET":
                    if tts_task is not None and not tts_task.done():
                        tts_task.cancel()
                        try:
                            await tts_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        tts_task = None

                    engine.reset()
                    session_state["transcript"] = ""
                    session_state["injected_transcript"] = ""
                    session_state["prev_ducking_action"] = DuckingAction.NONE
                    pcm_buffer.clear()
                    logger.info("Session %s: Session reset successfully", session_id)
                    await safe_ws.send_json(
                        {
                            "type": "CONTROL_ACK",
                            "action": "RESET",
                            "status": "ok",
                        }
                    )

                elif action in ("SET_TRANSCRIPT", "INJECT_TRANSCRIPT"):
                    injected_text = str(payload.get("text", "")).strip()
                    session_state["transcript"] = injected_text
                    session_state["injected_transcript"] = injected_text
                    logger.info("Session %s: Injected transcript: '%s'", session_id, injected_text)
                    await safe_ws.send_json(
                        {
                            "type": "TRANSCRIPT_STREAM",
                            "text": injected_text,
                            "is_final": True,
                            "language": "hi-IN",
                        }
                    )

                elif action == "STOP_CALL":
                    if tts_task is not None and not tts_task.done():
                        tts_task.cancel()
                    await safe_ws.send_json(
                        {
                            "type": "CONTROL_ACK",
                            "action": "STOP_CALL",
                            "status": "ok",
                        }
                    )
                    break

                else:
                    logger.warning("Session %s: Unsupported control action '%s'", session_id, action)

    except WebSocketDisconnect:
        logger.info("Session %s: WebSocket disconnected by client", session_id)
    except Exception as exc:
        logger.error("Session %s: Unexpected WebSocket exception: %s", session_id, exc, exc_info=True)
    finally:
        # Resource cleanup on connection termination
        if tts_task is not None and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except (asyncio.CancelledError, Exception):
                pass

        if stt_task is not None and not stt_task.done():
            stt_task.cancel()
            try:
                await stt_task
            except (asyncio.CancelledError, Exception):
                pass

        await stt.close()
        engine.reset()
        logger.info("Session %s: Connection lifecycle completed cleanly", session_id)
