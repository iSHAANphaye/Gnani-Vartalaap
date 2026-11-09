"""Continuous Expected Turn Silence State Machine and Speculative LLM Prefetch Engine.

Coordinates acoustic Voice Activity Detection, multimodal Indic semantic completeness
classification, two-stage soft-ducking cultural backchannel arbitration, and speculative
conversational dialogue prefetching.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Callable, Optional, Union

import numpy as np
import torch

from app.config import Settings, get_settings
from app.core.backchannel import DuckingAction, GateDecision, TwoStageBackchannelGate
from app.core.semantic import (
    HybridTurnClassifier,
    SemanticTurnEvaluation,
    TurnCategory,
)
from app.core.vad import AcousticDetector, VADResult
from app.services.llm_brain import LLMBrain

logger = logging.getLogger("vartalaap.engine")


class TurnState(str, Enum):
    """Conversational turn arbitration state machine states."""

    LISTENING = "LISTENING"
    EVALUATING_PAUSE = "EVALUATING_PAUSE"
    SPECULATIVE_PREFETCHING = "SPECULATIVE_PREFETCHING"
    TURN_GRANTED = "TURN_GRANTED"
    BOT_SPEAKING = "BOT_SPEAKING"
    BOT_DUCKED = "BOT_DUCKED"
    BACKCHANNEL_IGNORED = "BACKCHANNEL_IGNORED"


@dataclass
class EngineTelemetry:
    """Structured turn-taking telemetry snapshot emitted by DynamicTurnEngine."""

    state: TurnState
    elapsed_silence_ms: float
    dynamic_ets_ms: float
    p_complete: float
    p_incomplete: float
    category: TurnCategory
    prefetch_active: bool
    ducking_gain_db: float
    reason: str
    response_text: Optional[str] = None
    transcript: str = ""
    is_speech: bool = False
    timestamp: float = field(default_factory=time.time)


class DynamicTurnEngine:
    """Continuous Expected Turn Silence (ETS) state machine and speculative prefetch coordinator.

    Calculates dynamic turn silence thresholds based on acoustic prosody and trailing Indic
    lexical markers, and manages low-latency speculative prefetch execution and abortion.
    """

    MIN_ETS_MS: float = 250.0
    MAX_ETS_MS: float = 2000.0
    STANDARD_ETS_MS: float = 500.0
    PREFETCH_SILENCE_MS: float = 200.0
    PREFETCH_COMPLETION_THRESHOLD: float = 0.75

    def __init__(
        self,
        detector: Optional[AcousticDetector] = None,
        classifier: Optional[HybridTurnClassifier] = None,
        gate: Optional[TwoStageBackchannelGate] = None,
        llm: Optional[LLMBrain] = None,
        mode: str = "VARTALAAP",
        settings: Optional[Settings] = None,
        on_telemetry: Optional[Callable[[EngineTelemetry], None]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.detector = detector or AcousticDetector()
        self.classifier = classifier or HybridTurnClassifier(settings=self.settings)
        self.gate = gate or TwoStageBackchannelGate()
        self.llm = llm or LLMBrain(settings=self.settings)

        self._mode = mode.upper()
        if self._mode not in ("VARTALAAP", "STANDARD"):
            self._mode = "VARTALAAP"

        self.on_telemetry = on_telemetry

        self.min_ets_ms = float(self.settings.min_silence_threshold_ms)
        self.max_ets_ms = float(self.settings.max_silence_threshold_ms)

        # Internal state
        self._state: TurnState = TurnState.LISTENING
        self._bot_is_speaking: bool = False
        self._ducking_gain_db: float = 0.0
        self._elapsed_silence_ms: float = 0.0
        self._speech_duration_ms: float = 0.0
        self._is_speech: bool = False
        self._last_transcript: str = ""
        self._dynamic_ets_ms: float = self.min_ets_ms
        self._last_eval: Optional[SemanticTurnEvaluation] = None
        self._user_has_spoken: bool = False

        # Speculative prefetch management
        self._prefetch_task: Optional[asyncio.Task[str]] = None
        self._prefetch_cancel_token: Optional[asyncio.Event] = None
        self._prefetched_reply: Optional[str] = None
        self._last_cancelled_task: Optional[asyncio.Task[str]] = None

        self._last_telemetry: Optional[EngineTelemetry] = None

        logger.info(
            "DynamicTurnEngine initialized (mode='%s', min_ets=%.0fms, max_ets=%.0fms)",
            self._mode,
            self.min_ets_ms,
            self.max_ets_ms,
        )

    @staticmethod
    def compute_dynamic_ets(
        p_acoustic: float,
        lexical_penalty: float,
        min_ets_ms: float = 250.0,
        max_ets_ms: float = 2000.0,
    ) -> float:
        """Calculate Continuous Expected Turn Silence in milliseconds.

        Formula:
            ETS(ms) = 250ms + (2000ms - 250ms) * clamp(P_acoustic + S_lexical_penalty, 0.0, 1.0)
        """
        composite = max(0.0, min(1.0, float(p_acoustic + lexical_penalty)))
        return min_ets_ms + (max_ets_ms - min_ets_ms) * composite

    @property
    def state(self) -> TurnState:
        """Current state machine turn arbitration state."""
        return self._state

    @property
    def mode(self) -> str:
        """Active operation mode ('VARTALAAP' or 'STANDARD')."""
        return self._mode

    @property
    def is_bot_speaking(self) -> bool:
        """Whether bot audio playback is currently flagged as active."""
        return self._bot_is_speaking

    @property
    def is_prefetch_active(self) -> bool:
        """Whether a speculative background prefetch task is active or awaiting resolution."""
        if self._prefetch_task is not None and not self._prefetch_task.done():
            return True
        return self._state == TurnState.SPECULATIVE_PREFETCHING

    @property
    def prefetch_task(self) -> Optional[asyncio.Task[str]]:
        """Active or most recent speculative prefetch background task."""
        return self._prefetch_task

    @property
    def last_cancelled_task(self) -> Optional[asyncio.Task[str]]:
        """Reference to the most recently aborted speculative task."""
        return self._last_cancelled_task

    @property
    def prefetched_reply(self) -> Optional[str]:
        """Pre-generated response ready for zero-latency turn dispatch."""
        return self._prefetched_reply

    @property
    def dynamic_ets_ms(self) -> float:
        """Current dynamic silence timeout threshold in milliseconds."""
        return self._dynamic_ets_ms

    @property
    def ducking_gain_db(self) -> float:
        """Current target audio playback attenuation gain in decibels."""
        return self._ducking_gain_db

    @property
    def elapsed_silence_ms(self) -> float:
        """Accumulated pause duration in milliseconds."""
        return self._elapsed_silence_ms

    @property
    def last_telemetry(self) -> Optional[EngineTelemetry]:
        """Most recent structured telemetry record."""
        return self._last_telemetry

    def set_mode(self, mode: str) -> None:
        """Switch arbitration mode between 'VARTALAAP' and 'STANDARD'."""
        clean = mode.upper()
        if clean in ("VARTALAAP", "STANDARD"):
            self._mode = clean
            logger.info("DynamicTurnEngine mode updated to %s", self._mode)

    def set_bot_speaking(self, speaking: bool) -> None:
        """Update bot audio playback status."""
        self._bot_is_speaking = bool(speaking)
        if not self._bot_is_speaking and self._state in (TurnState.BOT_SPEAKING, TurnState.BOT_DUCKED):
            self._ducking_gain_db = 0.0
            self._state = TurnState.LISTENING

    def reset(self) -> None:
        """Reset internal turn-taking state machine and all subcomponents."""
        self._cancel_prefetch()
        self.detector.reset()
        self.gate.reset()
        self._state = TurnState.LISTENING
        self._bot_is_speaking = False
        self._ducking_gain_db = 0.0
        self._elapsed_silence_ms = 0.0
        self._speech_duration_ms = 0.0
        self._is_speech = False
        self._user_has_spoken = False
        self._last_transcript = ""
        self._dynamic_ets_ms = self.min_ets_ms
        self._last_eval = None
        self._prefetched_reply = None
        self._last_cancelled_task = None
        self._last_telemetry = None

    def _start_prefetch(self, prompt: str) -> None:
        """Initiate asynchronous background speculative prefetch with cooperative cancellation."""
        if not prompt or not prompt.strip():
            return

        if self._prefetch_task is not None and not self._prefetch_task.done():
            return

        self._prefetch_cancel_token = asyncio.Event()
        cancel_tok = self._prefetch_cancel_token

        async def _runner() -> str:
            try:
                reply = await self.llm.generate_reply(prompt, cancel_token=cancel_tok)
                self._prefetched_reply = reply
                return reply
            except asyncio.CancelledError:
                logger.debug("Speculative prefetch runner received CancelledError")
                raise
            except Exception as exc:
                logger.warning("Speculative prefetch failed: %s", exc)
                return ""

        self._prefetch_task = asyncio.create_task(_runner())
        # Add no-op done callback to prevent unretrieved task exception warnings
        self._prefetch_task.add_done_callback(lambda t: None)
        logger.debug("Speculative prefetch launched for prompt: '%s'", prompt)

    def _cancel_prefetch(self) -> Optional[asyncio.Task[str]]:
        """Silently abort active speculative prefetch task without race conditions."""
        cancelled_task: Optional[asyncio.Task[str]] = None

        if self._prefetch_cancel_token is not None:
            self._prefetch_cancel_token.set()
            self._prefetch_cancel_token = None

        if self._prefetch_task is not None:
            if not self._prefetch_task.done():
                self._prefetch_task.cancel()
                logger.debug("Active speculative prefetch task aborted")
            cancelled_task = self._prefetch_task
            self._last_cancelled_task = cancelled_task
            self._prefetch_task = None

        self._prefetched_reply = None
        return cancelled_task

    async def step(
        self,
        chunk_bytes: Optional[bytes] = None,
        transcript: str = "",
        bot_is_speaking: Optional[bool] = None,
        elapsed_silence_override_ms: Optional[float] = None,
        speech_duration_override_ms: Optional[float] = None,
        is_speech_override: Optional[bool] = None,
    ) -> EngineTelemetry:
        """Advance conversational state machine by ingesting audio and/or ASR updates.

        Args:
            chunk_bytes: 16-bit mono 16kHz PCM audio chunk (512 samples / 1024 bytes).
            transcript: Interim or finalized ASR transcript text.
            bot_is_speaking: External flag indicating whether bot audio is actively playing.
            elapsed_silence_override_ms: Explicit silence duration for deterministic unit testing.
            speech_duration_override_ms: Explicit speech duration for testing backchannel/barge-in.
            is_speech_override: Explicit VAD voice presence override.

        Returns:
            EngineTelemetry structured record with complete state, metrics, and actions.
        """
        if bot_is_speaking is not None:
            self._bot_is_speaking = bool(bot_is_speaking)

        if transcript:
            self._last_transcript = transcript
        else:
            transcript = self._last_transcript

        # 1. Acoustic VAD Ingestion
        if chunk_bytes is not None:
            vad_res: VADResult = self.detector.process_chunk(chunk_bytes)
            is_speech = vad_res.is_speech
            elapsed_silence_ms = vad_res.silence_duration_ms
            speech_duration_ms = vad_res.speech_duration_ms
        else:
            is_speech = self._is_speech
            elapsed_silence_ms = self._elapsed_silence_ms
            speech_duration_ms = self._speech_duration_ms

        if is_speech_override is not None:
            is_speech = bool(is_speech_override)
        if elapsed_silence_override_ms is not None:
            elapsed_silence_ms = float(elapsed_silence_override_ms)
        if speech_duration_override_ms is not None:
            speech_duration_ms = float(speech_duration_override_ms)

        self._is_speech = is_speech
        self._elapsed_silence_ms = elapsed_silence_ms
        self._speech_duration_ms = speech_duration_ms

        response_text: Optional[str] = None
        reason: str = ""

        # ======================================================================
        # Branch 1: Bot Audio Playback Active (Full-Duplex Arbitration)
        # ======================================================================
        if self._bot_is_speaking:
            self._cancel_prefetch()

            if is_speech:
                if self.mode == "STANDARD":
                    # Standard Fixed VAD: Binary barge-in immediately interrupts bot playback
                    self._state = TurnState.LISTENING
                    self._bot_is_speaking = False
                    self._ducking_gain_db = -100.0
                    reason = "STANDARD_VAD_BARGE_IN: binary voice activity interruption"
                else:
                    # Stage 1: Soft-duck bot playback (-18 dB) immediately upon voice onset
                    if not self.gate.is_ducked:
                        self.gate.on_speech_started(bot_is_speaking=True)

                    # Stage 2: Evaluate interim transcript tokens and speech duration
                    decision: GateDecision = self.gate.evaluate_utterance(
                        transcript=transcript,
                        speech_duration_ms=speech_duration_ms,
                        bot_is_speaking=True,
                    )

                    if decision.action == DuckingAction.RESTORE:
                        # Cultural backchannel confirmed; restore playback to 0 dB, bot continues
                        self._state = TurnState.BACKCHANNEL_IGNORED
                        self._ducking_gain_db = 0.0
                        reason = f"BACKCHANNEL_IGNORED ({decision.reason}): volume restored to 0 dB, bot continues"
                    elif decision.action == DuckingAction.HARD_FLUSH:
                        # Imperative command or prolonged barge-in; stop bot playback immediately
                        self._state = TurnState.LISTENING
                        self._bot_is_speaking = False
                        self._ducking_gain_db = -100.0
                        reason = f"HARD_FLUSH ({decision.reason}): bot playback aborted, turn transferred"
                    else:
                        # Voice detected, awaiting token confirmation; remain soft-ducked at -18 dB
                        self._state = TurnState.BOT_DUCKED
                        self._ducking_gain_db = decision.gain_db
                        reason = f"BOT_DUCKED ({decision.reason}): playback attenuated to {decision.gain_db:.1f} dB"
            else:
                self._state = TurnState.BOT_SPEAKING
                self._ducking_gain_db = 0.0
                reason = "BOT_PLAYBACK_ACTIVE"

            p_complete = 0.0
            p_incomplete = 1.0
            category = TurnCategory.NEUTRAL
            dynamic_ets = self.STANDARD_ETS_MS if self.mode == "STANDARD" else self.max_ets_ms
            self._dynamic_ets_ms = dynamic_ets

        # ======================================================================
        # Branch 2: Bot Silent (User Ingestion & Turn-Taking Arbitration)
        # ======================================================================
        else:
            self._ducking_gain_db = 0.0

            has_text = bool(transcript and transcript.strip())
            if is_speech:
                self._user_has_spoken = True
                # User speaking actively; immediately abort any speculative prefetch task
                self._cancel_prefetch()
                self._state = TurnState.LISTENING
                reason = "USER_SPEAKING"

                eval_res = self.classifier.evaluate(None, transcript)
                self._last_eval = eval_res
                p_complete = eval_res.p_complete
                p_incomplete = eval_res.p_incomplete
                category = eval_res.category

                if self.mode == "STANDARD":
                    dynamic_ets = self.STANDARD_ETS_MS
                else:
                    dynamic_ets = self.compute_dynamic_ets(
                        eval_res.p_acoustic,
                        eval_res.lexical_penalty,
                        min_ets_ms=self.min_ets_ms,
                        max_ets_ms=self.max_ets_ms,
                    )
                self._dynamic_ets_ms = dynamic_ets

            else:
                # If user has not spoken yet, remain in passive listening without granting turn
                if not self._user_has_spoken and not has_text:
                    self._state = TurnState.LISTENING
                    reason = "IDLE_LISTENING: awaiting speech onset"
                    p_complete = 0.0
                    p_incomplete = 1.0
                    category = TurnCategory.NEUTRAL
                    dynamic_ets = self.STANDARD_ETS_MS if self.mode == "STANDARD" else self.min_ets_ms
                    self._dynamic_ets_ms = dynamic_ets
                else:
                    # User in pause after speaking; evaluate acoustic prosody and lexical postpositions
                    recent_audio = self.detector.get_recent_audio(duration_sec=3.0) if chunk_bytes else None
                    eval_res = self.classifier.evaluate(recent_audio, transcript)
                    self._last_eval = eval_res
                    p_complete = eval_res.p_complete
                    p_incomplete = eval_res.p_incomplete
                    category = eval_res.category

                    if self.mode == "STANDARD":
                        dynamic_ets = self.STANDARD_ETS_MS
                    else:
                        dynamic_ets = self.compute_dynamic_ets(
                            eval_res.p_acoustic,
                            eval_res.lexical_penalty,
                            min_ets_ms=self.min_ets_ms,
                            max_ets_ms=self.max_ets_ms,
                        )
                    self._dynamic_ets_ms = dynamic_ets

                    # State Transition Condition 1: Pause duration has reached or exceeded ETS
                    if elapsed_silence_ms >= dynamic_ets:
                        self._state = TurnState.TURN_GRANTED
                        self._user_has_spoken = False
                        self.detector.reset()
                        reason = (
                            f"TURN_GRANTED: silence ({elapsed_silence_ms:.0f}ms) "
                            f"reached dynamic ETS ({dynamic_ets:.0f}ms)"
                        )

                        # Resolve prefetched reply
                        if self._prefetch_task is not None:
                            if self._prefetch_task.done():
                                if not self._prefetch_task.cancelled():
                                    try:
                                        self._prefetched_reply = self._prefetch_task.result()
                                    except Exception as exc:
                                        logger.warning("Prefetch task exception: %s", exc)
                            else:
                                try:
                                    self._prefetched_reply = await asyncio.wait_for(
                                        asyncio.shield(self._prefetch_task),
                                        timeout=1.0,
                                    )
                                except Exception as exc:
                                    logger.warning("Timed out awaiting prefetch task: %s", exc)

                        if not self._prefetched_reply:
                            self._prefetched_reply = await self.llm.generate_reply(transcript)

                        response_text = self._prefetched_reply
                        self._prefetch_task = None
                        self._prefetch_cancel_token = None

                    # State Transition Condition 2: Silence >= 200ms and P(complete) > 0.75 -> Speculative Prefetch
                    elif (
                        self.mode == "VARTALAAP"
                        and elapsed_silence_ms >= self.PREFETCH_SILENCE_MS
                        and p_complete > self.PREFETCH_COMPLETION_THRESHOLD
                    ):
                        if self._prefetch_task is None:
                            self._start_prefetch(transcript)
                        self._state = TurnState.SPECULATIVE_PREFETCHING
                        reason = (
                            f"SPECULATIVE_PREFETCHING: silence ({elapsed_silence_ms:.0f}ms) >= 200ms "
                            f"and P(complete)={p_complete:.2f} > 0.75"
                        )

                    # State Transition Condition 3: Silence < 200ms or P(complete) <= 0.75 -> Evaluating pause
                    else:
                        self._state = TurnState.EVALUATING_PAUSE
                        reason = (
                            f"EVALUATING_PAUSE: silence ({elapsed_silence_ms:.0f}ms) "
                            f"< dynamic ETS ({dynamic_ets:.0f}ms)"
                        )
        telemetry = EngineTelemetry(
            state=self._state,
            elapsed_silence_ms=float(elapsed_silence_ms),
            dynamic_ets_ms=float(self._dynamic_ets_ms),
            p_complete=float(p_complete),
            p_incomplete=float(p_incomplete),
            category=category,
            prefetch_active=self.is_prefetch_active,
            ducking_gain_db=float(self._ducking_gain_db),
            reason=reason,
            response_text=response_text,
            transcript=transcript,
            is_speech=is_speech,
        )

        self._last_telemetry = telemetry
        if self.on_telemetry is not None:
            try:
                self.on_telemetry(telemetry)
            except Exception as exc:
                logger.warning("Error executing on_telemetry callback: %s", exc)

        return telemetry
