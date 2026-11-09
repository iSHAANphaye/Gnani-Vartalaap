"""Two-Stage Soft-Ducking Cultural Gate and Conversational Turn Arbitration.

Implements a full-duplex conversational gate that distinguishes between brief Indic
cultural backchannels (e.g. "haanji", "achha", "seri", "సరే"), imperative interrupt
commands (e.g. "ruko", "cancel", "stop"), and sustained user barge-ins.

Stage 1: Soft-ducks bot playback to -18 dB at VAD speech detection (t ~ 2ms).
Stage 2: Evaluates interim ASR transcript and speech duration (t = 180-320ms) to
         either restore playback volume to 0 dB or hard-flush bot playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import FrozenSet, Iterable, Optional, Set
import unicodedata

from app.core.normalizer import CrossScriptNormalizer

logger = logging.getLogger("vartalaap.backchannel")


class DuckingAction(str, Enum):
    """Playback audio ducking action directives."""

    NONE = "NONE"
    DUCK = "DUCK"
    RESTORE = "RESTORE"
    HARD_FLUSH = "HARD_FLUSH"


@dataclass(frozen=True)
class GateDecision:
    """Decision output produced by the conversational backchannel gate."""

    action: DuckingAction
    gain_db: float
    reason: str
    is_backchannel: bool
    is_barge_in: bool


# Canonical catalog of imperative interrupt commands across Indic scripts and English
DEFAULT_INTERRUPT_COMMANDS: FrozenSet[str] = frozenset({
    # Hindi (Romanized & Devanagari)
    "ruko",
    "rukiye",
    "ruk",
    "roko",
    "chup",
    "chup raho",
    "chup baitho",
    "band karo",
    "bas",
    "bas karo",
    "khatam",
    "रुको",
    "रुकिए",
    "रुक",
    "रोको",
    "चुप",
    "चुप रहो",
    "बंद करो",
    "बस",
    "बस करो",

    # English
    "stop",
    "cancel",
    "wait",
    "hold on",
    "shut up",
    "pause",
    "stop it",
    "wait wait",
    "quit",
    "abort",
    "mute",
    "silence",
    "enough",

    # Tamil (Romanized & Native)
    "niruthu",
    "niruthunga",
    "pothum",
    "vaiyai moodu",
    "நிறுத்து",
    "நிறுத்துங்க",
    "போதும்",

    # Telugu (Romanized & Native)
    "aapu",
    "aapandi",
    "chal",
    "chalu",
    "aagandi",
    "ఆపు",
    "ఆపండి",
    "చాలు",
    "ఆగండి",

    # Kannada (Romanized & Native)
    "nillisi",
    "saaku",
    "saku",
    "nillu",
    "ನಿಲ್ಲಿಸಿ",
    "ಸಾಕು",
    "ನಿಲ್ಲು",
})


class TwoStageBackchannelGate:
    """Two-stage soft-ducking conversational turn and backchannel arbitrator.

    Prevents jarring bot playback cuts on brief cultural acknowledgments
    while ensuring low-latency hard-flush cutoffs on explicit barge-ins
    or imperative commands.
    """

    def __init__(
        self,
        normalizer: Optional[CrossScriptNormalizer] = None,
        interrupt_commands: Optional[Iterable[str]] = None,
        soft_duck_db: float = -18.0,
        hard_flush_db: float = -100.0,
        prolonged_threshold_ms: float = 500.0,
    ) -> None:
        self._normalizer = normalizer if normalizer is not None else CrossScriptNormalizer()
        self._soft_duck_db = float(soft_duck_db)
        self._hard_flush_db = float(hard_flush_db)
        self._prolonged_threshold_ms = float(prolonged_threshold_ms)

        # Build interrupt commands index
        cmds: Set[str] = set()
        raw_cmds = interrupt_commands if interrupt_commands is not None else DEFAULT_INTERRUPT_COMMANDS
        for cmd in raw_cmds:
            norm_cmd = self._normalizer.normalize(cmd).strip().lower()
            if norm_cmd:
                cmds.add(norm_cmd)
            raw_clean = unicodedata.normalize("NFKC", cmd).strip().lower()
            if raw_clean:
                cmds.add(raw_clean)

        self._interrupt_commands: FrozenSet[str] = frozenset(cmds)
        self._multiword_interrupts: FrozenSet[str] = frozenset(
            c for c in self._interrupt_commands if " " in c
        )

        # State tracking
        self._is_ducked: bool = False
        self._current_action: DuckingAction = DuckingAction.NONE
        self._current_gain_db: float = 0.0
        self._last_decision: Optional[GateDecision] = None

    @property
    def normalizer(self) -> CrossScriptNormalizer:
        """Linguistic normalizer used for script parsing."""
        return self._normalizer

    @property
    def backchannel_lexicon(self) -> FrozenSet[str]:
        """Recognized backchannel tokens and variants."""
        return self._normalizer.backchannel_variants

    @property
    def interrupt_commands(self) -> FrozenSet[str]:
        """Recognized imperative interrupt commands."""
        return self._interrupt_commands

    @property
    def is_ducked(self) -> bool:
        """Whether playback volume is currently attenuated."""
        return self._is_ducked

    @property
    def current_gain_db(self) -> float:
        """Current target audio playback gain in decibels."""
        return self._current_gain_db

    @property
    def current_action(self) -> DuckingAction:
        """Most recent ducking action directive."""
        return self._current_action

    @property
    def last_decision(self) -> Optional[GateDecision]:
        """Most recent gate decision, if any."""
        return self._last_decision

    @property
    def prolonged_threshold_ms(self) -> float:
        """Speech duration ceiling in milliseconds before classifying as barge-in."""
        return self._prolonged_threshold_ms

    def reset(self) -> None:
        """Reset internal gate tracking and audio gain to quiescent state."""
        self._is_ducked = False
        self._current_action = DuckingAction.NONE
        self._current_gain_db = 0.0
        self._last_decision = None

    def is_interrupt_command(self, text: str) -> bool:
        """Evaluate whether text contains an imperative interrupt command."""
        if not text:
            return False

        norm = self._normalizer.normalize(text).strip()
        raw = unicodedata.normalize("NFKC", text).strip().lower()

        # Direct match on normalized or raw utterance
        if norm in self._interrupt_commands or raw in self._interrupt_commands:
            return True

        # Check individual tokens in normalized text
        tokens = norm.split()
        if any(t in self._interrupt_commands for t in tokens):
            return True

        # Check individual tokens in raw text
        raw_tokens = raw.split()
        if any(t in self._interrupt_commands for t in raw_tokens):
            return True

        # Check multi-word phrase containment
        for phrase in self._multiword_interrupts:
            if phrase in norm or phrase in raw:
                return True

        return False

    def is_backchannel(self, text: str) -> bool:
        """Evaluate whether text is a recognized backchannel token or phrase."""
        if not text:
            return False
        # Interrupt commands take strict precedence over backchannels
        if self.is_interrupt_command(text):
            return False
        return self._normalizer.is_backchannel(text)

    def on_speech_started(self, bot_is_speaking: bool) -> GateDecision:
        """Stage 1: Process immediate VAD speech start detection (t ~ 2ms).

        If the bot is actively playing synthesized audio, softly ducks the playback
        to -18.0 dB without interrupting speech. If the bot is silent, no ducking
        action is necessary.
        """
        if bot_is_speaking:
            self._is_ducked = True
            self._current_action = DuckingAction.DUCK
            self._current_gain_db = self._soft_duck_db
            decision = GateDecision(
                action=DuckingAction.DUCK,
                gain_db=self._soft_duck_db,
                reason="VAD_SPEECH_STARTED_DUCK",
                is_backchannel=False,
                is_barge_in=False,
            )
        else:
            self._is_ducked = False
            self._current_action = DuckingAction.NONE
            self._current_gain_db = 0.0
            decision = GateDecision(
                action=DuckingAction.NONE,
                gain_db=0.0,
                reason="BOT_SILENT",
                is_backchannel=False,
                is_barge_in=False,
            )

        self._last_decision = decision
        return decision

    def evaluate_utterance(
        self,
        transcript: str,
        speech_duration_ms: float,
        bot_is_speaking: bool,
    ) -> GateDecision:
        """Stage 2: Evaluate interim transcript tokens and accumulated speech duration (t = 180-320ms).

        Cases:
        - Case A (Backchannel): bot_is_speaking=True AND speech_duration_ms < 500ms AND token in Backchannel Lexicon:
            Restores playback volume to 0.0 dB (BACKCHANNEL_IGNORED). Bot continues talking.
        - Case B (Imperative Command): bot_is_speaking=True AND token in Interrupt Commands:
            Triggers HARD_FLUSH to -100.0 dB (IMPERATIVE_BARGE_IN).
        - Case C (Extended Speech): bot_is_speaking=True AND speech_duration_ms >= 500ms:
            Triggers HARD_FLUSH to -100.0 dB (PROLONGED_BARGE_IN). User is speaking at length.
        - Non-speaking or pending:
            Appropriate quiescent or ducked status maintained.
        """
        if not bot_is_speaking:
            is_bc = self.is_backchannel(transcript)
            decision = GateDecision(
                action=DuckingAction.NONE,
                gain_db=0.0,
                reason="BOT_SILENT",
                is_backchannel=is_bc,
                is_barge_in=False,
            )
            self._last_decision = decision
            return decision

        # Case B: Imperative Interrupt Command (e.g. "ruko", "cancel", "stop")
        if self.is_interrupt_command(transcript):
            self._is_ducked = False
            self._current_action = DuckingAction.HARD_FLUSH
            self._current_gain_db = self._hard_flush_db
            decision = GateDecision(
                action=DuckingAction.HARD_FLUSH,
                gain_db=self._hard_flush_db,
                reason="IMPERATIVE_BARGE_IN",
                is_backchannel=False,
                is_barge_in=True,
            )
            self._last_decision = decision
            return decision

        # Case A: Cultural Backchannel (< 500ms duration and in backchannel lexicon)
        if speech_duration_ms < self._prolonged_threshold_ms and self.is_backchannel(transcript):
            self._is_ducked = False
            self._current_action = DuckingAction.RESTORE
            self._current_gain_db = 0.0
            decision = GateDecision(
                action=DuckingAction.RESTORE,
                gain_db=0.0,
                reason="BACKCHANNEL_IGNORED",
                is_backchannel=True,
                is_barge_in=False,
            )
            self._last_decision = decision
            return decision

        has_text = bool(transcript and transcript.strip())

        # Case C: Extended Speech / Prolonged Barge-In
        # If words have been recognized: prolonged speech >= 500ms triggers HARD_FLUSH
        if has_text and speech_duration_ms >= self._prolonged_threshold_ms:
            self._is_ducked = False
            self._current_action = DuckingAction.HARD_FLUSH
            self._current_gain_db = self._hard_flush_db
            decision = GateDecision(
                action=DuckingAction.HARD_FLUSH,
                gain_db=self._hard_flush_db,
                reason="PROLONGED_BARGE_IN",
                is_backchannel=False,
                is_barge_in=True,
            )
            self._last_decision = decision
            return decision

        # If transcript is empty (no lexical confirmation), do not abort playback on short sounds
        # like breathing or coughing. Only trigger after >= 1200ms of sustained acoustic sound.
        if not has_text and speech_duration_ms >= 1200.0:
            self._is_ducked = False
            self._current_action = DuckingAction.HARD_FLUSH
            self._current_gain_db = self._hard_flush_db
            decision = GateDecision(
                action=DuckingAction.HARD_FLUSH,
                gain_db=self._hard_flush_db,
                reason="PROLONGED_BARGE_IN",
                is_backchannel=False,
                is_barge_in=True,
            )
            self._last_decision = decision
            return decision

        # Pending speech (< 500ms with text, or < 1200ms without text): remains soft-ducked at -18 dB
        self._is_ducked = True
        self._current_action = DuckingAction.DUCK
        self._current_gain_db = self._soft_duck_db
        decision = GateDecision(
            action=DuckingAction.DUCK,
            gain_db=self._soft_duck_db,
            reason="PENDING_INTERIM_SPEECH",
            is_backchannel=False,
            is_barge_in=False,
        )
        self._last_decision = decision
        return decision
