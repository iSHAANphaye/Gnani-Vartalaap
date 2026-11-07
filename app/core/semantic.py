"""Indic Postposition Parser and Hybrid Acoustic-Lexical Turn Classifier.

Combines high-throughput multilingual lexical analysis across Indic scripts
(Devanagari, Tamil, Telugu, Kannada, Hinglish, English) with deep acoustic
prosody classification from the GPU-accelerated Smart-Turn model.
Fuses acoustic incompletion probabilities with grammatical closure markers,
hanging conjunctions, cognitive fillers, and postpositions in sub-15ms latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union
import unicodedata

import numpy as np
import torch

from app.config import Settings, get_settings
from app.core.normalizer import CrossScriptNormalizer
from app.services.smart_turn import SmartTurnEngine, get_smart_turn_engine

logger = logging.getLogger("vartalaap.semantic")


class TurnCategory(str, Enum):
    """Categorical classification of turn-boundary semantic markers."""

    HANGING_CONJUNCTION = "HANGING_CONJUNCTION"
    COGNITIVE_HESITATION = "COGNITIVE_HESITATION"
    POSTPOSITION = "POSTPOSITION"
    TERMINAL_CLOSURE = "TERMINAL_CLOSURE"
    SHORT_CONFIRMATION = "SHORT_CONFIRMATION"
    NEUTRAL = "NEUTRAL"


@dataclass
class SemanticTurnEvaluation:
    """Consolidated multimodal evaluation of conversational turn completeness."""

    p_acoustic: float
    lexical_penalty: float
    p_incomplete: float
    p_complete: float
    category: TurnCategory
    matched_markers: List[str]
    latency_ms: float


@dataclass(frozen=True)
class LexicalParseResult:
    """Internal structural result emitted by the Indic lexical parser."""

    category: TurnCategory
    penalty: float
    matched_markers: List[str]


# Default semantic penalty weights aligned with conversational Indic grammar
DEFAULT_CATEGORY_PENALTIES: Dict[TurnCategory, float] = {
    TurnCategory.POSTPOSITION: 0.85,
    TurnCategory.HANGING_CONJUNCTION: 0.80,
    TurnCategory.COGNITIVE_HESITATION: 0.95,
    TurnCategory.TERMINAL_CLOSURE: -0.65,
    TurnCategory.SHORT_CONFIRMATION: -0.75,
    TurnCategory.NEUTRAL: 0.0,
}

# ==============================================================================
# Grammatical Marker Catalogs Across Scripts (Hinglish, Devanagari, Tamil, Telugu, Kannada, English)
# ==============================================================================

# 1. Postpositions / Case markers (+0.85 penalty)
POSTPOSITION_MARKERS: FrozenSet[str] = frozenset({
    # Hindi / Hinglish (Romanized)
    "ke", "ka", "ki", "mein", "se", "ko", "par", "pe", "liye", "ke liye", "keliye", "vaaste",
    # Hindi (Devanagari)
    "के", "का", "की", "में", "से", "को", "पर", "पे", "लिए", "के लिए", "केलिए", "वास्ते",
    # Tamil (Romanized & Native)
    "ku", "la", "oda", "ukku", "la irundhu", "lirundhu", "il", "aal", "odu", "idam", "pathi", "paththi",
    "க்கு", "ல", "உக்கு", "ஓட", "லிருந்து", "இல்", "ஆல்", "கு", "இடம்", "பற்றி",
    # Telugu (Romanized & Native)
    "ki", "ku", "lo", "tho", "kosam", "koraku", "nunchi", "valla", "lo nunchi",
    "కి", "కు", "లో", "తో", "కోసం", "కొరకు", "నుంచి", "వల్ల",
    # Kannada (Romanized & Native)
    "alli", "ge", "inda", "ige", "annu", "gosuga", "keli",
    "ಅಲ್ಲಿ", "ಗೆ", "ಇಂದ", "ಇಗೆ", "ಅನ್ನು", "ಗೋಸ್ಕರ",
})

# 2. Hanging Conjunctions (+0.80 penalty)
HANGING_CONJUNCTION_MARKERS: FrozenSet[str] = frozenset({
    # Hindi / Hinglish (Romanized)
    "aur", "lekin", "magar", "kyunki", "ya", "parantu", "kintu", "warna", "aur phir", "tatha", "evam",
    # Hindi (Devanagari)
    "और", "लेकिन", "मगर", "क्योंकि", "या", "परन्तु", "किन्तु", "वरना", "और फिर", "तथा", "एवं",
    # English
    "and", "but", "so", "because", "or", "and then",
    # Tamil (Romanized & Native)
    "aana", "analum", "aanaal", "illana", "illenral", "illati", "matrum", "illa", "aadhunaala", "adhanaala", "adhunaala",
    "ஆனா", "ஆனால்", "இல்லனா", "மற்றும்", "ஆதுனால",
    # Telugu (Romanized & Native)
    "kaani", "mariyu", "leka", "endukante", "kaakapothe",
    "కానీ", "మరియు", "లేక", "ఎందుకంటే",
    # Kannada (Romanized & Native)
    "aadhare", "mattu", "athava", "yaakandre",
    "ಆದರೆ", "ಮತ್ತು", "ಅಥವಾ", "ಯಾಕಂದರೆ",
})

# 3. Cognitive Fillers / Hesitation (+0.95 penalty)
COGNITIVE_HESITATION_MARKERS: FrozenSet[str] = frozenset({
    # Hindi / Hinglish (Romanized)
    "matlab", "matlab ki", "woh", "woh kya hai na", "kya kehte hain", "kya bolte hain",
    "kya bolte", "are", "aree", "arrey", "arre", "yaani", "yaani ki",
    # Hindi (Devanagari)
    "मतलब", "मतलब की", "वो", "वो क्या है ना", "अरे", "यानी",
    # English
    "like", "you know", "um", "uh", "umm", "uhh", "er", "ah", "i mean", "actually", "basically",
    # Tamil (Romanized & Native)
    "seri na", "serina", "vandhu", "appo", "enna na", "enna solradhu",
    "வந்து", "சரி நா", "என்னன்னா",
    # Telugu (Romanized & Native)
    "ante", "antey", "adigoo", "choodu", "adi",
    "అంతే", "అది", "అంటే",
    # Kannada (Romanized & Native)
    "andre", "adhu", "ivaga",
    "ಅಂದ್ರೆ",
})

# 4. Terminal Closures (-0.65 reward toward completion)
TERMINAL_CLOSURE_MARKERS: FrozenSet[str] = frozenset({
    # Hindi / Hinglish (Romanized)
    "karo", "kijiye", "kijiyega", "karna", "kariye", "hai", "hain", "tha", "thi", "the",
    "hoga", "hogi", "honge", "chahiye", "gaya", "gayi", "gaye", "diya", "diye", "di",
    "liya", "liye", "li", "raha hai", "rahi hai", "rahe hain", "diya hai", "gaya hai",
    "kardiya", "kar diya", "ho gaya", "ho gayi", "bhar diya hai", "kijiye na", "boliye",
    # Hindi (Devanagari)
    "करो", "कीजिए", "कीजिये", "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे",
    "चाहिए", "गया", "गयी", "गए", "दिया", "दिए", "लिया", "लिए", "रहा है", "रही है", "रहे हैं",
    "दिया है", "गया है", "हो गया", "कर दिया", "बोलिए",
    # Tamil (Romanized & Native)
    "pannunga", "pannu", "irukku", "aachu", "venum", "mudinjidhu", "kudunga", "sollunga", "irukkum",
    "பண்ணுங்க", "இருக்கு", "ஆச்சு", "வேணும்", "முடிஞ்சிது", "குடுங்க", "சொல்லுங்க",
    # Telugu (Romanized & Native)
    "cheyandi", "cheyyi", "undi", "ayyindi", "kavali", "ivvandi", "cheppandi", "untundi",
    "చేయండి", "ఉంది", "అయ్యింది", "కావాలి", "ఇవ్వండి", "చెప్పండి",
    # Kannada (Romanized & Native)
    "maadi", "ide", "aayithu", "beku", "kodi", "heli",
    "ಮಾಡಿ", "ಇದೆ", "ಆಯಿತು", "ಬೇಕು",
    # English
    "please", "thanks", "thank you", "is done", "completed", "now",
})

# 5. Short Confirmations (-0.75 reward toward completion)
SHORT_CONFIRMATION_PHRASES: FrozenSet[str] = frozenset({
    # Hindi (Romanized & Devanagari)
    "haan", "han", "ha", "haanji", "hanji", "theek hai", "thik hai", "theek", "thik",
    "sahi hai", "sahi", "bilkul", "achha", "accha", "nahi", "nahin", "na",
    "हाँ", "हाँजी", "ठीक है", "ठीक", "सही है", "सही", "बिल्कुल", "अच्छा", "नहीं", "ना",
    # English
    "yes", "yeah", "yep", "yup", "no", "nope", "okay", "ok", "done", "confirm", "confirmed",
    "correct", "right", "sure", "fine", "got it", "all right", "alright",
    # Tamil (Romanized & Native)
    "seri", "aama", "aamaa", "apdiya", "illa", "illai",
    "சரி", "ஆமா", "ஆமாம்", "அப்படியா", "இல்லை",
    # Telugu (Romanized & Native)
    "sare", "avunu", "kaadu", "ledu",
    "సరే", "అవును", "కాదు", "లేదు",
    # Kannada (Romanized & Native)
    "sari", "haudu", "houdhu", "howdu", "illa",
    "ಸರಿ", "ಹೌದು", "ಇಲ್ಲ",
})

# Single-word constituent vocabulary for multi-token confirmation utterances
CONFIRMATION_CONSTITUENT_WORDS: FrozenSet[str] = frozenset({
    "haan", "han", "ha", "ji", "haanji", "hanji", "theek", "thik", "hai", "sahi",
    "bilkul", "achha", "accha", "nahi", "nahin", "na", "yes", "yeah", "yep", "yup",
    "no", "nope", "okay", "ok", "done", "confirm", "sure", "fine", "seri", "aama",
    "sare", "avunu", "sari", "haudu", "हाँ", "नहीं", "हाँजी", "ठीक", "है", "सही",
    "अच्छा", "बिल्कुल", "சரி", "ஆமா", "ஆமாம்", "சரே", "సరే", "అవును", "ಸರಿ", "ಹೌದು",
})

# Agglutinative case suffixes in native Dravidian orthography
DRAVIDIAN_AGGLUTINATIVE_SUFFIXES: Tuple[Tuple[str, TurnCategory], ...] = (
    # Tamil postposition suffixes
    ("லிருந்து", TurnCategory.POSTPOSITION),
    ("உக்கு", TurnCategory.POSTPOSITION),
    ("க்கு", TurnCategory.POSTPOSITION),
    ("இடம்", TurnCategory.POSTPOSITION),
    ("இல்", TurnCategory.POSTPOSITION),
    ("ஆல்", TurnCategory.POSTPOSITION),
    ("ஓட", TurnCategory.POSTPOSITION),
    ("கு", TurnCategory.POSTPOSITION),
    ("ல", TurnCategory.POSTPOSITION),
    # Telugu postposition suffixes
    ("కొరకు", TurnCategory.POSTPOSITION),
    ("కోసం", TurnCategory.POSTPOSITION),
    ("నుంచి", TurnCategory.POSTPOSITION),
    ("వల్ల", TurnCategory.POSTPOSITION),
    ("లో", TurnCategory.POSTPOSITION),
    ("తో", TurnCategory.POSTPOSITION),
    ("కి", TurnCategory.POSTPOSITION),
    ("కు", TurnCategory.POSTPOSITION),
    # Kannada postposition suffixes
    ("ಗೋಸ್ಕರ", TurnCategory.POSTPOSITION),
    ("ಅಲ್ಲಿ", TurnCategory.POSTPOSITION),
    ("ಇಂದ", TurnCategory.POSTPOSITION),
    ("ಅನ್ನು", TurnCategory.POSTPOSITION),
    ("ಿಗೆ", TurnCategory.POSTPOSITION),
    ("ಗೆ", TurnCategory.POSTPOSITION),
    # Tamil / Telugu native verb closures
    ("முடிஞ்சிது", TurnCategory.TERMINAL_CLOSURE),
    ("பண்ணுங்க", TurnCategory.TERMINAL_CLOSURE),
    ("செய்யండి", TurnCategory.TERMINAL_CLOSURE),
    ("చేయండి", TurnCategory.TERMINAL_CLOSURE),
    ("అయ్యింది", TurnCategory.TERMINAL_CLOSURE),
)


class IndicLexicalParser:
    """High-throughput multilingual parser for Indic conversational boundaries.

    Tokenizes, normalizes, and inspects trailing phrases and grammatical markers
    across Hindi, Hinglish, Tamil, Telugu, Kannada, and English. Emits categorical
    classifications and lexical penalty scores in sub-millisecond execution time.
    """

    def __init__(
        self,
        normalizer: Optional[CrossScriptNormalizer] = None,
        penalties: Optional[Dict[TurnCategory, float]] = None,
    ) -> None:
        self.normalizer = normalizer or CrossScriptNormalizer()
        self.penalties: Dict[TurnCategory, float] = (
            dict(penalties) if penalties is not None else dict(DEFAULT_CATEGORY_PENALTIES)
        )

    def parse(self, transcript: str) -> LexicalParseResult:
        """Parse transcript text and classify turn-boundary status.

        Args:
            transcript: Interim or finalized ASR transcript text.

        Returns:
            LexicalParseResult containing categorical classification, penalty,
            and list of matched grammatical markers.
        """
        if not transcript or not transcript.strip():
            return LexicalParseResult(
                category=TurnCategory.NEUTRAL,
                penalty=self.penalties[TurnCategory.NEUTRAL],
                matched_markers=[],
            )

        # Normalize orthography: separate hyphens/slashes to preserve attached clitics
        spaced_text = transcript.replace("-", " ").replace("/", " ")
        cleaned = self.normalizer.strip_punctuation(spaced_text)
        lowered = cleaned.lower().strip()
        tokens = [self.normalizer.collapse_elongation(tok) for tok in lowered.split()]

        if not tokens:
            return LexicalParseResult(
                category=TurnCategory.NEUTRAL,
                penalty=self.penalties[TurnCategory.NEUTRAL],
                matched_markers=[],
            )

        joined_normalized = " ".join(tokens)

        # 1. Short Confirmation Evaluation
        # If the utterance is short (<= 3 words) and consists exclusively of confirmation tokens
        if len(tokens) <= 3:
            if (
                joined_normalized in SHORT_CONFIRMATION_PHRASES
                or all(t in CONFIRMATION_CONSTITUENT_WORDS for t in tokens)
            ):
                return LexicalParseResult(
                    category=TurnCategory.SHORT_CONFIRMATION,
                    penalty=self.penalties[TurnCategory.SHORT_CONFIRMATION],
                    matched_markers=[joined_normalized],
                )

        # 2. Multi-word and Trailing Window Marker Evaluation (Windows 4 down to 1)
        max_window = min(4, len(tokens))
        for window_size in range(max_window, 0, -1):
            phrase = " ".join(tokens[-window_size:])

            # High priority: Cognitive Hesitation Fillers (+0.95)
            if phrase in COGNITIVE_HESITATION_MARKERS:
                return LexicalParseResult(
                    category=TurnCategory.COGNITIVE_HESITATION,
                    penalty=self.penalties[TurnCategory.COGNITIVE_HESITATION],
                    matched_markers=[phrase],
                )

            # High priority: Hanging Conjunctions (+0.80)
            if phrase in HANGING_CONJUNCTION_MARKERS:
                return LexicalParseResult(
                    category=TurnCategory.HANGING_CONJUNCTION,
                    penalty=self.penalties[TurnCategory.HANGING_CONJUNCTION],
                    matched_markers=[phrase],
                )

            # High priority: Postpositions and Case Markers (+0.85)
            if phrase in POSTPOSITION_MARKERS:
                return LexicalParseResult(
                    category=TurnCategory.POSTPOSITION,
                    penalty=self.penalties[TurnCategory.POSTPOSITION],
                    matched_markers=[phrase],
                )

            # Completion Reward: Terminal Verb Closures (-0.65)
            if phrase in TERMINAL_CLOSURE_MARKERS:
                return LexicalParseResult(
                    category=TurnCategory.TERMINAL_CLOSURE,
                    penalty=self.penalties[TurnCategory.TERMINAL_CLOSURE],
                    matched_markers=[phrase],
                )

            # Completion Reward: Short Confirmations (-0.75)
            if phrase in SHORT_CONFIRMATION_PHRASES:
                return LexicalParseResult(
                    category=TurnCategory.SHORT_CONFIRMATION,
                    penalty=self.penalties[TurnCategory.SHORT_CONFIRMATION],
                    matched_markers=[phrase],
                )

        # 3. Agglutinative Suffix Inspection for Dravidian Native Scripts
        last_token = tokens[-1]
        for suffix, cat in DRAVIDIAN_AGGLUTINATIVE_SUFFIXES:
            if len(last_token) > len(suffix) and last_token.endswith(suffix):
                return LexicalParseResult(
                    category=cat,
                    penalty=self.penalties[cat],
                    matched_markers=[suffix],
                )

        # 4. Neutral / Inconclusive Default
        return LexicalParseResult(
            category=TurnCategory.NEUTRAL,
            penalty=self.penalties[TurnCategory.NEUTRAL],
            matched_markers=[],
        )

    def get_penalty(self, transcript: str) -> float:
        """Extract lexical penalty score for a transcript string."""
        return self.parse(transcript).penalty

    def get_category(self, transcript: str) -> TurnCategory:
        """Extract matched semantic turn category for a transcript string."""
        return self.parse(transcript).category


class HybridTurnClassifier:
    """Multimodal turn-completion classifier fusing acoustic prosody and Indic semantics.

    Executes GPU-accelerated Smart-Turn acoustic feature inference on raw PCM audio,
    evaluates trailing linguistic boundary markers via IndicLexicalParser, and fuses
    the signals into calibrated completion probabilities within sub-15ms latency budget.
    """

    def __init__(
        self,
        smart_turn: Optional[SmartTurnEngine] = None,
        lexical_parser: Optional[IndicLexicalParser] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.smart_turn = smart_turn or get_smart_turn_engine(settings=self.settings)
        self.lexical_parser = lexical_parser or IndicLexicalParser()
        logger.info(
            "HybridTurnClassifier initialized (GPU acceleration: %s, Device: %s)",
            self.smart_turn.is_gpu,
            self.smart_turn.device,
        )

    @property
    def smart_turn_engine(self) -> SmartTurnEngine:
        """Access underlying acoustic SmartTurnEngine instance."""
        return self.smart_turn

    @property
    def lexical_parser_instance(self) -> IndicLexicalParser:
        """Access underlying IndicLexicalParser instance."""
        return self.lexical_parser

    def evaluate(
        self,
        audio_pcm: Optional[Union[np.ndarray, torch.Tensor]],
        transcript: str,
    ) -> SemanticTurnEvaluation:
        """Evaluate multimodal turn completeness over aligned audio and partial transcript.

        Args:
            audio_pcm: Raw 16kHz mono audio waveform samples (or None if unvoiced).
            transcript: Interim or finalized ASR transcript text.

        Returns:
            SemanticTurnEvaluation with acoustic probability, lexical penalty,
            clamped incomplete/complete probabilities, category, and latency.
        """
        t0 = time.perf_counter()

        # 1. Acoustic Prosodic Inference via Smart-Turn
        if audio_pcm is not None and len(audio_pcm) > 0:
            p_acoustic = self.smart_turn.predict_incomplete(audio_pcm)
        else:
            p_acoustic = 0.5  # Unbiased prior when audio frame is absent

        # 2. Indic Lexical Analysis
        lexical_result = self.lexical_parser.parse(transcript)
        lexical_penalty = lexical_result.penalty

        # 3. Decision Fusion & Probability Clamping
        # P(incomplete) = clamp(P_acoustic + S_lexical_penalty, 0.0, 1.0)
        p_incomplete = max(0.0, min(1.0, p_acoustic + lexical_penalty))
        p_complete = 1.0 - p_incomplete

        # Ensure GPU operations are completed for accurate end-to-end timing
        if self.smart_turn.is_gpu:
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SemanticTurnEvaluation(
            p_acoustic=float(p_acoustic),
            lexical_penalty=float(lexical_penalty),
            p_incomplete=float(p_incomplete),
            p_complete=float(p_complete),
            category=lexical_result.category,
            matched_markers=lexical_result.matched_markers,
            latency_ms=float(latency_ms),
        )
