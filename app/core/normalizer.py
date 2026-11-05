"""Cross-Script Linguistic Normalizer and Lexical Canonicalizer.

Provides high-throughput, low-latency cross-script text normalization across
Devanagari, Tamil, Telugu, Kannada, and Romanized/transliterated scripts.
Handles Unicode normalization, character elongation collapsing, punctuation
stripping, and canonical dictionary mapping to standard conversational tokens.
"""

from __future__ import annotations

import itertools
import logging
import re
import unicodedata
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("vartalaap.normalizer")

# Canonical dictionary mapping transliteration variants and native script tokens
# to unified conversational standard tokens.
CANONICAL_DICTIONARY: Dict[str, str] = {
    # Hindi (Romanized)
    "haan": "haan",
    "han": "haan",
    "ha": "haan",
    "haanji": "haanji",
    "haanjii": "haanji",
    "hanji": "haanji",
    "haan ji": "haanji",
    "han ji": "haanji",
    "achha": "achha",
    "accha": "achha",
    "achhaa": "achha",
    "acchaa": "achha",
    "acha": "achha",
    "theek hai": "theek hai",
    "thik hai": "theek hai",
    "theek": "theek",
    "thik": "theek",
    "sahi hai": "sahi hai",
    "sahi": "sahi",
    "bilkul": "bilkul",
    # Hindi (Devanagari)
    "हाँ": "haan",
    "हाँजी": "haanji",
    "हाँ जी": "haanji",
    "अच्छा": "achha",
    "ठीक है": "theek hai",
    "ठीक": "theek",
    "सही है": "sahi hai",
    "सही": "sahi",
    "बिल्कुल": "bilkul",

    # Tamil (Romanized & Native)
    "seri": "seri",
    "serii": "seri",
    "aamaa": "aama",
    "aama": "aama",
    "apdiya": "apdiya",
    "சரி": "seri",
    "ஆமா": "aama",
    "ஆமாம்": "aama",
    "அப்படியா": "apdiya",

    # Telugu (Romanized & Native)
    "sare": "sare",
    "saree": "sare",
    "avunu": "avunu",
    "ante": "ante",
    "సరే": "sare",
    "అవును": "avunu",
    "అంతే": "ante",

    # Kannada (Romanized & Native)
    "sari": "sari",
    "haudu": "haudu",
    "houdhu": "haudu",
    "howdu": "haudu",
    "ಸರಿ": "sari",
    "ಹೌದು": "haudu",

    # Universal & Colloquial English
    "hmm": "hmm",
    "hmmm": "hmm",
    "hm": "hmm",
    "okay": "okay",
    "ok": "okay",
    "yeah": "yeah",
    "yep": "yep",
    "yes": "yeah",
    "yup": "yep",
    "uh-huh": "hmm",
    "uh huh": "hmm",
    "umm": "hmm",
    "um": "hmm",
}

# Standardized set of canonical target tokens
STANDARD_BACKCHANNEL_TOKENS: FrozenSet[str] = frozenset({
    "haan",
    "haanji",
    "achha",
    "theek hai",
    "theek",
    "sahi hai",
    "sahi",
    "bilkul",
    "seri",
    "aama",
    "apdiya",
    "sare",
    "avunu",
    "ante",
    "sari",
    "haudu",
    "hmm",
    "okay",
    "yeah",
    "yep",
})

# Precompiled regex for character elongation (runs of 3 or more identical characters)
_ELONGATION_REGEX = re.compile(r"(.)\1{2,}", flags=re.IGNORECASE)


class CrossScriptNormalizer:
    """Normalizes conversational text across Indic and Romanized scripts.

    Performs Unicode NFKC normalization, strips punctuation across scripts,
    collapses character elongation using vocabulary-assisted heuristics, and
    maps phonetic variants to canonical tokens in sub-millisecond latency.
    """

    def __init__(
        self,
        canonical_dict: Optional[Dict[str, str]] = None,
        standard_tokens: Optional[Iterable[str]] = None,
    ) -> None:
        raw_dict = canonical_dict if canonical_dict is not None else CANONICAL_DICTIONARY
        # Clean and store canonical mapping (lowercased keys and normalized NFKC)
        self._canonical_dict: Dict[str, str] = {
            unicodedata.normalize("NFKC", k).strip().lower(): v
            for k, v in raw_dict.items()
        }

        if standard_tokens is not None:
            self._standard_tokens: FrozenSet[str] = frozenset(standard_tokens)
        else:
            self._standard_tokens = STANDARD_BACKCHANNEL_TOKENS

        # All recognized backchannel variants (keys and standard values)
        all_variants: Set[str] = set(self._canonical_dict.keys())
        all_variants.update(self._canonical_dict.values())
        all_variants.update(self._standard_tokens)
        self._all_variants: FrozenSet[str] = frozenset(all_variants)

    @property
    def canonical_dictionary(self) -> Dict[str, str]:
        """Dictionary of variant to canonical token mappings."""
        return self._canonical_dict

    @property
    def standard_tokens(self) -> FrozenSet[str]:
        """Set of target canonical tokens."""
        return self._standard_tokens

    @property
    def backchannel_variants(self) -> FrozenSet[str]:
        """All recognized backchannel tokens and variants across scripts."""
        return self._all_variants

    def strip_punctuation(self, text: str) -> str:
        """Strip punctuation marks across Unicode scripts while preserving letters and diacritics."""
        norm = unicodedata.normalize("NFKC", text)
        return "".join(c for c in norm if not unicodedata.category(c).startswith("P"))

    def _collapse_single_word(self, word: str) -> str:
        """Collapse character elongation for a single word.

        Tests length-2 and length-1 reductions against canonical vocabulary,
        favoring standard target tokens.
        """
        matches = list(_ELONGATION_REGEX.finditer(word))
        if not matches:
            return word

        combos = list(itertools.product([2, 1], repeat=len(matches)))
        candidates: List[Tuple[Tuple[int, ...], str]] = []

        for combo in combos:
            parts: List[str] = []
            last_idx = 0
            for match, rep_len in zip(matches, combo):
                parts.append(word[last_idx:match.start()])
                parts.append(word[match.start()] * rep_len)
                last_idx = match.end()
            parts.append(word[last_idx:])
            candidates.append((combo, "".join(parts)))

        # 1. Exact match to a standard token
        for combo, cand in candidates:
            cl = cand.lower()
            if cl in self._standard_tokens:
                return cand

        # 2. Match in canonical dictionary
        for combo, cand in candidates:
            cl = cand.lower()
            if cl in self._canonical_dict:
                return cand

        # 3. Default fallback: full reduction to single characters
        for combo, cand in candidates:
            if all(c == 1 for c in combo):
                return cand

        return candidates[-1][1]

    def collapse_elongation(self, text: str) -> str:
        """Collapse elongated character runs in a string or word sequence.

        Examples:
            'haaanji' -> 'haanji'
            'achhaaa' -> 'achha'
            'seriii' -> 'seri'
            'theeeek' -> 'theek'
        """
        if not text:
            return ""
        words = text.split()
        if not words:
            return ""
        return " ".join(self._collapse_single_word(w) for w in words)

    def canonicalize_token(self, token: str) -> str:
        """Map a token or phrase to its canonical form if known, else return as-is."""
        cleaned = token.strip().lower()
        return self._canonical_dict.get(cleaned, token)

    def normalize(self, text: str) -> str:
        """Fully normalize an utterance across scripts.

        Executes:
        1. Punctuation stripping and Unicode NFKC normalization.
        2. Whitespace compaction and Roman lowercasing.
        3. Whole-phrase canonical lookup.
        4. Token-level elongation collapsing and canonicalization.
        5. Recombined phrase canonical lookup.
        """
        if not text:
            return ""

        stripped = self.strip_punctuation(text)
        lowered = stripped.lower().strip()
        cleaned = " ".join(lowered.split())
        if not cleaned:
            return ""

        # Direct whole-phrase match
        if cleaned in self._canonical_dict:
            return self._canonical_dict[cleaned]

        words = cleaned.split()
        collapsed_words = [self._collapse_single_word(w) for w in words]
        mapped_words = [self._canonical_dict.get(w, w) for w in collapsed_words]
        recombined = " ".join(mapped_words)

        if recombined in self._canonical_dict:
            return self._canonical_dict[recombined]
        return recombined

    def is_backchannel(self, text: str) -> bool:
        """Determine whether text is a recognized backchannel token or phrase.

        Returns True for canonical tokens, transliteration variants, and
        multi-word phrases where all components are recognized backchannels.
        """
        if not text:
            return False

        normalized = self.normalize(text)
        if not normalized:
            return False

        # Direct match in variants or standard tokens
        if normalized in self._all_variants or normalized in self._standard_tokens:
            return True

        # Check raw stripped text in variants
        raw_clean = self.strip_punctuation(text).strip().lower()
        if raw_clean in self._all_variants:
            return True

        # Multi-word phrase where each token is a recognized backchannel
        tokens = normalized.split()
        if 1 <= len(tokens) <= 3 and all(t in self._all_variants for t in tokens):
            return True

        return False
