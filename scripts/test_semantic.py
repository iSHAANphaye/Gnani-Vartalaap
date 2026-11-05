"""Indic Postposition Parser and Hybrid Acoustic-Lexical Classifier Verification Suite.

Comprehensive functional validation and sub-15ms latency benchmarking:
1. Conjunction penalty: 'Mera electricity bill aur' -> HANGING_CONJUNCTION, P(incomplete) > 0.8.
2. Hesitation penalty: 'Mera account balance matlab ki' -> COGNITIVE_HESITATION, P(incomplete) > 0.85.
3. Postposition penalty: 'Savings account ka' -> POSTPOSITION, P(incomplete) > 0.75.
4. Terminal closure: 'Mera electricity bill bhar diya hai' -> TERMINAL_CLOSURE, P(incomplete) < 0.35, P(complete) > 0.65.
5. Short confirmation: 'Theek hai' -> SHORT_CONFIRMATION, P(complete) > 0.75.
6. Multi-script test: Native Devanagari, Tamil, Telugu, and Kannada sentences.
7. Latency benchmark across 50 evaluations on RTX 5060 (< 15ms combined latency).
8. Adversarial edge cases: empty string, pure punctuation, noise audio, None/empty inputs.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.semantic import (
    HybridTurnClassifier,
    IndicLexicalParser,
    SemanticTurnEvaluation,
    TurnCategory,
)


def create_silence_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate silent audio buffer for baseline conversational pause testing."""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


def create_noise_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate low-amplitude white noise buffer."""
    samples = int(duration_sec * sample_rate)
    return np.random.uniform(-0.05, 0.05, samples).astype(np.float32)


def test_conjunction_penalty(classifier: HybridTurnClassifier) -> bool:
    """Verify hanging conjunction penalty (+0.80) holds conversational turn open."""
    print("\n" + "=" * 70)
    print(" [TEST 1] Hanging Conjunction Penalty Verification")
    print("=" * 70)

    audio = create_silence_audio(0.5)
    transcript = "Mera electricity bill aur"

    eval_res = classifier.evaluate(audio, transcript)
    print(f"  Transcript        : '{transcript}'")
    print(f"  Category          : {eval_res.category.value}")
    print(f"  Matched Markers   : {eval_res.matched_markers}")
    print(f"  P(acoustic)       : {eval_res.p_acoustic:.4f}")
    print(f"  Lexical Penalty   : {eval_res.lexical_penalty:+.2f}")
    print(f"  P(incomplete)     : {eval_res.p_incomplete:.4f}")
    print(f"  P(complete)       : {eval_res.p_complete:.4f}")
    print(f"  Latency           : {eval_res.latency_ms:.2f} ms")

    assert eval_res.category == TurnCategory.HANGING_CONJUNCTION, (
        f"Expected HANGING_CONJUNCTION, got {eval_res.category}"
    )
    assert eval_res.p_incomplete > 0.8, (
        f"Expected P(incomplete) > 0.8, got {eval_res.p_incomplete:.4f}"
    )
    assert "aur" in eval_res.matched_markers
    print("  ✓ PASS: Hanging conjunction successfully penalized and classified.")
    return True


def test_hesitation_penalty(classifier: HybridTurnClassifier) -> bool:
    """Verify cognitive hesitation filler penalty (+0.95) blocks premature turn cut-off."""
    print("\n" + "=" * 70)
    print(" [TEST 2] Cognitive Hesitation Penalty Verification")
    print("=" * 70)

    audio = create_silence_audio(0.5)
    transcript = "Mera account balance matlab ki"

    eval_res = classifier.evaluate(audio, transcript)
    print(f"  Transcript        : '{transcript}'")
    print(f"  Category          : {eval_res.category.value}")
    print(f"  Matched Markers   : {eval_res.matched_markers}")
    print(f"  P(acoustic)       : {eval_res.p_acoustic:.4f}")
    print(f"  Lexical Penalty   : {eval_res.lexical_penalty:+.2f}")
    print(f"  P(incomplete)     : {eval_res.p_incomplete:.4f}")
    print(f"  P(complete)       : {eval_res.p_complete:.4f}")
    print(f"  Latency           : {eval_res.latency_ms:.2f} ms")

    assert eval_res.category == TurnCategory.COGNITIVE_HESITATION, (
        f"Expected COGNITIVE_HESITATION, got {eval_res.category}"
    )
    assert eval_res.p_incomplete > 0.85, (
        f"Expected P(incomplete) > 0.85, got {eval_res.p_incomplete:.4f}"
    )
    assert "matlab ki" in eval_res.matched_markers
    print("  ✓ PASS: Cognitive hesitation successfully identified and floor preserved.")
    return True


def test_postposition_penalty(classifier: HybridTurnClassifier) -> bool:
    """Verify Indic postposition penalty (+0.85) holds floor open after grammatical case marker."""
    print("\n" + "=" * 70)
    print(" [TEST 3] Indic Postposition Penalty Verification")
    print("=" * 70)

    audio = create_silence_audio(0.5)
    transcript = "Savings account ka"

    eval_res = classifier.evaluate(audio, transcript)
    print(f"  Transcript        : '{transcript}'")
    print(f"  Category          : {eval_res.category.value}")
    print(f"  Matched Markers   : {eval_res.matched_markers}")
    print(f"  P(acoustic)       : {eval_res.p_acoustic:.4f}")
    print(f"  Lexical Penalty   : {eval_res.lexical_penalty:+.2f}")
    print(f"  P(incomplete)     : {eval_res.p_incomplete:.4f}")
    print(f"  P(complete)       : {eval_res.p_complete:.4f}")
    print(f"  Latency           : {eval_res.latency_ms:.2f} ms")

    assert eval_res.category == TurnCategory.POSTPOSITION, (
        f"Expected POSTPOSITION, got {eval_res.category}"
    )
    assert eval_res.p_incomplete > 0.75, (
        f"Expected P(incomplete) > 0.75, got {eval_res.p_incomplete:.4f}"
    )
    assert "ka" in eval_res.matched_markers
    print("  ✓ PASS: Postposition case marker detected and turn held open.")
    return True


def test_terminal_closure(classifier: HybridTurnClassifier) -> bool:
    """Verify terminal verb closure rewards turn completion (-0.65)."""
    print("\n" + "=" * 70)
    print(" [TEST 4] Terminal Verb Closure Verification")
    print("=" * 70)

    audio = create_silence_audio(0.5)
    transcript = "Mera electricity bill bhar diya hai"

    eval_res = classifier.evaluate(audio, transcript)
    print(f"  Transcript        : '{transcript}'")
    print(f"  Category          : {eval_res.category.value}")
    print(f"  Matched Markers   : {eval_res.matched_markers}")
    print(f"  P(acoustic)       : {eval_res.p_acoustic:.4f}")
    print(f"  Lexical Penalty   : {eval_res.lexical_penalty:+.2f}")
    print(f"  P(incomplete)     : {eval_res.p_incomplete:.4f}")
    print(f"  P(complete)       : {eval_res.p_complete:.4f}")
    print(f"  Latency           : {eval_res.latency_ms:.2f} ms")

    assert eval_res.category == TurnCategory.TERMINAL_CLOSURE, (
        f"Expected TERMINAL_CLOSURE, got {eval_res.category}"
    )
    assert eval_res.p_incomplete < 0.35, (
        f"Expected P(incomplete) < 0.35, got {eval_res.p_incomplete:.4f}"
    )
    assert eval_res.p_complete > 0.65, (
        f"Expected P(complete) > 0.65, got {eval_res.p_complete:.4f}"
    )
    print("  ✓ PASS: Terminal closure rewarded toward prompt turn completion.")
    return True


def test_short_confirmation(classifier: HybridTurnClassifier) -> bool:
    """Verify short confirmation response ('Theek hai') rapidly signals turn completion."""
    print("\n" + "=" * 70)
    print(" [TEST 5] Short Confirmation Verification")
    print("=" * 70)

    audio = create_silence_audio(0.5)
    transcript = "Theek hai"

    eval_res = classifier.evaluate(audio, transcript)
    print(f"  Transcript        : '{transcript}'")
    print(f"  Category          : {eval_res.category.value}")
    print(f"  Matched Markers   : {eval_res.matched_markers}")
    print(f"  P(acoustic)       : {eval_res.p_acoustic:.4f}")
    print(f"  Lexical Penalty   : {eval_res.lexical_penalty:+.2f}")
    print(f"  P(incomplete)     : {eval_res.p_incomplete:.4f}")
    print(f"  P(complete)       : {eval_res.p_complete:.4f}")
    print(f"  Latency           : {eval_res.latency_ms:.2f} ms")

    assert eval_res.category == TurnCategory.SHORT_CONFIRMATION, (
        f"Expected SHORT_CONFIRMATION, got {eval_res.category}"
    )
    assert eval_res.p_complete > 0.75, (
        f"Expected P(complete) > 0.75, got {eval_res.p_complete:.4f}"
    )
    print("  ✓ PASS: Short confirmation classified and completion strongly accelerated.")
    return True


def test_multiscript_parsing(classifier: HybridTurnClassifier) -> bool:
    """Verify native Devanagari, Tamil, Telugu, and Kannada parsing."""
    print("\n" + "=" * 70)
    print(" [TEST 6] Multi-Script Cross-Lingual Parsing Verification")
    print("=" * 70)

    audio = create_silence_audio(0.5)

    cases: List[Tuple[str, TurnCategory, str]] = [
        # Devanagari (Hindi)
        ("मेरा बिजली का बिल और", TurnCategory.HANGING_CONJUNCTION, "Devanagari hanging conjunction"),
        ("मेरा अकाउंट बैलेंस मतलब की", TurnCategory.COGNITIVE_HESITATION, "Devanagari hesitation phrase"),
        ("सेविंग्स अकाउंट का", TurnCategory.POSTPOSITION, "Devanagari postposition case marker"),
        ("मेरा बिजली का बिल भर दिया है", TurnCategory.TERMINAL_CLOSURE, "Devanagari terminal verb closure"),
        ("हाँ", TurnCategory.SHORT_CONFIRMATION, "Devanagari affirmative token"),
        ("ठीक है", TurnCategory.SHORT_CONFIRMATION, "Devanagari confirmation phrase"),
        # Tamil (Native & Romanized)
        ("கரண்ட் பில் மற்றும்", TurnCategory.HANGING_CONJUNCTION, "Tamil conjunction marker"),
        ("En current bill aana", TurnCategory.HANGING_CONJUNCTION, "Tamil Romanized conjunction"),
        ("அக்கவுண்ட் ல", TurnCategory.POSTPOSITION, "Tamil postposition locative"),
        ("சென்னைக்கு", TurnCategory.POSTPOSITION, "Tamil agglutinative dative suffix"),
        ("வேலை முடிஞ்சிது", TurnCategory.TERMINAL_CLOSURE, "Tamil terminal verb closure"),
        ("சரி", TurnCategory.SHORT_CONFIRMATION, "Tamil confirmation token"),
        # Telugu (Native & Romanized)
        ("కరెంట్ బిల్ మరియు", TurnCategory.HANGING_CONJUNCTION, "Telugu conjunction marker"),
        ("కరెంట్ బిల్ కానీ", TurnCategory.HANGING_CONJUNCTION, "Telugu adversative conjunction"),
        ("అకౌంట్ లో", TurnCategory.POSTPOSITION, "Telugu postposition locative"),
        ("ఖాతాకి", TurnCategory.POSTPOSITION, "Telugu agglutinative dative suffix"),
        ("ఈ పని చేయండి", TurnCategory.TERMINAL_CLOSURE, "Telugu imperative verb closure"),
        ("సరే", TurnCategory.SHORT_CONFIRMATION, "Telugu confirmation token"),
        ("అవును", TurnCategory.SHORT_CONFIRMATION, "Telugu affirmative token"),
        # Kannada (Native & Romanized)
        ("ಕರೆಂಟ್ ಬಿಲ್ ಮತ್ತು", TurnCategory.HANGING_CONJUNCTION, "Kannada conjunction marker"),
        ("ಖಾತೆಗೆ", TurnCategory.POSTPOSITION, "Kannada agglutinative dative suffix"),
        ("ಕೆಲಸ ಮಾಡಿ", TurnCategory.TERMINAL_CLOSURE, "Kannada imperative verb closure"),
        ("ಸರಿ", TurnCategory.SHORT_CONFIRMATION, "Kannada confirmation token"),
        ("ಹೌದು", TurnCategory.SHORT_CONFIRMATION, "Kannada affirmative token"),
    ]

    for transcript, expected_category, description in cases:
        eval_res = classifier.evaluate(audio, transcript)
        assert eval_res.category == expected_category, (
            f"Failed on '{transcript}' ({description}): expected {expected_category.value}, "
            f"got {eval_res.category.value} (markers: {eval_res.matched_markers})"
        )
        print(f"  ✓ [{eval_res.category.value:<20}] '{transcript}' -> markers: {eval_res.matched_markers}")

    print(f"\n  ✓ PASS: All {len(cases)} multi-script test cases parsed accurately.")
    return True


def test_latency_benchmark(classifier: HybridTurnClassifier) -> bool:
    """Benchmark multimodal classifier latency across 50 evaluations (< 15ms target on RTX 5060)."""
    print("\n" + "=" * 70)
    print(" [TEST 7] Hybrid Classifier Latency Benchmark (50 Runs on RTX 5060)")
    print("=" * 70)

    audio = create_silence_audio(1.0)
    transcript = "Mera electricity bill bhar diya hai"

    # Prime CUDA kernels and warmup
    for _ in range(10):
        classifier.evaluate(audio, transcript)
    if classifier.smart_turn.is_gpu:
        torch.cuda.synchronize()

    iterations = 50
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        eval_res = classifier.evaluate(audio, transcript)
        latencies.append(eval_res.latency_ms)

    avg_ms = float(np.mean(latencies))
    min_ms = float(np.min(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))

    print(f"  ✓ Iterations      : {iterations}")
    print(f"  ✓ Device          : {classifier.smart_turn.device}")
    print(f"  ✓ Mean Latency    : {avg_ms:.3f} ms")
    print(f"  ✓ Median Latency  : {median_ms:.3f} ms")
    print(f"  ✓ Min Latency     : {min_ms:.3f} ms")
    print(f"  ✓ P95 Latency     : {p95_ms:.3f} ms")

    assert avg_ms < 15.0, f"Average latency {avg_ms:.2f} ms exceeds 15.0 ms target requirement!"
    print(f"  ✓ PASS: Combined latency ({avg_ms:.3f} ms) strictly satisfies < 15ms budget!")
    return True


def test_adversarial_edge_cases(classifier: HybridTurnClassifier) -> bool:
    """Verify stability under adversarial inputs: empty string, pure punctuation, noise audio."""
    print("\n" + "=" * 70)
    print(" [TEST 8] Adversarial Edge Cases & Robustness Verification")
    print("=" * 70)

    audio_silence = create_silence_audio(0.5)
    audio_noise = create_noise_audio(0.5)
    audio_empty = np.array([], dtype=np.float32)

    # 1. Empty string
    res_empty = classifier.evaluate(audio_silence, "")
    assert res_empty.category == TurnCategory.NEUTRAL
    assert res_empty.lexical_penalty == 0.0
    assert res_empty.matched_markers == []
    print("  ✓ Empty transcript -> NEUTRAL, penalty 0.0")

    # 2. Pure whitespace
    res_ws = classifier.evaluate(audio_silence, "    \t\n  ")
    assert res_ws.category == TurnCategory.NEUTRAL
    assert res_ws.lexical_penalty == 0.0
    print("  ✓ Whitespace transcript -> NEUTRAL, penalty 0.0")

    # 3. Pure punctuation
    res_punct = classifier.evaluate(audio_silence, "???... !!! ,,, ---")
    assert res_punct.category == TurnCategory.NEUTRAL
    assert res_punct.lexical_penalty == 0.0
    print("  ✓ Pure punctuation -> NEUTRAL, penalty 0.0")

    # 4. Pure noise audio with neutral transcript
    res_noise = classifier.evaluate(audio_noise, "Mera naam Rahul hai")
    assert 0.0 <= res_noise.p_incomplete <= 1.0
    assert 0.0 <= res_noise.p_complete <= 1.0
    assert abs(res_noise.p_incomplete + res_noise.p_complete - 1.0) < 1e-6
    print(f"  ✓ Noise audio + Neutral utterance -> P(complete) = {res_noise.p_complete:.4f}")

    # 5. Empty audio array
    res_no_audio = classifier.evaluate(audio_empty, "Theek hai")
    assert res_no_audio.category == TurnCategory.SHORT_CONFIRMATION
    assert res_no_audio.p_complete > 0.75
    print("  ✓ Zero-length audio array -> Handled gracefully with neutral prior")

    # 6. None audio input
    res_none_audio = classifier.evaluate(None, "Mera electricity bill aur")
    assert res_none_audio.category == TurnCategory.HANGING_CONJUNCTION
    assert res_none_audio.p_incomplete > 0.8
    print("  ✓ None audio -> Handled gracefully with neutral prior")

    # 7. Extreme audio amplitude clipping
    audio_clipped = np.ones(8000, dtype=np.float32) * 10.0
    res_clipped = classifier.evaluate(audio_clipped, "Theek hai")
    assert 0.0 <= res_clipped.p_incomplete <= 1.0
    assert 0.0 <= res_clipped.p_complete <= 1.0
    print("  ✓ Clipped audio waveform -> Normalized cleanly without overflow")

    print("  ✓ PASS: All adversarial edge cases handled safely.")
    return True


def main() -> None:
    """Run full Indic semantic parser and hybrid classifier test suite."""
    print("\n" + "#" * 70)
    print(" PROJECT VARTALAAP - INDIC SEMANTIC PARSER & HYBRID CLASSIFIER SUITE")
    print("#" * 70)

    total_start = time.perf_counter()

    classifier = HybridTurnClassifier()

    test_conjunction_penalty(classifier)
    test_hesitation_penalty(classifier)
    test_postposition_penalty(classifier)
    test_terminal_closure(classifier)
    test_short_confirmation(classifier)
    test_multiscript_parsing(classifier)
    test_latency_benchmark(classifier)
    test_adversarial_edge_cases(classifier)

    total_time = (time.perf_counter() - total_start) * 1000.0

    print("\n" + "=" * 70)
    print(f" ALL 8 TEST SUITES COMPLETED AND PASSED in {total_time:.1f} ms!")
    print("=" * 70)


if __name__ == "__main__":
    main()
