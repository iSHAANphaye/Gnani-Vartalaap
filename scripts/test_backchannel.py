"""Two-Stage Soft-Ducking Cultural Gate and Cross-Script Normalizer Verification Suite.

Comprehensive functional validation and sub-millisecond latency benchmarking:
1. Cross-script normalization across Devanagari, Tamil, Telugu, Kannada, and elongated Romanized text.
2. Stage 1 soft-ducking (-18 dB) triggered on VAD speech onset during active playback.
3. Stage 2 cultural backchannel resolution ("haanji", "achha", "seri", "సరే" restore volume to 0 dB).
4. Stage 2 imperative interrupt resolution ("ruko", "cancel", "stop" trigger HARD_FLUSH to -100 dB).
5. Stage 2 prolonged speech resolution (>500ms non-backchannel phrase triggers HARD_FLUSH).
6. High-precision execution latency benchmark verifying sub-millisecond (<1.0ms) evaluation budget.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import List, Tuple

# Ensure workspace root is present in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.backchannel import (
    DEFAULT_INTERRUPT_COMMANDS,
    DuckingAction,
    GateDecision,
    TwoStageBackchannelGate,
)
from app.core.normalizer import (
    CANONICAL_DICTIONARY,
    CrossScriptNormalizer,
    STANDARD_BACKCHANNEL_TOKENS,
)


def test_cross_script_normalization() -> bool:
    """Validate cross-script normalization, punctuation stripping, and elongation collapsing."""
    print("\n" + "=" * 70)
    print(" [TEST 1] Cross-Script Normalizer Verification")
    print("=" * 70)

    normalizer = CrossScriptNormalizer()

    # Test cases: (input_text, expected_normalized)
    test_cases: List[Tuple[str, str, str]] = [
        # Elongated Romanized
        ("haaanji", "haanji", "Hindi Romanized 3x vowel elongation"),
        ("achhaaa", "achha", "Hindi Romanized trailing 3x vowel elongation"),
        ("seriii", "seri", "Tamil Romanized 3x vowel elongation"),
        ("theeeek", "theek", "Hindi Romanized middle vowel elongation"),
        ("hmmmm", "hmm", "Universal consonant elongation"),
        ("haaaanjiii", "haanji", "Multi-site Romanized elongation"),
        # Devanagari (Hindi)
        ("हाँ", "haan", "Devanagari affirmative token"),
        ("हाँजी!", "haanji", "Devanagari polite affirmative with punctuation"),
        ("अच्छा...", "achha", "Devanagari acknowledgment with ellipsis"),
        ("ठीक है।", "theek hai", "Devanagari agreement phrase with Indic danda"),
        ("thik hai", "theek hai", "Hindi Romanized phonetic variation"),
        ("sahi hai", "sahi hai", "Hindi agreement phrase"),
        ("bilkul", "bilkul", "Hindi emphatic acknowledgment"),
        # Tamil
        ("சரி", "seri", "Tamil native script acknowledgment"),
        ("serii", "seri", "Tamil Romanized elongation variant"),
        ("ஆமா", "aama", "Tamil native affirmative"),
        ("aamaa", "aama", "Tamil Romanized elongation variant"),
        ("apdiya?", "apdiya", "Tamil conversational backchannel with question mark"),
        # Telugu
        ("సరే", "sare", "Telugu native script acknowledgment"),
        ("saree", "sare", "Telugu Romanized elongation variant"),
        ("అవును", "avunu", "Telugu native affirmative"),
        ("avunu!", "avunu", "Telugu Romanized affirmative with exclamation"),
        ("ante", "ante", "Telugu conversational filler/backchannel"),
        # Kannada
        ("ಸರಿ", "sari", "Kannada native script acknowledgment"),
        ("sari", "sari", "Kannada Romanized acknowledgment"),
        ("ಹೌದು", "haudu", "Kannada native affirmative"),
        ("haudu", "haudu", "Kannada Romanized affirmative"),
        # Universal / Colloquial
        ("  hmmm...  ", "hmm", "Universal nasal filler with whitespace and punctuation"),
        ("okay", "okay", "English standard acknowledgment"),
        ("ok", "okay", "English abbreviated acknowledgment"),
        ("yeah", "yeah", "English informal affirmative"),
        ("yep", "yep", "English quick affirmative"),
    ]

    for raw_input, expected, description in test_cases:
        actual = normalizer.normalize(raw_input)
        assert (
            actual == expected
        ), f"Normalization mismatch for '{raw_input}' ({description}): expected '{expected}', got '{actual}'"
        assert normalizer.is_backchannel(
            raw_input
        ), f"Expected '{raw_input}' to be recognized as backchannel token"
        print(f"  [OK] {raw_input!r:<16} -> {actual!r:<12} ({description})")

    # Verify dedicated elongation collapse method
    assert normalizer.collapse_elongation("haaanji") == "haanji"
    assert normalizer.collapse_elongation("achhaaa") == "achha"
    assert normalizer.collapse_elongation("seriii") == "seri"
    print("  [OK] Dedicated collapse_elongation confirmed for test specifications.")

    # Verify non-backchannel words are not falsely classified
    assert not normalizer.is_backchannel("namaste")
    assert not normalizer.is_backchannel("flight ticket")
    assert not normalizer.is_backchannel("kya aap sun rahe ho")
    print("  [OK] Non-backchannel phrases correctly rejected.")

    print(" [PASSED] Test 1: Cross-Script Normalizer Verification Succeeded.")
    return True


def test_stage1_soft_ducking() -> bool:
    """Validate Stage 1 immediate soft-ducking (-18 dB) when bot is speaking."""
    print("\n" + "=" * 70)
    print(" [TEST 2] Stage 1 Soft-Ducking (-18 dB) Onset Verification")
    print("=" * 70)

    gate = TwoStageBackchannelGate(soft_duck_db=-18.0)
    assert not gate.is_ducked
    assert gate.current_gain_db == 0.0

    # Scenario A: Bot is actively speaking -> Duck to -18 dB
    decision_speaking = gate.on_speech_started(bot_is_speaking=True)
    assert decision_speaking.action == DuckingAction.DUCK, f"Expected DUCK, got {decision_speaking.action}"
    assert decision_speaking.gain_db == -18.0, f"Expected -18.0 dB, got {decision_speaking.gain_db}"
    assert not decision_speaking.is_backchannel
    assert not decision_speaking.is_barge_in
    assert gate.is_ducked
    assert gate.current_gain_db == -18.0
    print("  [OK] Bot speaking: on_speech_started -> DuckingAction.DUCK, gain = -18.0 dB")

    # Reset gate
    gate.reset()
    assert not gate.is_ducked
    assert gate.current_gain_db == 0.0
    print("  [OK] Gate reset restores quiescent state (gain = 0.0 dB, not ducked)")

    # Scenario B: Bot is silent -> Action is NONE, 0.0 dB
    decision_silent = gate.on_speech_started(bot_is_speaking=False)
    assert decision_silent.action == DuckingAction.NONE, f"Expected NONE, got {decision_silent.action}"
    assert decision_silent.gain_db == 0.0, f"Expected 0.0 dB, got {decision_silent.gain_db}"
    assert not decision_silent.is_backchannel
    assert not decision_silent.is_barge_in
    assert not gate.is_ducked
    assert gate.current_gain_db == 0.0
    print("  [OK] Bot silent: on_speech_started -> DuckingAction.NONE, gain = 0.0 dB")

    print(" [PASSED] Test 2: Stage 1 Soft-Ducking Verification Succeeded.")
    return True


def test_stage2_backchannel_resolution() -> bool:
    """Validate Stage 2 backchannel resolution restoring volume to 0 dB."""
    print("\n" + "=" * 70)
    print(" [TEST 3] Stage 2 Backchannel Resolution Verification")
    print("=" * 70)

    gate = TwoStageBackchannelGate(soft_duck_db=-18.0, prolonged_threshold_ms=500.0)

    # Core required backchannel tokens
    target_tokens = ["haanji", "achha", "seri", "సరే"]

    for token in target_tokens:
        # Pre-duck via Stage 1
        s1 = gate.on_speech_started(bot_is_speaking=True)
        assert s1.action == DuckingAction.DUCK
        assert gate.is_ducked

        # Stage 2 evaluation with duration < 500ms
        duration_ms = 240.0
        decision = gate.evaluate_utterance(
            transcript=token,
            speech_duration_ms=duration_ms,
            bot_is_speaking=True,
        )

        assert (
            decision.action == DuckingAction.RESTORE
        ), f"Expected RESTORE for '{token}', got {decision.action}"
        assert (
            decision.gain_db == 0.0
        ), f"Expected gain_db 0.0 dB for '{token}', got {decision.gain_db}"
        assert (
            decision.reason == "BACKCHANNEL_IGNORED"
        ), f"Expected reason 'BACKCHANNEL_IGNORED', got '{decision.reason}'"
        assert decision.is_backchannel is True
        assert decision.is_barge_in is False
        assert not gate.is_ducked
        assert gate.current_gain_db == 0.0

        print(f"  [OK] Token '{token}' ({duration_ms}ms) -> RESTORE (0.0 dB, BACKCHANNEL_IGNORED)")

    # Additional multi-script backchannel assertions
    additional_tokens = ["हाँजी", "சரி", "theek hai", "hmmm", "okay"]
    for token in additional_tokens:
        gate.on_speech_started(bot_is_speaking=True)
        dec = gate.evaluate_utterance(token, speech_duration_ms=280.0, bot_is_speaking=True)
        assert dec.action == DuckingAction.RESTORE
        assert dec.gain_db == 0.0
        assert dec.reason == "BACKCHANNEL_IGNORED"
        print(f"  [OK] Multi-script token '{token}' -> RESTORE (0.0 dB, BACKCHANNEL_IGNORED)")

    print(" [PASSED] Test 3: Stage 2 Backchannel Resolution Succeeded.")
    return True


def test_stage2_imperative_resolution() -> bool:
    """Validate Stage 2 imperative interrupt commands triggering HARD_FLUSH (-100 dB)."""
    print("\n" + "=" * 70)
    print(" [TEST 4] Stage 2 Imperative Resolution Verification")
    print("=" * 70)

    gate = TwoStageBackchannelGate(hard_flush_db=-100.0)

    # Core required imperative commands
    imperative_tokens = ["ruko", "cancel", "stop"]

    for cmd in imperative_tokens:
        gate.on_speech_started(bot_is_speaking=True)
        decision = gate.evaluate_utterance(
            transcript=cmd,
            speech_duration_ms=180.0,
            bot_is_speaking=True,
        )

        assert (
            decision.action == DuckingAction.HARD_FLUSH
        ), f"Expected HARD_FLUSH for '{cmd}', got {decision.action}"
        assert (
            decision.gain_db == -100.0
        ), f"Expected -100.0 dB for '{cmd}', got {decision.gain_db}"
        assert (
            decision.reason == "IMPERATIVE_BARGE_IN"
        ), f"Expected reason 'IMPERATIVE_BARGE_IN', got '{decision.reason}'"
        assert decision.is_backchannel is False
        assert decision.is_barge_in is True
        assert gate.current_gain_db == -100.0

        print(f"  [OK] Command '{cmd}' -> HARD_FLUSH (-100.0 dB, IMPERATIVE_BARGE_IN)")

    # Extended Indic and English imperative commands
    extended_cmds = ["chup", "wait", "hold on", "band karo", "rukiye", "रुको", "बंद करो", "நிறுத்து", "ఆపు"]
    for cmd in extended_cmds:
        gate.on_speech_started(bot_is_speaking=True)
        dec = gate.evaluate_utterance(cmd, speech_duration_ms=150.0, bot_is_speaking=True)
        assert dec.action == DuckingAction.HARD_FLUSH
        assert dec.gain_db == -100.0
        assert dec.reason == "IMPERATIVE_BARGE_IN"
        print(f"  [OK] Extended command '{cmd}' -> HARD_FLUSH (-100.0 dB, IMPERATIVE_BARGE_IN)")

    # Embedded imperative command inside sentence (e.g. "please stop")
    gate.on_speech_started(bot_is_speaking=True)
    dec_embedded = gate.evaluate_utterance("please stop now", speech_duration_ms=250.0, bot_is_speaking=True)
    assert dec_embedded.action == DuckingAction.HARD_FLUSH
    assert dec_embedded.reason == "IMPERATIVE_BARGE_IN"
    print("  [OK] Embedded command 'please stop now' -> HARD_FLUSH (IMPERATIVE_BARGE_IN)")

    print(" [PASSED] Test 4: Stage 2 Imperative Resolution Succeeded.")
    return True


def test_stage2_prolonged_speech_resolution() -> bool:
    """Validate Stage 2 prolonged speech (>500ms non-backchannel phrase) triggers HARD_FLUSH."""
    print("\n" + "=" * 70)
    print(" [TEST 5] Stage 2 Prolonged Speech Resolution Verification")
    print("=" * 70)

    gate = TwoStageBackchannelGate(prolonged_threshold_ms=500.0, hard_flush_db=-100.0)

    # Sub-case A: Non-backchannel utterance under 500ms maintains soft-duck pending state
    gate.on_speech_started(bot_is_speaking=True)
    sub_decision = gate.evaluate_utterance(
        transcript="mujhe",
        speech_duration_ms=320.0,
        bot_is_speaking=True,
    )
    assert sub_decision.action == DuckingAction.DUCK
    assert sub_decision.gain_db == -18.0
    assert sub_decision.reason == "PENDING_INTERIM_SPEECH"
    assert not sub_decision.is_backchannel
    assert not sub_decision.is_barge_in
    print("  [OK] Interim non-backchannel speech (320ms < 500ms) -> DUCK (-18.0 dB, PENDING_INTERIM_SPEECH)")

    # Sub-case B: Non-backchannel utterance exceeding 500ms triggers PROLONGED_BARGE_IN
    prolonged_phrases = [
        ("mujhe flight ticket book karni hai", 620.0),
        ("ek minute suniye aap kya bol rahe hain", 850.0),
        ("i need to change my order address please", 550.0),
        ("enna vishayam theriyuma", 510.0),
    ]

    for phrase, dur_ms in prolonged_phrases:
        gate.on_speech_started(bot_is_speaking=True)
        decision = gate.evaluate_utterance(
            transcript=phrase,
            speech_duration_ms=dur_ms,
            bot_is_speaking=True,
        )

        assert (
            decision.action == DuckingAction.HARD_FLUSH
        ), f"Expected HARD_FLUSH for '{phrase}' ({dur_ms}ms), got {decision.action}"
        assert decision.gain_db == -100.0
        assert (
            decision.reason == "PROLONGED_BARGE_IN"
        ), f"Expected 'PROLONGED_BARGE_IN', got '{decision.reason}'"
        assert decision.is_backchannel is False
        assert decision.is_barge_in is True

        print(f"  [OK] Phrase '{phrase[:32]}...' ({dur_ms}ms) -> HARD_FLUSH (-100.0 dB, PROLONGED_BARGE_IN)")

    # Sub-case C: User starts with backchannel token but speaks for >= 500ms
    gate.on_speech_started(bot_is_speaking=True)
    extended_bc = gate.evaluate_utterance("haanji", speech_duration_ms=600.0, bot_is_speaking=True)
    assert extended_bc.action == DuckingAction.HARD_FLUSH
    assert extended_bc.reason == "PROLONGED_BARGE_IN"
    print("  [OK] Extended backchannel token (600ms >= 500ms) -> HARD_FLUSH (PROLONGED_BARGE_IN)")

    print(" [PASSED] Test 5: Stage 2 Prolonged Speech Resolution Succeeded.")
    return True


def test_latency_benchmark() -> bool:
    """Benchmark evaluation and normalization latency over 10,000 iterations (<1.0ms budget)."""
    print("\n" + "=" * 70)
    print(" [TEST 6] High-Precision Execution Latency Benchmark")
    print("=" * 70)

    gate = TwoStageBackchannelGate()
    normalizer = gate.normalizer

    iterations = 10_000

    benchmark_scenarios = [
        ("Normalizer: 'haaanji' (elongation)", lambda: normalizer.normalize("haaanji")),
        ("Normalizer: 'சரி' (Tamil script)", lambda: normalizer.normalize("சரி")),
        ("Normalizer: 'సరే' (Telugu script)", lambda: normalizer.normalize("సరే")),
        ("Stage 1: on_speech_started", lambda: gate.on_speech_started(bot_is_speaking=True)),
        ("Stage 2: evaluate_utterance ('haanji' backchannel)", lambda: gate.evaluate_utterance("haanji", 220.0, True)),
        ("Stage 2: evaluate_utterance ('ruko' imperative)", lambda: gate.evaluate_utterance("ruko", 180.0, True)),
        ("Stage 2: evaluate_utterance (barge-in 600ms)", lambda: gate.evaluate_utterance("ticket book karo", 600.0, True)),
    ]

    print(f" Running {iterations:,} warm-up and measured iterations per scenario...\n")
    print(f" {'Scenario':<48} | {'Mean (us)':<10} | {'P95 (us)':<9} | {'P99 (us)':<9} | {'Max (us)':<9} | Status")
    print("-" * 105)

    all_passed = True
    for label, fn in benchmark_scenarios:
        # Warmup
        for _ in range(500):
            fn()

        latencies_us: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)

        latencies_us.sort()
        mean_us = sum(latencies_us) / len(latencies_us)
        p95_us = latencies_us[int(0.95 * len(latencies_us))]
        p99_us = latencies_us[int(0.99 * len(latencies_us))]
        max_us = latencies_us[-1]

        # Budget: < 1.0 ms (1,000.0 us)
        within_budget = p99_us < 1000.0 and mean_us < 1000.0
        if not within_budget:
            all_passed = False

        status_str = "PASS (<1.0ms)" if within_budget else "FAIL (>1.0ms)"
        print(f" {label:<48} | {mean_us:<10.2f} | {p95_us:<9.2f} | {p99_us:<9.2f} | {max_us:<9.2f} | {status_str}")

    assert all_passed, "Execution latency exceeded 1.0ms budget!"
    print("\n [PASSED] Test 6: High-Precision Execution Latency Benchmark Succeeded.")
    return True


def run_all_tests() -> int:
    """Run all verification suite tests sequentially and return exit code."""
    print("=" * 70)
    print(" Vartalaap Two-Stage Cultural Gate & Normalizer Verification Suite")
    print("=" * 70)

    t_start = time.perf_counter()

    results = [
        test_cross_script_normalization(),
        test_stage1_soft_ducking(),
        test_stage2_backchannel_resolution(),
        test_stage2_imperative_resolution(),
        test_stage2_prolonged_speech_resolution(),
        test_latency_benchmark(),
    ]

    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    if all(results):
        print(f" ALL 6 SUITE TESTS PASSED SUCCESSFULLY in {t_total:.3f}s!")
        print("=" * 70)
        return 0
    else:
        print(" SOME SUITE TESTS FAILED!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
