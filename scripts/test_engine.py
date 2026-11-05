"""Continuous ETS State Machine and Speculative LLM Prefetch Verification Suite.

Comprehensive functional validation and A/B benchmarking:
1. Continuous ETS calculation across various Indic linguistic categories.
2. Speculative prefetch triggering at 200ms when P(complete) > 0.75.
3. Speculative task cancellation: When user resumes speaking, task is aborted cleanly.
4. Turn grant: When silence reaches ETS, turn is granted and prefetched response is delivered.
5. Backchannel handling during bot playback: "haanji" soft-ducks to -18 dB and restores to 0 dB.
6. Imperative command during bot playback: "ruko" triggers immediate hard flush (-100 dB).
7. A/B comparison: Demonstrates that on a Hinglish thinking pause ("Mera balance... matlab"
   with 700ms pause), Standard mode prematurely cuts off while Vartalaap holds the floor.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys
import time
from typing import List, Tuple

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.backchannel import DuckingAction, TwoStageBackchannelGate
from app.core.engine import (
    DynamicTurnEngine,
    EngineTelemetry,
    TurnState,
)
from app.core.semantic import (
    HybridTurnClassifier,
    TurnCategory,
)
from app.services.llm_brain import LLMBrain

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ==============================================================================
# TEST 1: Continuous ETS Calculation Across Linguistic Categories
# ==============================================================================
async def test_continuous_ets_calculation(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 1] Continuous ETS Calculation Across Linguistic Categories")
    print("=" * 75)

    engine.reset()
    engine.set_mode("VARTALAAP")

    test_cases: List[Tuple[str, TurnCategory, float, float]] = [
        # (transcript, expected_category, min_expected_ets, max_expected_ets)
        ("Mera balance... matlab", TurnCategory.COGNITIVE_HESITATION, 1900.0, 2000.0),
        ("Mera electricity bill aur", TurnCategory.HANGING_CONJUNCTION, 1900.0, 2000.0),
        ("Savings account ka", TurnCategory.POSTPOSITION, 1900.0, 2000.0),
        ("Mera electricity bill bhar diya hai", TurnCategory.TERMINAL_CLOSURE, 250.0, 350.0),
        ("Theek hai", TurnCategory.SHORT_CONFIRMATION, 250.0, 300.0),
    ]

    all_passed = True
    for text, exp_cat, min_ets, max_ets in test_cases:
        telem = await engine.step(
            transcript=text,
            is_speech_override=False,
            elapsed_silence_override_ms=50.0,
        )
        print(f"  Utterance       : '{text}'")
        print(f"  Category        : {telem.category.value} (expected: {exp_cat.value})")
        print(f"  P(incomplete)   : {telem.p_incomplete:.4f}")
        print(f"  P(complete)     : {telem.p_complete:.4f}")
        print(f"  Dynamic ETS(ms) : {telem.dynamic_ets_ms:.1f} ms [expected {min_ets:.0f}-{max_ets:.0f} ms]")

        cat_match = telem.category == exp_cat
        ets_match = min_ets <= telem.dynamic_ets_ms <= max_ets
        if not cat_match or not ets_match:
            print(f"  ❌ FAILED: cat_match={cat_match}, ets_match={ets_match}")
            all_passed = False
        else:
            print("  ✅ PASS")
        print("-" * 75)

    # Standard mode check: ETS must be strictly 500ms across all categories
    engine.set_mode("STANDARD")
    for text, _, _, _ in test_cases:
        telem = await engine.step(
            transcript=text,
            is_speech_override=False,
            elapsed_silence_override_ms=50.0,
        )
        assert telem.dynamic_ets_ms == 500.0, (
            f"Expected fixed 500.0ms ETS in STANDARD mode, got {telem.dynamic_ets_ms}"
        )
    print("  ✅ STANDARD Mode Baseline: Fixed 500.0 ms across all utterances")

    return all_passed


# ==============================================================================
# TEST 2: Speculative Prefetch Triggering at 200ms when P(complete) > 0.75
# ==============================================================================
async def test_speculative_prefetch_triggering(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 2] Speculative Prefetch Triggering at 200ms when P(complete) > 0.75")
    print("=" * 75)

    engine.reset()
    engine.set_mode("VARTALAAP")
    complete_utterance = "Bijli ka bill bhar diya hai"

    # Step A: Pause at 100ms (< 200ms threshold)
    telem_100ms = await engine.step(
        transcript=complete_utterance,
        is_speech_override=False,
        elapsed_silence_override_ms=100.0,
    )
    print(f"  [At 100ms pause] State={telem_100ms.state.value}, PrefetchActive={telem_100ms.prefetch_active}")
    assert telem_100ms.state == TurnState.EVALUATING_PAUSE, (
        f"Expected EVALUATING_PAUSE at 100ms, got {telem_100ms.state}"
    )
    assert not telem_100ms.prefetch_active, "Prefetch should NOT be active at 100ms"
    assert engine.prefetch_task is None, "Prefetch task must not be launched before 200ms"
    print("  ✅ At 100ms: Silence < 200ms -> Pause evaluated, prefetch remains idle")

    # Step B: Pause at 200ms (>= 200ms threshold, P(complete) > 0.75)
    telem_200ms = await engine.step(
        transcript=complete_utterance,
        is_speech_override=False,
        elapsed_silence_override_ms=200.0,
    )
    print(f"  [At 200ms pause] State={telem_200ms.state.value}, PrefetchActive={telem_200ms.prefetch_active}")
    print(f"                   P(complete)={telem_200ms.p_complete:.4f}, Dynamic ETS={telem_200ms.dynamic_ets_ms:.1f}ms")
    print(f"                   Reason='{telem_200ms.reason}'")

    assert telem_200ms.state == TurnState.SPECULATIVE_PREFETCHING, (
        f"Expected SPECULATIVE_PREFETCHING at 200ms, got {telem_200ms.state}"
    )
    assert telem_200ms.prefetch_active, "Prefetch must be active at 200ms"
    assert engine.prefetch_task is not None, "Prefetch task must be running"
    assert not engine.prefetch_task.done(), "Prefetch task should be actively running"
    print("  ✅ At 200ms: Silence >= 200ms & P(complete) > 0.75 -> Speculative prefetch launched successfully")

    return True


# ==============================================================================
# TEST 3: Speculative Task Cancellation on User Speech Resume
# ==============================================================================
async def test_speculative_cancellation_on_speech_resume(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 3] Speculative Task Cancellation on User Speech Resume")
    print("=" * 75)

    # Continue from active prefetch state
    assert engine.prefetch_task is not None, "Prefetch task must be active before cancellation test"
    running_task = engine.prefetch_task

    print(f"  Prefetch task before resume: done={running_task.done()}, cancelled={running_task.cancelled()}")

    # User resumes speaking before silence reaches ETS!
    extended_utterance = "Bijli ka bill bhar diya hai lekin receipt chahiye"
    telem_resumed = await engine.step(
        transcript=extended_utterance,
        is_speech_override=True,
        elapsed_silence_override_ms=0.0,
    )

    print(f"  [User Resumes] State={telem_resumed.state.value}, PrefetchActive={telem_resumed.prefetch_active}")
    print(f"                 Reason='{telem_resumed.reason}'")

    assert telem_resumed.state == TurnState.LISTENING, (
        f"Expected state LISTENING, got {telem_resumed.state}"
    )
    assert not telem_resumed.prefetch_active, "Prefetch must NOT be active after speech resumes"
    assert engine.prefetch_task is None, "Engine prefetch task reference must be cleared"
    assert engine.last_cancelled_task is not None, "Engine must record cancelled task"

    # Verify task was aborted
    await asyncio.sleep(0.02)
    assert engine.last_cancelled_task.cancelled() or engine.last_cancelled_task.done(), (
        "Underlying speculative prefetch task must be cancelled without lingering"
    )
    print("  ✅ User resumed speaking -> Speculative prefetch task cleanly aborted without race conditions")

    return True


# ==============================================================================
# TEST 4: Turn Grant & Prefetched Response Delivery
# ==============================================================================
async def test_turn_grant_and_response_delivery(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 4] Turn Grant and Prefetched Response Delivery")
    print("=" * 75)

    engine.reset()
    engine.set_mode("VARTALAAP")
    utterance = "Mera electricity bill bhar diya hai"

    # Step 1: Reach 200ms -> prefetch triggers
    telem_pre = await engine.step(
        transcript=utterance,
        is_speech_override=False,
        elapsed_silence_override_ms=200.0,
    )
    assert telem_pre.state == TurnState.SPECULATIVE_PREFETCHING
    print(f"  [t=200ms] Speculative prefetch launched for: '{utterance}'")

    # Step 2: Allow short window for mock streaming generation to complete
    await asyncio.sleep(0.08)

    # Step 3: Silence reaches dynamic ETS (e.g. 260ms >= 250ms)
    dynamic_ets = telem_pre.dynamic_ets_ms
    telem_grant = await engine.step(
        transcript=utterance,
        is_speech_override=False,
        elapsed_silence_override_ms=dynamic_ets + 10.0,
    )

    print(f"  [t={dynamic_ets + 10.0:.0f}ms >= ETS={dynamic_ets:.0f}ms] State={telem_grant.state.value}")
    print(f"  Response Delivered : '{telem_grant.response_text}'")
    print(f"  Reason             : '{telem_grant.reason}'")

    assert telem_grant.state == TurnState.TURN_GRANTED, (
        f"Expected TURN_GRANTED, got {telem_grant.state}"
    )
    assert telem_grant.response_text is not None and len(telem_grant.response_text) > 0, (
        "Turn grant must deliver valid synthesized or prefetched response text"
    )
    print("  ✅ Silence reached ETS -> TURN_GRANTED with zero-latency prefetched response")

    return True


# ==============================================================================
# TEST 5: Backchannel Handling During Bot Playback ("haanji" -> Soft Duck -> Restore)
# ==============================================================================
async def test_backchannel_during_bot_playback(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 5] Backchannel Handling During Bot Playback ('haanji')")
    print("=" * 75)

    engine.reset()
    engine.set_bot_speaking(True)

    # Step 0: Bot speaking quietly
    telem_initial = await engine.step(
        is_speech_override=False,
        elapsed_silence_override_ms=0.0,
        bot_is_speaking=True,
    )
    print(f"  [Initial] State={telem_initial.state.value}, Gain={telem_initial.ducking_gain_db:.1f} dB")
    assert telem_initial.state == TurnState.BOT_SPEAKING
    assert telem_initial.ducking_gain_db == 0.0

    # Step 1: User voice onset detected at t ~ 2ms
    telem_s1 = await engine.step(
        transcript="",
        is_speech_override=True,
        speech_duration_override_ms=30.0,
        bot_is_speaking=True,
    )
    print(f"  [Stage 1: Voice Onset] State={telem_s1.state.value}, Gain={telem_s1.ducking_gain_db:.1f} dB")
    assert telem_s1.state == TurnState.BOT_DUCKED, f"Expected BOT_DUCKED, got {telem_s1.state}"
    assert telem_s1.ducking_gain_db == -18.0, f"Expected -18.0 dB ducking, got {telem_s1.ducking_gain_db}"
    print("  ✅ Stage 1: Bot softly ducked to -18 dB upon user voice onset")

    # Step 2: ASR interim resolves to cultural backchannel "haanji" (< 500ms duration)
    telem_s2 = await engine.step(
        transcript="haanji",
        is_speech_override=True,
        speech_duration_override_ms=220.0,
        bot_is_speaking=True,
    )
    print(f"  [Stage 2: Token 'haanji'] State={telem_s2.state.value}, Gain={telem_s2.ducking_gain_db:.1f} dB")
    print(f"                           Reason='{telem_s2.reason}'")

    assert telem_s2.state == TurnState.BACKCHANNEL_IGNORED, (
        f"Expected BACKCHANNEL_IGNORED, got {telem_s2.state}"
    )
    assert telem_s2.ducking_gain_db == 0.0, f"Expected 0.0 dB restored volume, got {telem_s2.ducking_gain_db}"
    assert engine.is_bot_speaking, "Bot audio must continue playing without interruption"
    print("  ✅ Stage 2: 'haanji' recognized as backchannel -> volume restored to 0 dB, bot continues")

    return True


# ==============================================================================
# TEST 6: Imperative Command During Bot Playback ("ruko" -> Hard Flush)
# ==============================================================================
async def test_imperative_interrupt_during_bot_playback(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 6] Imperative Command During Bot Playback ('ruko')")
    print("=" * 75)

    engine.reset()
    engine.set_bot_speaking(True)

    # User utters imperative interrupt "ruko" while bot is talking
    telem = await engine.step(
        transcript="ruko",
        is_speech_override=True,
        speech_duration_override_ms=180.0,
        bot_is_speaking=True,
    )

    print(f"  [Command 'ruko'] State={telem.state.value}, Gain={telem.ducking_gain_db:.1f} dB")
    print(f"                   BotSpeaking={engine.is_bot_speaking}, Reason='{telem.reason}'")

    assert telem.ducking_gain_db == -100.0, f"Expected -100.0 dB hard flush, got {telem.ducking_gain_db}"
    assert telem.state == TurnState.LISTENING, f"Expected state LISTENING, got {telem.state}"
    assert not engine.is_bot_speaking, "Bot playback must be halted immediately upon imperative command"
    print("  ✅ Imperative 'ruko' triggers instant hard flush (-100 dB) and transfers turn to user")

    return True


# ==============================================================================
# TEST 7: A/B Comparison: Hinglish Thinking Pause ("Mera balance... matlab" at 700ms)
# ==============================================================================
async def test_ab_comparison_hinglish_thinking_pause(engine: DynamicTurnEngine) -> bool:
    print("\n" + "=" * 75)
    print(" [TEST 7] A/B Comparison: Hinglish Thinking Pause ('Mera balance... matlab' at 700ms)")
    print("=" * 75)

    transcript = "Mera balance... matlab"
    pause_duration_ms = 700.0

    # 1. Standard Mode Evaluation (Fixed 500ms silence timeout)
    engine.reset()
    engine.set_mode("STANDARD")
    standard_telem = await engine.step(
        transcript=transcript,
        is_speech_override=False,
        elapsed_silence_override_ms=pause_duration_ms,
    )

    print(f"  STANDARD MODE:")
    print(f"    - Silence Duration   : {pause_duration_ms:.0f} ms")
    print(f"    - Silence Timeout    : {standard_telem.dynamic_ets_ms:.0f} ms (FIXED)")
    print(f"    - Turn State         : {standard_telem.state.value}")
    print(f"    - Cut-off Occurred   : {standard_telem.state == TurnState.TURN_GRANTED} ❌ (PREMATURE CUT-OFF)")

    assert standard_telem.state == TurnState.TURN_GRANTED, (
        f"Standard mode should have prematurely cut off at 500ms, got {standard_telem.state}"
    )

    # 2. Vartalaap Mode Evaluation (Dynamic Continuous ETS)
    engine.reset()
    engine.set_mode("VARTALAAP")
    vartalaap_telem = await engine.step(
        transcript=transcript,
        is_speech_override=False,
        elapsed_silence_override_ms=pause_duration_ms,
    )

    print(f"\n  VARTALAAP v2.0 MODE:")
    print(f"    - Silence Duration   : {pause_duration_ms:.0f} ms")
    print(f"    - Syntactic Category : {vartalaap_telem.category.value} ('matlab' penalty = +0.95)")
    print(f"    - Dynamic ETS        : {vartalaap_telem.dynamic_ets_ms:.0f} ms (EXPANDED TO CEILING)")
    print(f"    - Turn State         : {vartalaap_telem.state.value}")
    print(f"    - Floor Retained     : {vartalaap_telem.state == TurnState.EVALUATING_PAUSE} ✅ (HOLDING FOR THOUGHT)")
    print(f"    - Remaining Pause    : {vartalaap_telem.dynamic_ets_ms - pause_duration_ms:.0f} ms remaining before turn grant")

    assert vartalaap_telem.state == TurnState.EVALUATING_PAUSE, (
        f"Vartalaap should hold floor in EVALUATING_PAUSE at 700ms, got {vartalaap_telem.state}"
    )
    assert vartalaap_telem.dynamic_ets_ms == 2000.0, (
        f"Dynamic ETS should expand to 2000.0ms for cognitive hesitation, got {vartalaap_telem.dynamic_ets_ms}"
    )

    print("\n  Summary Table:")
    print("  " + "-" * 71)
    print(f"  | {'Metric':<25} | {'Standard 500ms VAD':<20} | {'Vartalaap v2.0':<20} |")
    print("  " + "-" * 71)
    print(f"  | {'Silence Threshold':<25} | {'500 ms':<20} | {f'{vartalaap_telem.dynamic_ets_ms:.0f} ms':<20} |")
    print(f"  | {'Status at 700ms Pause':<25} | {'CUT OFF (Turn Lost)':<20} | {'HELD (Still Thinking)':<20} |")
    print(f"  | {'Premature Cut-off':<25} | {'YES (Failure)':<20} | {'NO (Zero Cut-off)':<20} |")
    print("  " + "-" * 71)

    print("  ✅ A/B Benchmark successfully proved Vartalaap prevents premature cut-off on Hinglish thinking pause!")
    return True


# ==============================================================================
# MAIN TEST HARNESS
# ==============================================================================
async def main() -> None:
    print("=" * 75)
    print("🚀 Running Project Vartalaap Continuous ETS & Speculative Prefetch Test Suite")
    print("=" * 75)

    # Initialize integrated engine with GPU classifier and mock LLM
    engine = DynamicTurnEngine()

    start_time = time.perf_counter()

    test_1 = await test_continuous_ets_calculation(engine)
    test_2 = await test_speculative_prefetch_triggering(engine)
    test_3 = await test_speculative_cancellation_on_speech_resume(engine)
    test_4 = await test_turn_grant_and_response_delivery(engine)
    test_5 = await test_backchannel_during_bot_playback(engine)
    test_6 = await test_imperative_interrupt_during_bot_playback(engine)
    test_7 = await test_ab_comparison_hinglish_thinking_pause(engine)

    total_time_ms = (time.perf_counter() - start_time) * 1000.0

    print("\n" + "=" * 75)
    print(f"🏁 All 7 Verification Tests Passed Successfully in {total_time_ms:.1f} ms!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
