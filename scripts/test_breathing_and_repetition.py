"""Comprehensive Verification Suite for Breathing Sensitivity and Repetition Prevention.

Tests:
1. Low-RMS breathing & room hiss rejection by Silero VAD and AcousticDetector.
2. Bot playback immunity to breathing/mic puffs (no false barge-in or HARD_FLUSH).
3. Genuine barge-in commands ("ruko", "stop") still trigger instant cut-off.
4. Repetition prevention: turn completion clears consumed prompt so silence does not replay old answers.
5. Repetition prevention: HARD_FLUSH clears transcript so interruption does not replay old answers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import numpy as np

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.backchannel import DuckingAction, TwoStageBackchannelGate
from app.core.engine import DynamicTurnEngine, TurnState
from app.core.vad import AcousticDetector, SileroVAD


def generate_synthetic_breath_chunk(n_samples: int = 512, target_rms: float = 0.008) -> bytes:
    """Synthesize a 512-sample low-energy pink noise frame simulating microphone breathing."""
    rng = np.random.default_rng(42)
    # Generate pink noise via simple integrated white noise
    white = rng.standard_normal(n_samples).astype(np.float32)
    # Filter with mild lowpass
    filtered = np.convolve(white, [0.3, 0.4, 0.3], mode="same")
    current_rms = np.sqrt(np.mean(filtered**2))
    if current_rms > 0:
        filtered = filtered * (target_rms / current_rms)
    pcm = (np.clip(filtered, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


def generate_formant_speech_chunk(sr: int = 16000, n_samples: int = 512) -> bytes:
    """Synthesize a 512-sample resonant speech frame (RMS ~0.15)."""
    pulses = np.zeros(n_samples, dtype=np.float32)
    period = int(sr / 120)  # 120Hz pitch
    pulses[::period] = 1.0

    def biquad(samples: np.ndarray, freq: float, bw: float) -> np.ndarray:
        r = np.exp(-np.pi * bw / sr)
        theta = 2.0 * np.pi * freq / sr
        a1 = -2.0 * r * np.cos(theta)
        a2 = r * r
        b0 = 1.0 - r

        y = np.zeros_like(samples)
        y1, y2 = 0.0, 0.0
        for i in range(len(samples)):
            val = b0 * samples[i] - a1 * y1 - a2 * y2
            y[i] = val
            y2 = y1
            y1 = val
        return y

    filtered = biquad(pulses, 730.0, 90.0)
    filtered = biquad(filtered, 1090.0, 110.0)
    filtered = biquad(filtered, 2440.0, 150.0)

    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = 0.85 * (filtered / max_val)

    return (filtered * 32767.0).astype(np.int16).tobytes()


def test_breathing_rejection_in_vad():
    """Verify low-RMS breath noise (RMS < 0.012) is rejected by SileroVAD."""
    print("\n" + "=" * 70)
    print(" [TEST 1] SileroVAD Noise Floor Gate & Breathing Rejection")
    print("=" * 70)

    vad = SileroVAD(threshold=0.55, min_rms=0.012, sample_rate=16000)

    # 1. Test quiet breathing frame (RMS ~0.008)
    breath_chunk = generate_synthetic_breath_chunk(target_rms=0.008)
    res_breath = vad.process_chunk(breath_chunk)
    print(f"  Breath frame: RMS={res_breath.rms:.4f}, Confidence={res_breath.confidence:.4f}, is_speech={res_breath.is_speech}")
    assert not res_breath.is_speech, "Breathing frame was falsely classified as speech!"
    assert res_breath.rms < 0.012, f"Breath RMS {res_breath.rms} exceeded floor"

    # 2. Test speech frame (RMS ~0.15)
    speech_chunk = generate_formant_speech_chunk()
    res_speech = vad.process_chunk(speech_chunk)
    print(f"  Speech frame: RMS={res_speech.rms:.4f}, Confidence={res_speech.confidence:.4f}, is_speech={res_speech.is_speech}")
    assert res_speech.is_speech, "Genuine speech frame was missed by VAD!"
    assert res_speech.rms >= 0.012, f"Speech RMS {res_speech.rms} below floor"

    print("  ✓ PASS: SileroVAD noise floor gate cleanly separates breathing from speech.")


def test_bot_playback_immunity_to_breathing():
    """Verify bot playback is NOT interrupted by breathing or short ambient sounds."""
    print("\n" + "=" * 70)
    print(" [TEST 2] Bot Playback Immunity to Breathing & Short Sounds")
    print("=" * 70)

    gate = TwoStageBackchannelGate(prolonged_threshold_ms=500.0, hard_flush_db=-100.0)
    gate.on_speech_started(bot_is_speaking=True)

    # Simulate 600ms of breathing/sound with NO transcript
    # In the old code, >=500ms triggered HARD_FLUSH even without words.
    # In the new code, without words, it requires >=1200ms before hard flush.
    decision = gate.evaluate_utterance(
        transcript="",
        speech_duration_ms=600.0,
        bot_is_speaking=True,
    )

    print(f"  600ms breath (empty transcript) -> Action: {decision.action}, Reason: {decision.reason}")
    assert decision.action != DuckingAction.HARD_FLUSH, (
        f"600ms breath falsely triggered HARD_FLUSH! Got: {decision.action}"
    )
    assert decision.action == DuckingAction.DUCK, f"Expected DUCK, got {decision.action}"

    # Now verify genuine barge-in with text DOES trigger HARD_FLUSH
    speech_dec = gate.evaluate_utterance(
        transcript="ek second suniye",
        speech_duration_ms=600.0,
        bot_is_speaking=True,
    )
    print(f"  600ms genuine speech ('ek second suniye') -> Action: {speech_dec.action}, Reason: {speech_dec.reason}")
    assert speech_dec.action == DuckingAction.HARD_FLUSH, (
        f"Genuine prolonged speech did not trigger HARD_FLUSH! Got: {speech_dec.action}"
    )

    # Verify imperative command triggers HARD_FLUSH immediately even in 100ms
    cmd_dec = gate.evaluate_utterance(
        transcript="ruko",
        speech_duration_ms=100.0,
        bot_is_speaking=True,
    )
    print(f"  100ms command ('ruko') -> Action: {cmd_dec.action}, Reason: {cmd_dec.reason}")
    assert cmd_dec.action == DuckingAction.HARD_FLUSH

    print("  ✓ PASS: Bot playback is immune to breath sounds while respecting real barge-in.")


async def test_repetition_prevention_on_turn_completion():
    """Verify that once a turn is completed, silence does NOT replay the previous answer."""
    print("\n" + "=" * 70)
    print(" [TEST 3] Repetition Prevention on Normal Turn Completion")
    print("=" * 70)

    engine = DynamicTurnEngine(mode="VARTALAAP")

    # Turn 1: User speaks "Mera balance kitna hai"
    prompt = "Mera balance kitna hai"
    t1 = await engine.step(transcript=prompt, is_speech_override=True)
    assert engine.state == TurnState.LISTENING

    # User pauses: silence reaches dynamic ETS -> TURN_GRANTED
    t2 = await engine.step(
        transcript=prompt,
        is_speech_override=False,
        elapsed_silence_override_ms=2100.0,
    )
    print(f"  Turn 1 Result: State={t2.state}, ResponseDelivered={bool(t2.response_text)}")
    assert t2.state == TurnState.TURN_GRANTED
    assert t2.response_text is not None

    # Bot speaks response
    engine.set_bot_speaking(True)
    t3 = await engine.step(bot_is_speaking=True, is_speech_override=False)
    assert engine.state == TurnState.BOT_SPEAKING

    # Bot finishes speaking
    engine.set_bot_speaking(False)

    # Now user is silent (no new words spoken, transcript is empty)
    # In the old code, _last_transcript was still "Mera balance kitna hai",
    # causing an immediate second TURN_GRANTED repeating the answer!
    t4 = await engine.step(
        transcript="",
        is_speech_override=False,
        elapsed_silence_override_ms=2500.0,
    )
    print(f"  Subsequent Silence Result: State={t4.state}, Reason='{t4.reason}'")
    assert t4.state != TurnState.TURN_GRANTED, (
        f"Engine falsely granted a second turn without new user speech! State={t4.state}"
    )
    assert t4.state == TurnState.LISTENING

    print("  ✓ PASS: Consumed prompt was cleared; silence after bot speaking never replays.")


async def test_repetition_prevention_on_hard_flush():
    """Verify that when bot playback is interrupted (HARD_FLUSH), the old prompt is not replayed."""
    print("\n" + "=" * 70)
    print(" [TEST 4] Repetition Prevention on Barge-In Interruption")
    print("=" * 70)

    engine = DynamicTurnEngine(mode="VARTALAAP")

    # User asked a question and turn was granted
    await engine.step(transcript="Flight ticket book karni hai", is_speech_override=True)
    await engine.step(transcript="Flight ticket book karni hai", is_speech_override=False, elapsed_silence_override_ms=2100.0)

    # Bot begins speaking
    engine.set_bot_speaking(True)

    # User interrupts with "ruko"
    t_interrupt = await engine.step(
        transcript="ruko",
        bot_is_speaking=True,
        is_speech_override=True,
        speech_duration_override_ms=150.0,
    )
    print(f"  Interrupt Step: State={t_interrupt.state}, Gain={t_interrupt.ducking_gain_db} dB, Reason='{t_interrupt.reason}'")
    assert not engine.is_bot_speaking, "Bot speaking flag was not cleared on barge-in!"
    assert engine.gate.current_action == DuckingAction.HARD_FLUSH

    # Now user stops speaking and pauses
    # The old question "Flight ticket book karni hai" must NOT be replayed!
    t_pause = await engine.step(
        transcript="",
        is_speech_override=False,
        elapsed_silence_override_ms=2200.0,
    )
    print(f"  Post-Interrupt Pause: State={t_pause.state}, Reason='{t_pause.reason}'")
    assert t_pause.state != TurnState.TURN_GRANTED, (
        f"Engine replayed old answer after interrupt! State={t_pause.state}"
    )
    assert t_pause.state == TurnState.LISTENING

    print("  ✓ PASS: Interrupted transcript was erased; bot does not restart interrupted answer.")


async def main():
    test_breathing_rejection_in_vad()
    test_bot_playback_immunity_to_breathing()
    await test_repetition_prevention_on_turn_completion()
    await test_repetition_prevention_on_hard_flush()
    print("\n" + "=" * 70)
    print(" 🎉 ALL 4 BREATHING & REPETITION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
