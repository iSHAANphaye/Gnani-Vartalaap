"""Benchmark Test Sample Generator for Project Vartalaap.

Synthesizes 30 curated Indic code-mixed acoustic stress scenarios as 16kHz mono 16-bit PCM
WAV files and generates the comprehensive manifest.json.
Uses realistic human vocal formant synthesis with pitch contours, syllable modulation,
formant resonances, and natural speech envelopes to ensure accurate Silero VAD parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import wave

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vartalaap.benchmark.generator")

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT_DIR / "benchmark" / "test_samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"


def synthesize_speech_waveform(
    duration_sec: float,
    f0_base: float = 135.0,
    syllable_rate: float = 3.8,
    sr: int = 16000,
) -> np.ndarray:
    """Synthesize speech-like acoustic audio using dynamic formant resonances.

    Produces realistic human vocal formant harmonics (F1, F2, F3) with natural pitch
    prosody, syllable envelope modulation, and onset/offset windowing.
    """
    n_samples = int(duration_sec * sr)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.int16)

    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Syllable modulation envelope
    syllable_env = np.maximum(0.0, np.sin(2.0 * np.pi * syllable_rate * t)) ** 1.35

    # Smooth raised-cosine ramp at onsets and offsets
    ramp_len = min(int(0.025 * sr), max(1, n_samples // 6))
    ramp = np.ones(n_samples, dtype=np.float32)
    if ramp_len > 0:
        ramp[:ramp_len] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        ramp[-ramp_len:] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len - 1, -1, -1) / ramp_len))

    # Dynamic pitch with prosodic contour
    f0 = f0_base + 16.0 * np.sin(2.0 * np.pi * 1.4 * t) + 4.0 * np.sin(2.0 * np.pi * 5.8 * t)
    phase = np.cumsum(2.0 * np.pi * f0 / sr)

    # Time-varying formant trajectories
    f1 = 530.0 + 220.0 * np.sin(2.0 * np.pi * 2.0 * t)
    f2 = 1580.0 + 420.0 * np.cos(2.0 * np.pi * 2.4 * t)
    f3 = 2580.0 + 280.0 * np.sin(2.0 * np.pi * 1.1 * t)

    audio = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 28):
        h_freq = f0 * h
        h_phase = h * phase
        amp1 = np.exp(-((h_freq - f1) ** 2) / (2.0 * (115.0 ** 2)))
        amp2 = 0.65 * np.exp(-((h_freq - f2) ** 2) / (2.0 * (165.0 ** 2)))
        amp3 = 0.35 * np.exp(-((h_freq - f3) ** 2) / (2.0 * (265.0 ** 2)))
        amp = (amp1 + amp2 + amp3) / np.sqrt(h)
        audio += (amp * np.sin(h_phase)).astype(np.float32)

    # Ambient vocal fold breath noise
    noise = np.random.normal(0, 0.025, n_samples).astype(np.float32)
    signal = (audio * syllable_env + noise * 0.06) * ramp

    max_val = np.max(np.abs(signal)) + 1e-6
    normalized = (signal / max_val) * 0.75 * 32767.0
    return normalized.astype(np.int16)


def synthesize_silence(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Synthesize clean digital silence (zeros)."""
    n_samples = int(duration_sec * sr)
    return np.zeros(n_samples, dtype=np.int16)


def write_wav(file_path: Path, pcm_samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Save 16kHz mono 16-bit PCM audio to WAV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_samples.tobytes())


def generate_all_samples() -> Dict[str, any]:
    """Generate all 30 test scenarios across Categories A, B, C, D, and E."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    samples_meta: List[Dict[str, any]] = []

    # ==========================================================================
    # CATEGORY A (10 Samples): Mid-utterance Code-Mixing Pauses
    # ==========================================================================
    cat_a_defs = [
        (
            "sample_01_hinglish_ke_postposition",
            "Hinglish",
            "Mera account balance ke",
            750,
            "baare mein batao",
            "ke",
            "Mid-utterance pause after Hindi genitive postposition 'ke'",
            1400,
            1200,
        ),
        (
            "sample_02_hinglish_lekin_conjunction",
            "Hinglish",
            "Flight timing dekhna tha lekin",
            650,
            "subah ki flight cancel ho gayi",
            "lekin",
            "Mid-utterance pause after contrastive conjunction 'lekin'",
            1600,
            1500,
        ),
        (
            "sample_03_tanglish_illa_conjunction",
            "Tanglish",
            "Train ticket confirm aayiducha illa",
            800,
            "waiting list-la irukka?",
            "illa",
            "Mid-utterance pause after Tamil disjunctive conjunction 'illa'",
            1700,
            1300,
        ),
        (
            "sample_04_hinglish_aur_conjunction",
            "Hinglish",
            "Mera electricity bill aur",
            700,
            "water bill dono kitna aaya?",
            "aur",
            "Mid-utterance pause after Hindi additive conjunction 'aur'",
            1400,
            1500,
        ),
        (
            "sample_05_hinglish_mein_postposition",
            "Hinglish",
            "Savings account mein",
            850,
            "paise transfer karna hai",
            "mein",
            "Mid-utterance pause after Hindi locative postposition 'mein'",
            1300,
            1400,
        ),
        (
            "sample_06_telugu_kaani_conjunction",
            "Telugu",
            "Nenu kotha account open cheddam anukuntunna kaani",
            750,
            "minimum balance entha undali?",
            "kaani",
            "Mid-utterance pause after Telugu conjunction 'kaani'",
            2000,
            1400,
        ),
        (
            "sample_07_tanglish_aana_conjunction",
            "Tanglish",
            "Credit card statement check pannanum aana",
            650,
            "password marandhuten",
            "aana",
            "Mid-utterance pause after Tamil conjunction 'aana'",
            1800,
            1100,
        ),
        (
            "sample_08_tanglish_pathi_postposition",
            "Tanglish",
            "Fixed deposit interest rate pathi",
            800,
            "details solla mudiyuma?",
            "pathi",
            "Mid-utterance pause after Tamil postposition 'pathi'",
            1600,
            1200,
        ),
        (
            "sample_09_tanglish_aadhunaala_conjunction",
            "Tanglish",
            "UPI payment fail aagiduchi aadhunaala",
            700,
            "refund eppo credit aagum?",
            "aadhunaala",
            "Mid-utterance pause after Tamil causal conjunction 'aadhunaala'",
            1800,
            1300,
        ),
        (
            "sample_10_hinglish_ya_conjunction",
            "Hinglish",
            "Mera KYC verify ho gaya kya ya",
            900,
            "mujhe branch jaana padega?",
            "ya",
            "Mid-utterance pause after Hindi alternative conjunction 'ya'",
            1600,
            1400,
        ),
    ]

    for idx, (sid, lang, p1, pause_ms, p2, cue, note, sp1_ms, sp2_ms) in enumerate(cat_a_defs, start=1):
        sp1 = synthesize_speech_waveform(sp1_ms / 1000.0, f0_base=130.0 + idx * 2.0)
        sil = synthesize_silence(pause_ms / 1000.0)
        sp2 = synthesize_speech_waveform(sp2_ms / 1000.0, f0_base=130.0 + idx * 2.0)
        audio = np.concatenate([sp1, sil, sp2])

        wav_path = SAMPLES_DIR / f"{sid}.wav"
        write_wav(wav_path, audio)

        samples_meta.append({
            "id": f"sample_{idx:02d}",
            "sample_name": sid,
            "category": "A",
            "category_name": "MID_UTTERANCE_CODE_MIXING_PAUSE",
            "language": lang,
            "file_path": str(wav_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "transcript_part1": p1,
            "pause_duration_ms": pause_ms,
            "transcript_part2": p2,
            "full_transcript": f"{p1} {p2}",
            "speech1_duration_ms": sp1_ms,
            "speech2_duration_ms": sp2_ms,
            "total_duration_ms": sp1_ms + pause_ms + sp2_ms,
            "linguistic_cue": cue,
            "bot_is_speaking": False,
            "description": note,
            "expected_standard_vad": "PREMATURE_CUTOFF_AT_500MS",
            "expected_vartalaap": "HOLD_FLOOR_DYNAMIC_ETS_EXPANSION",
        })

    # ==========================================================================
    # CATEGORY B (6 Samples): Cognitive Hesitation and Fillers
    # ==========================================================================
    cat_b_defs = [
        (
            "sample_11_hinglish_filler_woh_kya_hai_na",
            "Hinglish",
            "Mujhe ek loan query puchna tha woh kya hai na",
            850,
            "interest rate kitna chal raha hai?",
            "woh kya hai na",
            "Hesitation filler 'woh kya hai na' before specification",
            1900,
            1400,
        ),
        (
            "sample_12_hinglish_filler_matlab",
            "Hinglish",
            "Mera balance cut ho gaya matlab",
            950,
            "ATM se cash nahi nikla lekin debit ho gaya",
            "matlab",
            "Cognitive reformulation pause after 'matlab'",
            1500,
            1600,
        ),
        (
            "sample_13_telugu_filler_ante",
            "Telugu",
            "Nenu repu branch ki vasthanu ante",
            800,
            "passbook printing kosam rammantara?",
            "ante",
            "Telugu discourse hesitation marker 'ante'",
            1600,
            1400,
        ),
        (
            "sample_14_tanglish_filler_seri_na",
            "Tanglish",
            "Naan online payment try pannen seri na",
            750,
            "server error nu display aagudhu",
            "seri na",
            "Tamil colloquial hesitation tag 'seri na'",
            1700,
            1300,
        ),
        (
            "sample_15_hinglish_filler_ummm",
            "Hinglish",
            "Delivery address update karna tha ummm",
            900,
            "Bangalore office ke naye location par",
            "ummm",
            "Phonetic vowel lengthening / vocalized hesitation 'ummm'",
            1500,
            1500,
        ),
        (
            "sample_16_hinglish_filler_actually",
            "Hinglish",
            "Branch ka working hours check karna tha actually",
            1000,
            "Saturday ko open rehta hai kya?",
            "actually",
            "English-Indic discourse marker 'actually' preceding 1000ms pause",
            1800,
            1400,
        ),
    ]

    for idx, (sid, lang, p1, pause_ms, p2, cue, note, sp1_ms, sp2_ms) in enumerate(cat_b_defs, start=11):
        sp1 = synthesize_speech_waveform(sp1_ms / 1000.0, f0_base=125.0 + (idx - 10) * 3.0)
        sil = synthesize_silence(pause_ms / 1000.0)
        sp2 = synthesize_speech_waveform(sp2_ms / 1000.0, f0_base=125.0 + (idx - 10) * 3.0)
        audio = np.concatenate([sp1, sil, sp2])

        wav_path = SAMPLES_DIR / f"{sid}.wav"
        write_wav(wav_path, audio)

        samples_meta.append({
            "id": f"sample_{idx:02d}",
            "sample_name": sid,
            "category": "B",
            "category_name": "COGNITIVE_HESITATION_AND_FILLERS",
            "language": lang,
            "file_path": str(wav_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "transcript_part1": p1,
            "pause_duration_ms": pause_ms,
            "transcript_part2": p2,
            "full_transcript": f"{p1} {p2}",
            "speech1_duration_ms": sp1_ms,
            "speech2_duration_ms": sp2_ms,
            "total_duration_ms": sp1_ms + pause_ms + sp2_ms,
            "linguistic_cue": cue,
            "bot_is_speaking": False,
            "description": note,
            "expected_standard_vad": "PREMATURE_CUTOFF_AT_500MS",
            "expected_vartalaap": "HOLD_FLOOR_DYNAMIC_ETS_CEILING_2000MS",
        })

    # ==========================================================================
    # CATEGORY C (6 Samples): Cultural Supportive Backchannels
    # ==========================================================================
    cat_c_defs = [
        (
            "sample_17_backchannel_haanji",
            "Hindi",
            "haanji",
            260,
            "Brief supportive backchannel 'haanji' during bot explanation",
        ),
        (
            "sample_18_backchannel_achha",
            "Hindi",
            "achha",
            290,
            "Attentive listening backchannel 'achha' during bot breakdown",
        ),
        (
            "sample_19_backchannel_seri",
            "Tamil",
            "seri",
            220,
            "Tamil acknowledgment 'seri' while bot reads OTP instructions",
        ),
        (
            "sample_20_backchannel_theek_hai",
            "Hindi",
            "theek hai",
            340,
            "Confirmation backchannel 'theek hai' while agent speaks",
        ),
        (
            "sample_21_backchannel_ante",
            "Telugu",
            "ante",
            270,
            "Telugu discourse token 'ante' as listener feedback",
        ),
        (
            "sample_22_backchannel_sari",
            "Kannada",
            "sari",
            250,
            "Kannada/Telugu affirmative token 'sari' during bot readout",
        ),
    ]

    for idx, (sid, lang, text, dur_ms, note) in enumerate(cat_c_defs, start=17):
        # 100ms silence + brief backchannel burst + 150ms trailing silence
        sil_pre = synthesize_silence(0.1)
        sp = synthesize_speech_waveform(dur_ms / 1000.0, f0_base=145.0, syllable_rate=4.5)
        sil_post = synthesize_silence(0.15)
        audio = np.concatenate([sil_pre, sp, sil_post])

        wav_path = SAMPLES_DIR / f"{sid}.wav"
        write_wav(wav_path, audio)

        total_ms = 100 + dur_ms + 150
        samples_meta.append({
            "id": f"sample_{idx:02d}",
            "sample_name": sid,
            "category": "C",
            "category_name": "CULTURAL_SUPPORTIVE_BACKCHANNEL",
            "language": lang,
            "file_path": str(wav_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "transcript_part1": text,
            "pause_duration_ms": 0,
            "transcript_part2": "",
            "full_transcript": text,
            "speech1_duration_ms": dur_ms,
            "speech2_duration_ms": 0,
            "total_duration_ms": total_ms,
            "linguistic_cue": text,
            "bot_is_speaking": True,
            "description": note,
            "expected_standard_vad": "FALSE_BARGE_IN_BOT_TERMINATION",
            "expected_vartalaap": "SOFT_DUCK_18DB_AND_RESTORE_ZERO_INTERRUPTION",
        })

    # ==========================================================================
    # CATEGORY D (4 Samples): Genuine Imperative Barge-ins
    # ==========================================================================
    cat_d_defs = [
        (
            "sample_23_bargein_ruko_ruko",
            "Hindi",
            "ruko ruko",
            460,
            "Urgent imperative interruption command 'ruko ruko'",
        ),
        (
            "sample_24_bargein_cancel_karo",
            "Hinglish",
            "cancel karo",
            520,
            "Imperative transaction cancellation command 'cancel karo'",
        ),
        (
            "sample_25_bargein_stop_immediately",
            "English",
            "stop immediately",
            580,
            "Direct urgent cessation command 'stop immediately'",
        ),
        (
            "sample_26_bargein_wait_suno",
            "Hinglish",
            "wait suno",
            490,
            "Attention-grabbing interruption command 'wait suno'",
        ),
    ]

    for idx, (sid, lang, text, dur_ms, note) in enumerate(cat_d_defs, start=23):
        sil_pre = synthesize_silence(0.08)
        sp = synthesize_speech_waveform(dur_ms / 1000.0, f0_base=155.0, syllable_rate=5.0)
        sil_post = synthesize_silence(0.2)
        audio = np.concatenate([sil_pre, sp, sil_post])

        wav_path = SAMPLES_DIR / f"{sid}.wav"
        write_wav(wav_path, audio)

        total_ms = 80 + dur_ms + 200
        samples_meta.append({
            "id": f"sample_{idx:02d}",
            "sample_name": sid,
            "category": "D",
            "category_name": "GENUINE_IMPERATIVE_BARGE_IN",
            "language": lang,
            "file_path": str(wav_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "transcript_part1": text,
            "pause_duration_ms": 0,
            "transcript_part2": "",
            "full_transcript": text,
            "speech1_duration_ms": dur_ms,
            "speech2_duration_ms": 0,
            "total_duration_ms": total_ms,
            "linguistic_cue": text,
            "bot_is_speaking": True,
            "description": note,
            "expected_standard_vad": "GENUINE_INTERRUPT_TRIGGERED",
            "expected_vartalaap": "SUB_60MS_HARD_FLUSH_CANCELLATION",
        })

    # ==========================================================================
    # CATEGORY E (4 Samples): Clean Syntactic Terminal Closures
    # ==========================================================================
    cat_e_defs = [
        (
            "sample_27_terminal_balance_batao",
            "Hindi",
            "Mera current balance batao",
            "batao",
            1200,
            "Imperative terminal closure with verb 'batao' (-0.65 penalty)",
        ),
        (
            "sample_28_terminal_haan_confirm_hai",
            "Hinglish",
            "Haan confirm hai",
            "confirm hai",
            1000,
            "Affirmative terminal completion 'confirm hai' (-0.75 penalty)",
        ),
        (
            "sample_29_terminal_ticket_book_kar_do",
            "Hinglish",
            "Ticket book kar do",
            "kar do",
            1100,
            "Direct action terminal closure 'kar do' (-0.65 penalty)",
        ),
        (
            "sample_30_terminal_theek_hai_shukriya",
            "Urdu/Hindi",
            "Theek hai shukriya",
            "shukriya",
            950,
            "Gratitude closure token 'shukriya' (-0.75 penalty)",
        ),
    ]

    for idx, (sid, lang, text, cue, dur_ms, note) in enumerate(cat_e_defs, start=27):
        sp = synthesize_speech_waveform(dur_ms / 1000.0, f0_base=120.0, syllable_rate=3.6)
        sil_post = synthesize_silence(0.6)  # 600ms of post-utterance silence to measure turnaround
        audio = np.concatenate([sp, sil_post])

        wav_path = SAMPLES_DIR / f"{sid}.wav"
        write_wav(wav_path, audio)

        total_ms = dur_ms + 600
        samples_meta.append({
            "id": f"sample_{idx:02d}",
            "sample_name": sid,
            "category": "E",
            "category_name": "CLEAN_SYNTACTIC_TERMINAL_CLOSURE",
            "language": lang,
            "file_path": str(wav_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "transcript_part1": text,
            "pause_duration_ms": 600,
            "transcript_part2": "",
            "full_transcript": text,
            "speech1_duration_ms": dur_ms,
            "speech2_duration_ms": 0,
            "total_duration_ms": total_ms,
            "linguistic_cue": cue,
            "bot_is_speaking": False,
            "description": note,
            "expected_standard_vad": "FIXED_500MS_DEAD_AIR_LATENCY",
            "expected_vartalaap": "FAST_DEAD_AIR_TURNAROUND_SUB_350MS",
        })

    manifest = {
        "version": "2.0.0",
        "benchmark_suite": "Project Vartalaap Two-Tier Benchmark",
        "sample_rate_hz": 16000,
        "format": "WAV 16-bit Linear PCM Mono",
        "total_samples": len(samples_meta),
        "categories": {
            "A": {
                "name": "Mid-Utterance Code-Mixing Pauses",
                "count": 10,
                "expected": "Standard VAD prematurely cuts off at 500ms; Vartalaap holds turn (>1200ms ETS)",
            },
            "B": {
                "name": "Cognitive Hesitation and Fillers",
                "count": 6,
                "expected": "Standard VAD cuts off; Vartalaap expands ETS up to 2000ms",
            },
            "C": {
                "name": "Cultural Supportive Backchannels",
                "count": 6,
                "expected": "Standard VAD falsely interrupts bot; Vartalaap ducks -18dB and restores 0dB",
            },
            "D": {
                "name": "Genuine Imperative Barge-ins",
                "count": 4,
                "expected": "Both interrupt, Vartalaap achieves sub-60ms hard flush cancellation",
            },
            "E": {
                "name": "Clean Syntactic Terminal Closures",
                "count": 4,
                "expected": "Vartalaap reduces dead-air turnaround to <350ms vs Standard 500ms",
            },
        },
        "samples": samples_meta,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Generated %d stress test WAV audio files in %s", len(samples_meta), SAMPLES_DIR)
    logger.info("Manifest written to %s", MANIFEST_PATH)
    return manifest


if __name__ == "__main__":
    generate_all_samples()
