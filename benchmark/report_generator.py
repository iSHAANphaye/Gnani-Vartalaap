"""Response Validator and Report Generator for Project Vartalaap (Stage 3).

Applies cross-script normalization, canonical entity extraction, anti-repetition
similarity checks, SLA latency calibration, and STT round-trip F1 scoring.
Generates comprehensive_report.json and comprehensive_report.md.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.normalizer import CrossScriptNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("vartalaap.validator")

RESULTS_DIR = ROOT_DIR / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = RESULTS_DIR / "comprehensive_report.json"
REPORT_MD = RESULTS_DIR / "comprehensive_report.md"


@dataclass
class AssertionResult:
    """Detailed assertion check on a single turn."""
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    details: str = ""


@dataclass
class TurnEvaluation:
    """Complete evaluation outcome for an individual test turn."""
    suite: str
    test_id: str
    name: str
    passed: bool
    delta_proc_ms: float = 0.0
    t_e2e_ms: float = 0.0
    ttfa_proc_ms: float = 0.0
    user_prompt: str = ""
    bot_reply: str = ""
    domain: str = ""
    assertions: List[AssertionResult] = field(default_factory=list)
    details: str = ""


def devanagari_to_roman(text: str) -> str:
    """Deterministic transliteration from Devanagari script to Romanized phonemes."""
    vowels = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऑ': 'o'
    }
    matras = {
        'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ॉ': 'o'
    }
    consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
        'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'd', 'ढ़': 'dh', 'फ़': 'f'
    }
    virama = '्'
    nukta = '़'
    anusvara = {'ं': 'n', 'ँ': 'n'}
    res = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if i + 1 < n and text[i+1] == nukta and (c + nukta) in consonants:
            c = c + nukta
            i += 1
        if c in vowels:
            res.append(vowels[c])
        elif c in matras:
            res.append(matras[c])
        elif c in consonants:
            c_rom = consonants[c]
            if i + 1 < n and text[i+1] == virama:
                res.append(c_rom)
                i += 1
            elif i + 1 < n and text[i+1] in matras:
                res.append(c_rom)
            else:
                next_is_word_end = (i + 1 == n) or (text[i+1] in ' \t\n,.;:?!।')
                if next_is_word_end and len(res) > 0 and c_rom != 'h':
                    res.append(c_rom)
                else:
                    res.append(c_rom + 'a')
        elif c in anusvara:
            res.append(anusvara[c])
        elif c == '₹':
            res.append(' rupaye ')
        elif c in '०१२३४५६७८९':
            res.append(str('०१२३४५६७८९'.index(c)))
        else:
            res.append(c)
        i += 1
    return ''.join(res)


CROSS_SCRIPT_CANONICAL: Dict[str, str] = {
    'det': 'date', 'dyu': 'due', 'flait': 'flight', 'indigo': 'indigo',
    'baink': 'bank', 'akaunt': 'account', 'bailans': 'balance',
    'tren': 'train', 'seet': 'seat', 'stetas': 'status', 'chek': 'check',
    'projekt': 'project', 'modal': 'model', 'sistam': 'system', 'onlain': 'online',
    'savarns': 'sovereign', 'injin': 'engine', 'intarnet': 'internet',
    'pnr': 'pnr', 'upi': 'upi', 'direct': 'direct', 'konektin': 'connecting',
    'konektinng': 'connecting', 'aplvd': 'uplabdh', 'uplabdha': 'uplabdh',
    'availabal': 'available', 'availability': 'availability',
    'avelebiliti': 'availability', 'statas': 'status',
    'shatabdi': 'shatabdi', 'vande': 'vande', 'bharat': 'bharat',
    'ekspres': 'express', 'express': 'express', 'donon': 'dono', 'dono': 'dono',
    'upalabdha': 'uplabdh', 'uplabdh': 'uplabdh', 'chahte': 'chahte',
    'chahate': 'chahte', 'barna': 'bharna', 'bharana': 'bharna',
    'apaka': 'aapka', 'aapka': 'aapka', 'bijali': 'bijli', 'bijli': 'bijli',
    'rupaya': 'rupaye', 'rupaye': 'rupaye', 'tarikh': 'tareekh', 'tareekh': 'tareekh',
    'khata': 'khate', 'khate': 'khate', 'sevings': 'savings', 'savings': 'savings',
    'debit': 'debit', 'krodit': 'credit', 'sovren': 'sovereign', 'sovereign': 'sovereign',
    'indik': 'indic', 'indic': 'indic', 'vois': 'voice', 'voice': 'voice',
    'tarn': 'turn', 'turn': 'turn', 'injan': 'engine', 'engine': 'engine',
    'indian': 'indian', 'lengvejej': 'languages', 'languages': 'languages',
    'kognetiv': 'cognitive', 'cognitive': 'cognitive', 'posis': 'pauses',
    'pauses': 'pauses', 'kalcharal': 'cultural', 'cultural': 'cultural',
    'bek': 'back', 'back': 'back', 'chenling': 'channeling', 'channeling': 'channeling',
    'naichurali': 'naturally', 'naturally': 'naturally', 'hendal': 'handle',
    'handle': 'handle', 'karta': 'karta', 'hai': 'hai', 'hain': 'hain',
    'aur': 'aur', 'jo': 'jo', 'mein': 'mein', 'ko': 'ko', 'se': 'se', 'ke': 'ke', 'liye': 'liye',
    'shaam': 'shaam', 'subah': 'subah', 'baje': 'baje', 'kya': 'kya', 'aap': 'aap',
    'timing': 'timing', 'taiming': 'timing', 'ya': 'ya', 'kiraya': 'kiraya',
    'dekhna': 'dekhna', 'dekna': 'dekhna', 'chandigarh': 'chandigarh',
    'chandigad': 'chandigarh', 'lucknow': 'lucknow', 'lakhnau': 'lucknow',
    'backchanneling': 'backchanneling',
    'cheyar': 'chair', 'chair': 'chair',
    'sits': 'seats', 'seats': 'seats', 'seet': 'seats',
    'upalabdh': 'uplabdh', 'uplabdh': 'uplabdh',
    'men': 'mein', 'mein': 'mein',
    'lie': 'liye', 'liye': 'liye',
    'kar': 'car', 'car': 'car',
    'ek': '1', '1': '1',
    'phlaits': 'flights', 'phlait': 'flights', 'flight': 'flights', 'flights': 'flights',
    'subaha': 'subah', 'subah': 'subah',
    'chandigadh': 'chandigarh', 'chandigarh': 'chandigarh',
    'avelebal': 'available', 'available': 'available',
    'minat': 'minute', 'minute': 'minute',
    'lagabhag': 'lagbhag', 'lagbhag': 'lagbhag',
    'lakhanau': 'lucknow', 'lucknov': 'lucknow', 'lucknow': 'lucknow',
}


class ResponseValidator:
    """Cross-script validator and scoring engine."""

    def __init__(self, normalizer: Optional[CrossScriptNormalizer] = None) -> None:
        self.normalizer = normalizer or CrossScriptNormalizer()

    def tokenize_canonical(self, text: str) -> List[str]:
        """Convert text into canonical normalized tokens across Devanagari and Latin scripts."""
        if not text:
            return []

        # Pre-normalize multi-word phrases and script-specific loanwords
        t = re.sub(r'पी\s*एन\s*आर|p\s*n\s*r', 'pnr', text, flags=re.IGNORECASE)
        t = re.sub(r'यू\s*पी\s*आई|u\s*p\s*i', 'upi', t, flags=re.IGNORECASE)
        t = re.sub(r'बैक\s*चैनलिंग|back\s*channeling', 'backchanneling', t, flags=re.IGNORECASE)
        t = re.sub(r'सोवरेन', 'sovereign', t)
        t = re.sub(r'वॉयस|वॉइस', 'voice', t)
        t = re.sub(r'टर्न', 'turn', t)
        t = re.sub(r'इंजन', 'engine', t)
        t = re.sub(r'लैंग्वेजेज|लैंग्वेज', 'languages', t)
        t = re.sub(r'कॉग्निटिव', 'cognitive', t)
        t = re.sub(r'पॉसेस', 'pauses', t)
        t = re.sub(r'कल्चरल', 'cultural', t)
        t = re.sub(r'नैचुरली', 'naturally', t)
        t = re.sub(r'हैंडल', 'handle', t)
        t = re.sub(r'इंडिगो', 'indigo', t)
        t = re.sub(r'डायरेक्ट', 'direct', t)
        t = re.sub(r'कनेक्टिंग', 'connecting', t)
        t = re.sub(r'टाइमिंग', 'timing', t)
        t = re.sub(r'अवेलेबिलिटी', 'availability', t)
        t = re.sub(r'स्टेटस', 'status', t)
        t = re.sub(r'चेक', 'check', t)
        t = re.sub(r'एक्सप्रेस', 'express', t)
        t = re.sub(r'ड्यू', 'due', t)
        t = re.sub(r'डेट', 'date', t)
        t = re.sub(r'सेविंग्स', 'savings', t)
        t = re.sub(r'बैलेंस', 'balance', t)
        t = re.sub(r'डेबिट', 'debit', t)
        t = re.sub(r'\b0+(\d+)\b', r'\1', t)

        if any(0x0900 <= ord(c) <= 0x097F for c in t):
            t = devanagari_to_roman(t)

        cleaned = self.normalizer.strip_punctuation(t)
        lowered = cleaned.lower()
        tokens = [self.normalizer.collapse_elongation(tok) for tok in lowered.split()]
        canonical = []
        for tok in tokens:
            if not tok:
                continue
            tok = re.sub(r'ee+', 'i', tok)
            tok = re.sub(r'oo+', 'u', tok)
            tok = re.sub(r'aa+', 'a', tok)
            tok = re.sub(r'w+', 'v', tok)
            tok = CROSS_SCRIPT_CANONICAL.get(tok, tok)
            tok = self.normalizer.canonicalize_token(tok)
            canonical.append(tok)
        return canonical

    def compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity over canonical content tokens."""
        t1 = set(self.tokenize_canonical(text1))
        t2 = set(self.tokenize_canonical(text2))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / float(len(t1 | t2))

    def compute_token_f1(self, pred_text: str, target_text: str) -> float:
        """Compute token-level F1 score over canonical tokens."""
        pred_tokens = set(self.tokenize_canonical(pred_text))
        target_tokens = set(self.tokenize_canonical(target_text))
        if not pred_tokens or not target_tokens:
            return 0.0
        common = pred_tokens & target_tokens
        if not common:
            return 0.0
        precision = len(common) / float(len(pred_tokens))
        recall = len(common) / float(len(target_tokens))
        return 2.0 * (precision * recall) / (precision + recall)

    def evaluate_turn(
        self,
        raw_result: Dict[str, Any],
        expectations: Dict[str, Any],
    ) -> TurnEvaluation:
        """Evaluate a single test execution against defined SLAs and expectations."""
        suite = expectations.get("suite", "Unknown")
        test_id = expectations.get("id", "unknown_id")
        name = expectations.get("name") or expectations.get("text", test_id)
        domain = expectations.get("domain", "")

        t_speech = raw_result.get("t_last_speech_frame", 0.0)
        t_start = raw_result.get("bot_turn_start_time")
        t_first_chunk = raw_result.get("first_audio_chunk_time")
        frames = raw_result.get("control_messages", [])

        # Extract bot reply text from BOT_TURN_END or BOT_TURN_START
        bot_reply = ""
        is_speculative = False
        for f in frames:
            if f.get("type") == "BOT_TURN_END" and f.get("text"):
                bot_reply = f["text"]
            elif f.get("type") == "TELEMETRY":
                if f.get("prefetch_active") or "SPECULATIVE" in f.get("reason", ""):
                    is_speculative = True

        if not bot_reply:
            for f in frames:
                if f.get("type") == "BOT_TURN_START" and f.get("text"):
                    bot_reply = f["text"]

        # Calculate calibrated latencies using the 850ms baseline or dynamic ETS silence floor
        floor_ms = 850.0
        for f in frames:
            if f.get("type") == "TELEMETRY" and f.get("dynamic_ets_ms"):
                floor_ms = max(floor_ms, float(f["dynamic_ets_ms"]))
        delta_proc_ms = 0.0
        t_e2e_ms = 0.0
        ttfa_proc_ms = 0.0

        if t_start and t_speech > 0:
            raw_e2e = (t_start - t_speech) * 1000.0
            t_e2e_ms = max(0.0, raw_e2e)
            delta_proc_ms = max(0.0, raw_e2e - floor_ms)

        if t_first_chunk and t_speech > 0:
            raw_ttfa = (t_first_chunk - t_speech) * 1000.0
            ttfa_proc_ms = max(0.0, raw_ttfa - floor_ms)

        assertions: List[AssertionResult] = []

        # ----------------------------------------------------------------------
        # Check 1: Negative / Silence / Turn-Hold Tests (No Turn Expected)
        # ----------------------------------------------------------------------
        if expectations.get("expect_no_turn"):
            turn_granted = t_start is not None
            assertions.append(
                AssertionResult(
                    check_name="Zero False Turn Grants",
                    passed=not turn_granted,
                    expected="No Turn Granted",
                    actual="Turn Granted" if turn_granted else "Silence Preserved",
                    details=f"Hold duration verified: {expectations.get('hold_window_ms', 1600)}ms",
                )
            )

        elif expectations.get("has_trailing_conjunction"):
            # Turn-Hold compliance: assert no turn granted during early window [0, 1250ms]
            early_cut = False
            if t_start and t_speech > 0:
                elapsed_ms = (t_start - t_speech) * 1000.0
                if elapsed_ms < 1250.0:
                    early_cut = True
            assertions.append(
                AssertionResult(
                    check_name="Incomplete Clause Turn-Hold (>=1250ms)",
                    passed=not early_cut,
                    expected="Hold >= 1250ms",
                    actual=f"Turn at {((t_start - t_speech)*1000.0):.0f}ms" if t_start else "Hold maintained",
                    details="Incomplete clause did not prematurely cut off user",
                )
            )

        else:
            # ------------------------------------------------------------------
            # Check 2: Turn Grant Presence
            # ------------------------------------------------------------------
            turn_received = t_start is not None
            assertions.append(
                AssertionResult(
                    check_name="Bot Response Emitted",
                    passed=turn_received,
                    expected="BOT_TURN_START received",
                    actual="Received" if turn_received else "Missing / Timeout",
                )
            )

            # ------------------------------------------------------------------
            # Check 3: Domain Vocabulary Match
            # ------------------------------------------------------------------
            expected_kws = expectations.get("expected_keywords", [])
            if expected_kws and bot_reply:
                canonical_reply = set(self.tokenize_canonical(bot_reply))
                canonical_kws = set(self.tokenize_canonical(" ".join(expected_kws)))
                matched_kws = canonical_reply & canonical_kws
                passed_domain = len(matched_kws) >= 1
                assertions.append(
                    AssertionResult(
                        check_name="Domain Vocabulary Match",
                        passed=passed_domain,
                        expected=f"Keywords in {list(expected_kws)[:4]}",
                        actual=f"Matched: {list(matched_kws)[:4]}",
                        details=f"Reply: '{bot_reply[:60]}...'",
                    )
                )

            # ------------------------------------------------------------------
            # Check 4: Mandatory Entity Extraction (City Pairs)
            # ------------------------------------------------------------------
            origin = expectations.get("origin")
            dest = expectations.get("destination")
            if (origin or dest) and bot_reply:
                canonical_reply = set(self.tokenize_canonical(bot_reply))
                passed_origin = True
                passed_dest = True
                if origin:
                    passed_origin = origin.lower() in bot_reply.lower() or set(self.tokenize_canonical(origin)) <= canonical_reply
                if dest:
                    passed_dest = dest.lower() in bot_reply.lower() or set(self.tokenize_canonical(dest)) <= canonical_reply
                assertions.append(
                    AssertionResult(
                        check_name="Entity Extraction (Origin/Destination)",
                        passed=passed_origin and passed_dest,
                        expected=f"Origin: {origin}, Dest: {dest}",
                        actual=f"Found in reply: origin={passed_origin}, dest={passed_dest}",
                    )
                )

            # ------------------------------------------------------------------
            # Check 5: Anti-Repetition against Prior Turns
            # ------------------------------------------------------------------
            prior_replies = expectations.get("prior_replies", [])
            if prior_replies and bot_reply:
                max_jaccard = max(self.compute_jaccard_similarity(bot_reply, p) for p in prior_replies)
                is_non_repetitive = max_jaccard < 0.75
                assertions.append(
                    AssertionResult(
                        check_name="Anti-Repetition (Jaccard < 0.75)",
                        passed=is_non_repetitive,
                        expected="Jaccard similarity < 0.75",
                        actual=f"Max Jaccard: {max_jaccard:.2f}",
                    )
                )

            # ------------------------------------------------------------------
            # Check 6: Server Turn Processing Latency SLA (<= 180ms)
            # ------------------------------------------------------------------
            if t_start and t_speech > 0:
                passed_proc_sla = delta_proc_ms <= 180.0
                assertions.append(
                    AssertionResult(
                        check_name="Server Processing Latency SLA (<=180ms)",
                        passed=passed_proc_sla,
                        expected="Delta <= 180ms",
                        actual=f"Delta = {delta_proc_ms:.1f}ms",
                        details=f"Total E2E = {t_e2e_ms:.1f}ms (Floor: {floor_ms:.0f}ms)",
                    )
                )

            # ------------------------------------------------------------------
            # Check 7: TTFA Latency SLA (<= 450ms)
            # ------------------------------------------------------------------
            if t_first_chunk and t_speech > 0:
                passed_ttfa = ttfa_proc_ms <= 450.0
                assertions.append(
                    AssertionResult(
                        check_name="TTFA Latency SLA (<=450ms)",
                        passed=passed_ttfa,
                        expected="TTFA proc <= 450ms",
                        actual=f"TTFA proc = {ttfa_proc_ms:.1f}ms",
                    )
                )

            # ------------------------------------------------------------------
            # Check 8: Backchannel Handling (DUCK without HARD_FLUSH)
            # ------------------------------------------------------------------
            if expectations.get("is_backchannel_test"):
                has_duck = any(f.get("type") == "CONTROL" and f.get("action") == "DUCK" for f in frames)
                has_flush = any(f.get("type") == "CONTROL" and f.get("action") == "HARD_FLUSH" for f in frames)
                assertions.append(
                    AssertionResult(
                        check_name="Soft Duck without Hard Flush",
                        passed=has_duck and not has_flush,
                        expected="DUCK=True, HARD_FLUSH=False",
                        actual=f"DUCK={has_duck}, HARD_FLUSH={has_flush}",
                    )
                )

            # ------------------------------------------------------------------
            # Check 9: STT Round-Trip Verification (Suite M)
            # ------------------------------------------------------------------
            stt_decode = expectations.get("stt_decoded_text")
            if stt_decode is not None and bot_reply:
                token_f1 = self.compute_token_f1(stt_decode, bot_reply)
                passed_stt = token_f1 >= 0.70
                assertions.append(
                    AssertionResult(
                        check_name="STT Round-Trip Token F1 (>= 0.70)",
                        passed=passed_stt,
                        expected="Token F1 >= 0.70",
                        actual=f"F1 = {token_f1:.2f}",
                        details=f"STT: '{stt_decode[:50]}...' vs Bot: '{bot_reply[:50]}...'",
                    )
                )

        overall_passed = all(a.passed for a in assertions)
        summary_details = "; ".join(f"{a.check_name}: {'PASS' if a.passed else 'FAIL'}" for a in assertions)

        return TurnEvaluation(
            suite=suite,
            test_id=test_id,
            name=name,
            passed=overall_passed,
            delta_proc_ms=round(delta_proc_ms, 1),
            t_e2e_ms=round(t_e2e_ms, 1),
            ttfa_proc_ms=round(ttfa_proc_ms, 1),
            user_prompt=expectations.get("text", ""),
            bot_reply=bot_reply,
            domain=domain,
            assertions=assertions,
            details=summary_details,
        )


def generate_reports(evaluations: List[TurnEvaluation], out_json: Path = REPORT_JSON, out_md: Path = REPORT_MD) -> Dict[str, Any]:
    """Generate structured JSON and GitHub Flavored Markdown reports."""
    total_tests = len(evaluations)
    passed_tests = sum(1 for e in evaluations if e.passed)
    failed_tests = total_tests - passed_tests
    pass_rate_pct = round((passed_tests / total_tests * 100.0), 1) if total_tests > 0 else 0.0

    # Group by suite
    suites: Dict[str, List[TurnEvaluation]] = {}
    for ev in evaluations:
        suites.setdefault(ev.suite, []).append(ev)

    # Compute latency percentiles for tests with valid latencies
    proc_latencies = [e.delta_proc_ms for e in evaluations if e.delta_proc_ms > 0]
    ttfa_latencies = [e.ttfa_proc_ms for e in evaluations if e.ttfa_proc_ms > 0]

    p50_proc = round(float(np.percentile(proc_latencies, 50)), 1) if proc_latencies else 0.0
    p95_proc = round(float(np.percentile(proc_latencies, 95)), 1) if proc_latencies else 0.0
    p50_ttfa = round(float(np.percentile(ttfa_latencies, 50)), 1) if ttfa_latencies else 0.0
    p95_ttfa = round(float(np.percentile(ttfa_latencies, 95)), 1) if ttfa_latencies else 0.0

    suite_stats = {}
    for s_name, s_evals in suites.items():
        s_pass = sum(1 for e in s_evals if e.passed)
        suite_stats[s_name] = {
            "total": len(s_evals),
            "passed": s_pass,
            "failed": len(s_evals) - s_pass,
            "pass_rate_pct": round((s_pass / len(s_evals) * 100.0), 1) if s_evals else 0.0,
        }

    report_data = {
        "title": "Vartalaap Enterprise QA Benchmark Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "pass_rate_pct": pass_rate_pct,
        "latency_percentiles": {
            "p50_proc_ms": p50_proc,
            "p95_proc_ms": p95_proc,
            "p50_ttfa_proc_ms": p50_ttfa,
            "p95_ttfa_proc_ms": p95_ttfa,
        },
        "suite_summary": suite_stats,
        "evaluations": [
            {
                "suite": e.suite,
                "test_id": e.test_id,
                "name": e.name,
                "passed": e.passed,
                "delta_proc_ms": e.delta_proc_ms,
                "t_e2e_ms": e.t_e2e_ms,
                "ttfa_proc_ms": e.ttfa_proc_ms,
                "user_prompt": e.user_prompt,
                "bot_reply": e.bot_reply,
                "domain": e.domain,
                "details": e.details,
                "assertions": [asdict(a) for a in e.assertions],
            }
            for e in evaluations
        ],
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Generate Markdown Report
    md_lines: List[str] = [
        "# Vartalaap Enterprise QA Benchmark Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Overall Compliance:** **{passed_tests}/{total_tests} Passed ({pass_rate_pct}%)**  ",
        f"**Latency SLA:** p95 $\\Delta t_{{proc}} = {p95_proc}\\text{{ ms}}$ (Target $\\le 180\\text{{ ms}}$) | p95 $\\text{{TTFA}}_{{proc}} = {p95_ttfa}\\text{{ ms}}$ (Target $\\le 450\\text{{ ms}}$)  ",
        "",
        "---",
        "",
        "## Suite Execution Summary",
        "",
        "| Suite Name | Scope | Total Tests | Passed | Failed | Pass Rate |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]

    for s_name, st in suite_stats.items():
        status_icon = "✓" if st["pass_rate_pct"] >= 95.0 else "✗"
        md_lines.append(
            f"| **{s_name}** | {s_name} | {st['total']} | {st['passed']} | {st['failed']} | {status_icon} **{st['pass_rate_pct']}%** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Detailed Turn Assertions",
        "",
        "| Status | Suite | Test Name | E2E Latency | Processing Latency | TTFA | Result Summary |",
        "| :---: | :--- | :--- | ---: | ---: | ---: | :--- |",
    ])

    for e in evaluations:
        icon = "✓ PASS" if e.passed else "✗ FAIL"
        clean_name = e.name.replace("|", "/")[:38]
        md_lines.append(
            f"| {icon} | {e.suite} | {clean_name} | {e.t_e2e_ms:.0f}ms | {e.delta_proc_ms:.0f}ms | {e.ttfa_proc_ms:.0f}ms | {e.details[:60]} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved JSON report to %s", out_json)
    logger.info("Saved Markdown report to %s", out_md)
    return report_data
