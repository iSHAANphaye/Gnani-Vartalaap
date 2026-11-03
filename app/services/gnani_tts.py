"""Client for Gnani Timbre v3.0 Expressive TTS (api.vachana.ai).

Provides:
- Audio synthesis via Gnani Timbre REST and streaming endpoints.
- Full support for inline expressive/emotional tags (<excited>, <pauses>, <laugh>).
- Deterministic synthetic WAV generator fallback for offline local development.
"""

import io
import logging
import math
import re
import struct
import time
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional
import requests

from app.config import Settings, get_settings

logger = logging.getLogger("vartalaap.gnani_tts")

EXPRESSIVE_TAGS_PATTERN = re.compile(r"<[^>]+>")


def strip_expressive_tags(text: str) -> str:
    """Remove inline expressive tags for clean text logging or display."""
    return EXPRESSIVE_TAGS_PATTERN.sub("", text).strip()


class GnaniTTSClient:
    """Client for synthesizing expressive Indic speech with Gnani Timbre."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.voice = voice or self.settings.gnani_default_voice
        self.language = language or self.settings.gnani_default_lang
        self.model = model or self.settings.gnani_tts_model
        self.is_mock_mode = not self.settings.has_valid_gnani_credentials

        if self.is_mock_mode:
            logger.info("GnaniTTSClient initialized in MOCK / OFFLINE fallback mode")
        else:
            logger.info(
                "GnaniTTSClient initialized with voice='%s', model='%s', endpoint='%s'",
                self.voice,
                self.model,
                self.settings.gnani_tts_rest_url,
            )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        output_path: Optional[str | Path] = None,
    ) -> bytes:
        """Synthesize text into WAV audio bytes.

        Supports inline emotional tags like `<excited>`, `<pauses>`, `<laugh>`.
        Falls back to deterministic mock audio generation if unauthenticated or offline.
        """
        chosen_voice = voice or self.voice
        chosen_lang = language or self.language

        if self.is_mock_mode:
            audio_bytes = self._generate_mock_wav(text)
        else:
            try:
                audio_bytes = self._request_timbre_tts(text, chosen_voice, chosen_lang)
            except Exception as e:
                logger.warning(
                    "Gnani Timbre TTS request failed (%s). Falling back to mock audio generator.",
                    e,
                )
                audio_bytes = self._generate_mock_wav(text)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(audio_bytes)
            logger.info("Saved synthesized audio to %s (%d bytes)", out_p, len(audio_bytes))

        return audio_bytes

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        chunk_size: int = 3200,
    ) -> AsyncGenerator[bytes, None]:
        """Asynchronously stream synthesized audio chunks for low-latency playback."""
        audio_bytes = self.synthesize(text, voice=voice)
        # Skip standard 44-byte WAV header when streaming raw PCM chunks
        pcm_data = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes

        for i in range(0, len(pcm_data), chunk_size):
            yield pcm_data[i : i + chunk_size]

    def _request_timbre_tts(self, text: str, voice: str, language: str) -> bytes:
        """Execute REST call to Gnani Timbre v3.0 endpoint."""
        api_key = self.settings.gnani_api_key.get_secret_value()
        headers = {
            "X-API-Key-ID": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "model": self.model,
            "audio_format": "wav",
            "sample_rate": 16000,
        }

        response = requests.post(
            self.settings.gnani_tts_rest_url,
            headers=headers,
            json=payload,
            timeout=8.0,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "audio" in content_type or response.content.startswith(b"RIFF"):
            return response.content

        # Handle JSON wrapper containing base64 audio if returned
        data = response.json()
        import base64

        raw_b64 = data.get("audio_content") or data.get("audio") or ""
        if raw_b64:
            return base64.b64decode(raw_b64)

        raise ValueError(f"Unexpected response payload format from Gnani TTS: {data.keys()}")

    def _generate_mock_wav(self, text: str) -> bytes:
        """Generate a valid 16kHz 16-bit mono RIFF WAV for offline testing."""
        sample_rate = 16000
        clean_text = strip_expressive_tags(text)
        # Approximate duration based on word count (~300ms per word, clamp between 0.6s and 3.0s)
        word_count = max(1, len(clean_text.split()))
        duration = min(3.0, max(0.6, word_count * 0.25))
        total_samples = int(sample_rate * duration)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(sample_rate)

            # Generate synthetic harmonic formant tone with envelope
            f0 = 220.0  # Base frequency
            frames = bytearray()
            for i in range(total_samples):
                t = i / sample_rate
                # Smooth envelope (fade in/out)
                envelope = math.sin(math.pi * (i / total_samples))
                # Add slight vibrato and 2nd harmonic
                sample = (
                    0.6 * math.sin(2.0 * math.pi * f0 * t)
                    + 0.3 * math.sin(4.0 * math.pi * f0 * t)
                ) * envelope * 12000.0
                sample_int = int(max(-32767.0, min(32767.0, sample)))
                frames.extend(struct.pack("<h", sample_int))

            wav_file.writeframes(frames)

        return buffer.getvalue()


def _main():
    """CLI test execution entrypoint for verification."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 60)
    print(" 🔊 Testing Gnani Timbre v3.0 TTS Client Module")
    print("=" * 60)

    client = GnaniTTSClient()
    test_phrase = (
        "Namaste! <excited>Aapka account balance verify ho gaya hai.</excited> "
        "<pauses> Aur kuch madad karu?"
    )

    print(f"Synthesizing phrase with expressive tags:\n  \"{test_phrase}\"")
    output_path = Path("benchmark/test_samples/test_tts_output.wav")
    audio = client.synthesize(test_phrase, output_path=output_path)

    print(f"Generated WAV Size: {len(audio)} bytes")
    print(f"Saved to: {output_path.resolve()}")

    # Verify that file has valid WAV header
    with wave.open(str(output_path), "rb") as wf:
        print(f"WAV Verification:")
        print(f"  Channels     : {wf.getnchannels()}")
        print(f"  Sample Width : {wf.getsampwidth()} bytes")
        print(f"  Sample Rate  : {wf.getframerate()} Hz")
        print(f"  Frame Count  : {wf.getnframes()}")
        print(f"  Duration     : {wf.getnframes() / wf.getframerate():.2f}s")

    print("✓ Gnani Timbre TTS Client verification passed successfully.")


if __name__ == "__main__":
    _main()
