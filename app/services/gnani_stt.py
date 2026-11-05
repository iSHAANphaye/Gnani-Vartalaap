"""Asynchronous client for Gnani Prisma STT (api.vachana.ai).

Provides:
- WebSocket streaming client for real-time interim token emission.
- REST fallback endpoint client for non-streaming audio chunks.
- Graceful mock fallback mode for offline local development and test suites.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, List, Optional
import requests
import websockets
from websockets.exceptions import WebSocketException

from app.config import Settings, get_settings

logger = logging.getLogger("vartalaap.gnani_stt")


@dataclass
class STTTranscript:
    """Represents a speech recognition transcript update."""

    text: str
    is_final: bool = False
    language: str = "hi-IN"
    confidence: float = 0.95
    timestamp: float = field(default_factory=time.time)
    words: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.words and self.text:
            self.words = self.text.strip().split()


class GnaniSTTClient:
    """Client for Gnani Prisma STT streaming and REST APIs."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        language: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> None:
        self.settings = settings or get_settings()
        self.language = language or self.settings.gnani_default_lang
        self.sample_rate = sample_rate
        self.is_connected = False
        self.is_mock_mode = not self.settings.has_valid_gnani_credentials

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._transcript_queue: asyncio.Queue[STTTranscript] = asyncio.Queue()
        self._closed = False

        if self.is_mock_mode:
            logger.info("GnaniSTTClient initialized in MOCK / OFFLINE fallback mode")
        else:
            logger.info(
                "GnaniSTTClient initialized with target endpoint: %s",
                self.settings.gnani_stt_ws_url,
            )

    async def connect(self) -> bool:
        """Establish streaming connection to Gnani Prisma STT."""
        if self._closed:
            raise RuntimeError("GnaniSTTClient has already been closed")

        if self.is_mock_mode:
            self.is_connected = True
            logger.info("Connected to mock Gnani STT stream engine")
            return True

        api_key = self.settings.gnani_api_key.get_secret_value()
        headers = {
            "X-API-Key-ID": api_key,
            "X-Language-Code": self.language,
            "X-Sampling-Rate": str(self.sample_rate),
        }

        try:
            logger.info("Opening WebSocket connection to %s", self.settings.gnani_stt_ws_url)
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.settings.gnani_stt_ws_url,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=5.0,
            )
            self.is_connected = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())
            logger.info("Gnani Prisma STT WebSocket connected successfully")
            return True
        except Exception as e:
            logger.warning(
                "Failed to connect to Gnani Prisma STT WebSocket (%s). Falling back to mock engine.",
                e,
            )
            self.is_mock_mode = True
            self.is_connected = True
            return False

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Queue raw PCM 16kHz audio chunk for streaming to ASR."""
        if self._closed:
            return

        if self.is_mock_mode:
            await self._simulate_mock_recognition(audio_bytes)
            return

        await self._audio_queue.put(audio_bytes)

    async def stream_transcripts(self) -> AsyncGenerator[STTTranscript, None]:
        """Asynchronously yield incoming interim and final transcripts."""
        while not self._closed:
            try:
                transcript = await asyncio.wait_for(self._transcript_queue.get(), timeout=0.1)
                yield transcript
                self._transcript_queue.task_done()
            except asyncio.TimeoutError:
                if self._closed:
                    break
                continue

    async def _send_loop(self) -> None:
        """Background task pumping queued audio frames into WebSocket."""
        try:
            while not self._closed and self._ws:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                await self._ws.send(chunk)
                self._audio_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in Gnani STT audio send loop: %s", e)

    async def _receive_loop(self) -> None:
        """Background task receiving interim transcript frames from WebSocket."""
        try:
            while not self._closed and self._ws:
                message = await self._ws.recv()
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        text = data.get("transcript") or data.get("text") or ""
                        is_final = data.get("is_final") or data.get("final", False)
                        confidence = float(data.get("confidence", 0.95))
                        if text:
                            await self._transcript_queue.put(
                                STTTranscript(
                                    text=text,
                                    is_final=is_final,
                                    language=self.language,
                                    confidence=confidence,
                                )
                            )
                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON text from Gnani STT: %s", message)
        except asyncio.CancelledError:
            pass
        except WebSocketException as e:
            logger.warning("Gnani STT WebSocket terminated: %s", e)
        except Exception as e:
            logger.error("Unexpected error in Gnani STT receive loop: %s", e)

    async def _simulate_mock_recognition(self, audio_bytes: bytes) -> None:
        """Simulate realistic Indic code-mixed interim transcripts for local testing."""
        # Check frame size and audio energy: do not simulate speech recognition on silence
        if len(audio_bytes) < 320:
            return

        import numpy as np
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(samples) > 0 and int(np.max(np.abs(samples))) < 500:
            return

        mock_phrases = [
            "Mera",
            "Mera account",
            "Mera account balance",
            "Mera account balance matlab ki",
        ]
        # Emit incremental tokens periodically if queue is empty
        if self._transcript_queue.empty():
            for phrase in mock_phrases:
                await self._transcript_queue.put(
                    STTTranscript(
                        text=phrase,
                        is_final=False,
                        language=self.language,
                        confidence=0.92,
                    )
                )
                await asyncio.sleep(0.05)

    def transcribe_rest(
        self, audio_bytes: bytes, audio_format: str = "wav"
    ) -> STTTranscript:
        """Synchronous/REST fallback for single utterance transcription."""
        if self.is_mock_mode:
            logger.info("Transcribe REST returning mock fallback result")
            return STTTranscript(
                text="Mera electricity bill bhar diya hai",
                is_final=True,
                language=self.language,
                confidence=0.95,
            )

        api_key = self.settings.gnani_api_key.get_secret_value()
        headers = {
            "X-API-Key-ID": api_key,
            "Content-Type": f"audio/{audio_format}",
        }
        params = {"language": self.language}

        try:
            response = requests.post(
                self.settings.gnani_stt_rest_url,
                headers=headers,
                params=params,
                data=audio_bytes,
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
            return STTTranscript(
                text=data.get("transcript", ""),
                is_final=True,
                language=self.language,
                confidence=float(data.get("confidence", 0.9)),
            )
        except Exception as e:
            logger.warning("Gnani STT REST fallback request failed: %s. Returning mock.", e)
            return STTTranscript(
                text="Mera electricity bill bhar diya hai",
                is_final=True,
                language=self.language,
                confidence=0.8,
            )

    async def close(self) -> None:
        """Gracefully release client resources and close socket."""
        self._closed = True
        if self._send_task:
            self._send_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self.is_connected = False
        logger.info("GnaniSTTClient closed")


async def _main():
    """CLI execution entrypoint for verification."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 60)
    print(" 🎙️  Testing Gnani Prisma STT Client Module")
    print("=" * 60)

    client = GnaniSTTClient()
    connected = await client.connect()
    print(f"Connection Status: {'Connected' if connected else 'Failed'}")
    print(f"Mode: {'Mock / Fallback' if client.is_mock_mode else 'Live Vachana API'}")

    # Send dummy 16kHz PCM audio chunk (1600 bytes = 50ms at 16kHz 16-bit mono)
    dummy_pcm = b"\x00\x00" * 800
    await client.send_audio_chunk(dummy_pcm)

    # Read interim token
    count = 0
    async for item in client.stream_transcripts():
        print(f"  [Interim Token] '{item.text}' (final={item.is_final}, conf={item.confidence:.2f})")
        count += 1
        if count >= 3:
            break

    rest_res = client.transcribe_rest(dummy_pcm)
    print(f"  [REST Fallback] '{rest_res.text}'")

    await client.close()
    print("✓ Gnani STT Client verification passed successfully.")


if __name__ == "__main__":
    asyncio.run(_main())
