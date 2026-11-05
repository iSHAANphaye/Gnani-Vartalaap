"""Voice Activity Detection and Rolling Audio Ring Buffer.

High-throughput, low-latency audio stream buffer and CPU-accelerated Silero VAD
inference for full-duplex conversational turn management.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import silero_vad
import torch

logger = logging.getLogger("vartalaap.vad")


@dataclass(frozen=True)
class VADResult:
    """Voice activity detection classification result for an audio frame."""

    is_speech: bool
    confidence: float
    silence_duration_ms: float
    speech_duration_ms: float


class AudioRingBuffer:
    """Thread-safe circular buffer for rolling 16kHz mono 16-bit PCM audio.

    Maintains up to `capacity_sec` seconds of audio (default 8.0s / 128,000 samples /
    256,000 bytes) with zero reallocations during active streaming.
    """

    def __init__(self, capacity_sec: float = 8.0, sample_rate: int = 16000) -> None:
        if capacity_sec <= 0:
            raise ValueError(f"capacity_sec must be positive, got {capacity_sec}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        self._sample_rate = sample_rate
        self._capacity_sec = float(capacity_sec)
        self._capacity_samples = int(self._capacity_sec * self._sample_rate)
        self._capacity_bytes = self._capacity_samples * 2  # 16-bit PCM = 2 bytes/sample

        self._buffer = np.zeros(self._capacity_samples, dtype=np.int16)
        self._write_pos: int = 0
        self._size: int = 0
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        """Audio sampling rate in Hertz."""
        return self._sample_rate

    @property
    def capacity_samples(self) -> int:
        """Maximum sample capacity."""
        return self._capacity_samples

    @property
    def capacity_bytes(self) -> int:
        """Maximum byte capacity."""
        return self._capacity_bytes

    @property
    def capacity_sec(self) -> float:
        """Maximum buffer duration in seconds."""
        return self._capacity_sec

    @property
    def size(self) -> int:
        """Current number of audio samples held in buffer."""
        with self._lock:
            return self._size

    @property
    def duration_sec(self) -> float:
        """Current audio duration held in buffer (seconds)."""
        with self._lock:
            return self._size / self._sample_rate

    @property
    def is_full(self) -> bool:
        """Whether buffer has reached maximum capacity."""
        with self._lock:
            return self._size >= self._capacity_samples

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def append(self, chunk: Union[bytes, bytearray, np.ndarray]) -> int:
        """Append incoming audio samples into circular ring buffer.

        Accepts raw 16-bit PCM bytes/bytearray or a 1D NumPy array (int16 or float32).
        Returns number of samples written.
        """
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            if len(chunk) == 0:
                return 0
            if len(chunk) % 2 != 0:
                raise ValueError(
                    f"Audio byte chunk length must be a multiple of 2, got {len(chunk)} bytes"
                )
            samples = np.frombuffer(chunk, dtype=np.int16)
        elif isinstance(chunk, np.ndarray):
            if chunk.size == 0:
                return 0
            if chunk.ndim > 1:
                chunk = chunk.squeeze()
            if chunk.dtype == np.int16:
                samples = chunk
            elif np.issubdtype(chunk.dtype, np.floating):
                clamped = np.clip(chunk, -1.0, 1.0)
                samples = (clamped * 32767.0).astype(np.int16)
            else:
                samples = chunk.astype(np.int16)
        else:
            raise TypeError(
                f"Unsupported chunk type: {type(chunk)}. Expected bytes, bytearray, or np.ndarray."
            )

        n = len(samples)
        if n == 0:
            return 0

        with self._lock:
            if n >= self._capacity_samples:
                # Chunk meets or exceeds total capacity: keep only most recent samples
                self._buffer[:] = samples[-self._capacity_samples :]
                self._write_pos = 0
                self._size = self._capacity_samples
                return n

            space = self._capacity_samples - self._write_pos
            if n <= space:
                self._buffer[self._write_pos : self._write_pos + n] = samples
                self._write_pos = (self._write_pos + n) % self._capacity_samples
            else:
                self._buffer[self._write_pos :] = samples[:space]
                remainder = n - space
                self._buffer[:remainder] = samples[space:]
                self._write_pos = remainder

            self._size = min(self._capacity_samples, self._size + n)
            return n

    def extend(self, chunks: Iterable[Union[bytes, bytearray, np.ndarray]]) -> int:
        """Append multiple audio chunks sequentially."""
        total = 0
        for chunk in chunks:
            total += self.append(chunk)
        return total

    def get_raw_bytes(self) -> bytes:
        """Return chronological 16-bit mono PCM bytes currently held in buffer."""
        with self._lock:
            if self._size == 0:
                return b""
            if self._size < self._capacity_samples:
                return self._buffer[: self._size].tobytes()

            # Wrapped buffer: oldest samples start at write_pos
            unrolled = np.concatenate(
                (self._buffer[self._write_pos :], self._buffer[: self._write_pos])
            )
            return unrolled.tobytes()

    def get_recent_audio(self, duration_sec: float = 8.0) -> np.ndarray:
        """Retrieve most recent audio as normalized float32 waveform [-1.0, 1.0].

        If buffer contains fewer samples than requested duration, result is left-padded
        with zeros to match the exact requested length. This format directly supplies
        the input tensor required by the Smart-Turn acoustic classification model.
        """
        if duration_sec <= 0:
            raise ValueError(f"duration_sec must be positive, got {duration_sec}")

        req_samples = int(duration_sec * self._sample_rate)

        with self._lock:
            if self._size == 0:
                return np.zeros(req_samples, dtype=np.float32)

            if self._size < self._capacity_samples:
                avail = self._buffer[: self._size]
            else:
                avail = np.concatenate(
                    (self._buffer[self._write_pos :], self._buffer[: self._write_pos])
                )

            if self._size < req_samples:
                pad_len = req_samples - self._size
                padded = np.pad(avail, (pad_len, 0), mode="constant", constant_values=0)
                return padded.astype(np.float32) / 32768.0
            else:
                recent = avail[-req_samples:]
                return recent.astype(np.float32) / 32768.0

    def clear(self) -> None:
        """Reset circular buffer to empty state."""
        with self._lock:
            self._write_pos = 0
            self._size = 0
            self._buffer.fill(0)

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()


class SileroVAD:
    """CPU-bound Silero Voice Activity Detector.

    Evaluates 512-sample (32ms at 16kHz) 16-bit PCM audio frames in <2.0ms on CPU,
    preserving GPU resources for downstream transformer classification models.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in range [0.0, 1.0], got {threshold}")
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Silero VAD supports sample rates [8000, 16000], got {sample_rate}")

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._chunk_samples = 512 if sample_rate == 16000 else 256
        self._chunk_bytes = self._chunk_samples * 2
        self._chunk_duration_ms = (self._chunk_samples / self._sample_rate) * 1000.0

        # Load Silero VAD ONNX model on CPU
        logger.info("Loading Silero VAD ONNX engine on CPU (sample_rate=%d)...", sample_rate)
        self._model = silero_vad.load_silero_vad(onnx=True)

        self._silence_duration_ms: float = 0.0
        self._speech_duration_ms: float = 0.0
        self._lock = threading.Lock()

        # Warm up ONNX session
        self._warmup()

    def _warmup(self, iterations: int = 5) -> None:
        """Execute warmup frames to prime ONNX Runtime caches and CPU execution threads."""
        dummy = torch.zeros(self._chunk_samples, dtype=torch.float32)
        for _ in range(iterations):
            _ = self._model(dummy, self._sample_rate)
        self._model.reset_states()

    @property
    def threshold(self) -> float:
        """Speech probability decision threshold."""
        return self._threshold

    @property
    def sample_rate(self) -> int:
        """Audio sampling rate in Hertz."""
        return self._sample_rate

    @property
    def silence_duration_ms(self) -> float:
        """Elapsed silence duration in milliseconds since last active speech."""
        with self._lock:
            return self._silence_duration_ms

    @property
    def speech_duration_ms(self) -> float:
        """Elapsed active speech duration in milliseconds since last silence."""
        with self._lock:
            return self._speech_duration_ms

    def process_chunk(self, chunk_bytes: bytes) -> VADResult:
        """Process a single 16-bit PCM mono audio chunk and update speech/silence tracking.

        Args:
            chunk_bytes: 16-bit PCM bytes (512 samples / 1024 bytes at 16kHz).

        Returns:
            VADResult with speech flag, raw confidence, and accumulated durations.
        """
        if len(chunk_bytes) != self._chunk_bytes:
            raise ValueError(
                f"Expected chunk size of {self._chunk_bytes} bytes ({self._chunk_samples} samples), "
                f"got {len(chunk_bytes)} bytes"
            )

        # Convert 16-bit PCM buffer to normalized float32 tensor
        audio_np = (
            np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        audio_tensor = torch.from_numpy(audio_np)

        with self._lock:
            raw_out = self._model(audio_tensor, self._sample_rate)
            confidence = float(raw_out.item())
            is_speech = confidence >= self._threshold

            if is_speech:
                self._speech_duration_ms += self._chunk_duration_ms
                self._silence_duration_ms = 0.0
            else:
                self._silence_duration_ms += self._chunk_duration_ms
                self._speech_duration_ms = 0.0

            return VADResult(
                is_speech=is_speech,
                confidence=confidence,
                silence_duration_ms=self._silence_duration_ms,
                speech_duration_ms=self._speech_duration_ms,
            )

    def reset(self) -> None:
        """Reset internal recurrent states and tracking timers."""
        with self._lock:
            self._model.reset_states()
            self._silence_duration_ms = 0.0
            self._speech_duration_ms = 0.0


class AcousticDetector:
    """Integrated acoustic detector combining circular ring buffer and Silero VAD.

    Serves as the primary acoustic ingestion coordinator for full-duplex conversational sessions.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ring_buffer_duration_sec: float = 8.0,
        sample_rate: int = 16000,
    ) -> None:
        self.ring_buffer = AudioRingBuffer(
            capacity_sec=ring_buffer_duration_sec,
            sample_rate=sample_rate,
        )
        self.vad = SileroVAD(
            threshold=threshold,
            sample_rate=sample_rate,
        )

    @property
    def silence_duration_ms(self) -> float:
        """Elapsed silence duration in milliseconds."""
        return self.vad.silence_duration_ms

    @property
    def speech_duration_ms(self) -> float:
        """Elapsed speech duration in milliseconds."""
        return self.vad.speech_duration_ms

    def process_chunk(self, chunk_bytes: bytes) -> VADResult:
        """Ingest chunk into rolling ring buffer and evaluate acoustic speech activity."""
        self.ring_buffer.append(chunk_bytes)
        return self.vad.process_chunk(chunk_bytes)

    def get_recent_audio(self, duration_sec: float = 8.0) -> np.ndarray:
        """Retrieve recent normalized audio waveform for acoustic turn evaluation."""
        return self.ring_buffer.get_recent_audio(duration_sec=duration_sec)

    def get_raw_bytes(self) -> bytes:
        """Retrieve raw chronological audio bytes currently held in buffer."""
        return self.ring_buffer.get_raw_bytes()

    def reset(self) -> None:
        """Reset both buffer storage and VAD inference states."""
        self.ring_buffer.reset()
        self.vad.reset()
