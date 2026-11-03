"""Centralized application configuration for Project Vartalaap.

Adheres to Pydantic Settings v2 with strict validation and zero secret leakage.
Reads environment variables from `.env` with fallback defaults matching
technical architecture specifications.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with sensitive credential masking."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Server Network Configuration
    host: str = Field(default="127.0.0.1", description="FastAPI bind host")
    port: int = Field(default=8000, description="FastAPI bind port")
    debug: bool = Field(default=True, description="Debug mode flag")

    # Gnani AI Credentials & Base URLs
    gnani_api_key: SecretStr = Field(
        default=SecretStr("your_gnani_api_key_here"),
        description="Gnani API token used in X-API-Key-ID header",
    )
    gnani_stt_ws_url: str = Field(
        default="wss://api.vachana.ai/stt/v3/stream",
        description="Gnani Prisma STT WebSocket endpoint",
    )
    gnani_stt_rest_url: str = Field(
        default="https://api.vachana.ai/stt/v3",
        description="Gnani Prisma STT REST fallback endpoint",
    )
    gnani_tts_ws_url: str = Field(
        default="wss://api.vachana.ai/api/v1/tts",
        description="Gnani Timbre TTS streaming WebSocket endpoint",
    )
    gnani_tts_rest_url: str = Field(
        default="https://api.vachana.ai/api/v1/tts/inference",
        description="Gnani Timbre TTS REST inference endpoint",
    )

    # Speech Synthesis Voice Config
    gnani_default_voice: str = Field(
        default="Shivani",
        description="Default Timbre TTS voice identity",
    )
    gnani_tts_model: str = Field(
        default="timbre-v3.0",
        description="Gnani Timbre model version",
    )
    gnani_default_lang: str = Field(
        default="hi-IN",
        description="Default BCP-47 language code for Indic processing",
    )

    # Conversational LLM Config
    llm_provider: str = Field(
        default="gnani",
        description="Dialogue orchestrator provider ('gnani' | 'openai' | 'mock')",
    )
    llm_model: str = Field(
        default="evon-v3.3",
        description="LLM model identifier",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr("your_llm_api_key_here"),
        description="LLM provider API key",
    )

    # Vartalaap Engine Parameters
    default_silence_threshold_ms: int = Field(
        default=600,
        description="Baseline silence timeout before turn transition",
    )
    min_silence_threshold_ms: int = Field(
        default=250,
        description="Minimum floor for dynamic silence timeout",
    )
    max_silence_threshold_ms: int = Field(
        default=2000,
        description="Maximum ceiling for dynamic silence timeout",
    )
    hesitation_extended_ms: int = Field(
        default=1800,
        description="Extended silence timeout when cognitive filler is detected",
    )
    conjunction_extended_ms: int = Field(
        default=1500,
        description="Extended silence timeout when hanging conjunction is detected",
    )

    # Device Acceleration
    torch_device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="PyTorch compute target: 'cuda' (RTX 5060) or 'cpu'",
    )

    # Smart-Turn v3.2 Model Settings
    model_dir: Path = Field(
        default=Path("models"),
        description="Local directory storing model weights",
    )
    smart_turn_repo: str = Field(
        default="pipecat-ai/smart-turn-v3",
        description="Hugging Face repository ID for Smart-Turn weights",
    )
    smart_turn_gpu_model: str = Field(
        default="smart-turn-v3.2-gpu.onnx",
        description="Smart-Turn GPU ONNX weight filename",
    )
    smart_turn_cpu_model: str = Field(
        default="smart-turn-v3.2-cpu.onnx",
        description="Smart-Turn CPU fallback ONNX weight filename",
    )

    @field_validator("min_silence_threshold_ms", "max_silence_threshold_ms")
    @classmethod
    def validate_silence_bounds(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Silence thresholds must be non-negative")
        return v

    @property
    def has_valid_gnani_credentials(self) -> bool:
        """Return True only if a real, non-placeholder Gnani API key is configured."""
        val = self.gnani_api_key.get_secret_value().strip()
        return bool(val) and val != "your_gnani_api_key_here"

    @property
    def has_valid_llm_credentials(self) -> bool:
        """Return True only if a real, non-placeholder LLM API key is configured."""
        val = self.llm_api_key.get_secret_value().strip()
        return bool(val) and val != "your_llm_api_key_here"

    def get_smart_turn_path(self, prefer_gpu: bool = True) -> Path:
        """Get absolute or relative path to appropriate Smart-Turn model weights."""
        target_file = self.smart_turn_gpu_model if prefer_gpu else self.smart_turn_cpu_model
        return self.model_dir / target_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
