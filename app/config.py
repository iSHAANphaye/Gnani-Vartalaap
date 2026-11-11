"""Centralized application configuration for Project Vartalaap.

Adheres to Pydantic Settings v2 with strict validation and zero secret leakage.
Reads environment variables from `.env` with fallback defaults matching
technical architecture specifications.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple
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
    host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    port: int = Field(default=7860, description="FastAPI bind port")
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
        description="Dialogue orchestrator provider ('gnani' | 'openai' | 'groq' | 'gemini' | 'mock')",
    )
    llm_model: str = Field(
        default="evon-v3.3",
        description="LLM model identifier",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr("your_llm_api_key_here"),
        description="LLM provider API key",
    )
    llm_base_url: str = Field(
        default="https://api.vachana.ai/v1",
        description="LLM provider base API URL",
    )

    # Multi-Provider LLM Keys
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API key",
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Groq API key",
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key",
    )

    # Concurrency & Admission Control
    max_concurrent_sessions: int = Field(
        default=45,
        description="Hard session admission control cap (protects system RAM)",
    )
    save_debug_wav: bool = Field(
        default=False,
        description="Persist debug audio files to disk (disabled in production)",
    )

    # Vartalaap Engine Parameters & Audit Thresholds
    default_silence_threshold_ms: int = Field(
        default=600,
        description="Baseline silence timeout before turn transition",
    )
    min_silence_threshold_ms: int = Field(
        default=850,
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
        default=1800,
        description="Extended silence timeout when hanging conjunction is detected",
    )

    # Speculative Prefetch & Two-Tier Barge-In Timing (v2.0 Audit SLA)
    speculative_silence_ms: float = Field(
        default=150.0,
        description="Elapsed silence ms before triggering speculative Smart-Turn EoT evaluation",
    )
    speculative_eot_threshold: float = Field(
        default=0.85,
        description="Minimum EoT probability (1.0 - P_incomplete) required to fire speculative LLM",
    )
    soft_duck_trigger_ms: float = Field(
        default=60.0,
        description="Acoustic voice onset duration to trigger client soft-ducking",
    )
    hard_flush_trigger_ms: float = Field(
        default=450.0,
        description="Sustained speech duration to upgrade to hard flush barge-in",
    )
    soft_duck_gain_db: float = Field(
        default=-18.0,
        description="Soft-ducking attenuation gain in dB",
    )
    smart_turn_warmup_interval_sec: float = Field(
        default=3.0,
        description="Interval in seconds for background zero-tensor CUDA persistence warmup",
    )
    smart_turn_max_batch_size: int = Field(
        default=8,
        description="Max micro-batch size for Smart-Turn inference",
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
        """Return True if any valid, non-placeholder LLM API key is configured."""
        if self.gemini_api_key.get_secret_value().strip():
            return True
        if self.groq_api_key.get_secret_value().strip():
            return True
        if self.openai_api_key.get_secret_value().strip():
            return True
        val = self.llm_api_key.get_secret_value().strip()
        return bool(val) and val != "your_llm_api_key_here"

    @property
    def resolved_llm_config(self) -> Tuple[str, str, str, str]:
        """Resolve active LLM provider, model, base_url, and api_key."""
        g_key = self.gemini_api_key.get_secret_value().strip()
        if g_key and g_key != "your_gemini_api_key_here":
            return "gemini", "gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai", g_key

        groq_key = self.groq_api_key.get_secret_value().strip()
        if groq_key and groq_key != "your_groq_api_key_here":
            return "groq", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", groq_key

        o_key = self.openai_api_key.get_secret_value().strip()
        if o_key and o_key != "your_openai_api_key_here":
            return "openai", "gpt-4o-mini", "https://api.openai.com/v1", o_key

        val = self.llm_api_key.get_secret_value().strip()
        if bool(val) and val != "your_llm_api_key_here":
            return self.llm_provider, self.llm_model, self.llm_base_url, val

        return "mock", "mock-indic-v1", "", ""

    def get_smart_turn_path(self, prefer_gpu: bool = True) -> Path:
        """Get absolute or relative path to appropriate Smart-Turn model weights."""
        target_file = self.smart_turn_gpu_model if prefer_gpu else self.smart_turn_cpu_model
        return self.model_dir / target_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
