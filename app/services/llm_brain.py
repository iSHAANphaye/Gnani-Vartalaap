"""Conversational dialogue engine with streaming and cooperative task cancellation.

Provides:
- Asynchronous token streaming generator (generate_response)
- Asynchronous turn reply generation (generate_reply)
- Deterministic mock mode for local testing and benchmark evaluation
- Integration hook for Gnani Evon and OpenAI-compatible API providers
- Clean cooperative cancellation handling during speculative prefetch execution
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp

from app.config import Settings, get_settings

logger = logging.getLogger("vartalaap.llm_brain")


class LLMBrain:
    """Conversational dialogue engine supporting token streaming and speculative prefetching."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        token_delay_ms: float = 15.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = (provider or self.settings.llm_provider).lower()
        self.model = model or self.settings.llm_model
        self.base_url = base_url or getattr(self.settings, "llm_base_url", "https://api.vachana.ai/v1")
        self.token_delay_s = max(0.0, token_delay_ms / 1000.0)

        self.is_mock_mode = (
            self.provider == "mock"
            or not self.settings.has_valid_llm_credentials
        )
        self._mock_override_reply: Optional[str] = None

        if self.is_mock_mode:
            logger.info(
                "LLMBrain initialized in deterministic MOCK mode (provider='%s', model='%s')",
                self.provider,
                self.model,
            )
        else:
            logger.info(
                "LLMBrain initialized with remote provider='%s', model='%s', endpoint='%s'",
                self.provider,
                self.model,
                self.base_url,
            )

    def set_mock_reply(self, reply: Optional[str]) -> None:
        """Set or clear a custom mock reply string for deterministic testing."""
        self._mock_override_reply = reply

    def _select_mock_response(self, user_text: str) -> str:
        """Select a domain-appropriate conversational response for Indic dialogue."""
        if self._mock_override_reply is not None:
            return self._mock_override_reply

        clean = user_text.lower()

        if any(w in clean for w in ("bill", "bijli", "electricity", "kanta")):
            return "Ji, aapka bijli ka bill kul 1,450 rupaye hai. Kya aap abhi bharna chahte hain?"
        if any(w in clean for w in ("balance", "khata", "savings", "account", "paisa")):
            return "Aapke savings khate ka kul balance 24,850 rupaye hai."
        if any(w in clean for w in ("consumer", "number", "id", "kramank")):
            return "Ji boliye, main aapka consumer number note kar raha hoon."
        if any(w in clean for w in ("namaste", "namaskar", "hello", "hi", "vanakkam")):
            return "Namaste! Main Vartalaap voice assistant hoon. Aaj main aapki kya sahayata kar sakta hoon?"
        if any(w in clean for w in ("theek", "shukriya", "dhanyawaad", "thanks", "ok")):
            return "Aapka bahut-bahut swagat hai! Kya koi aur jankari chahiye?"

        return "Ji bilkul, main samajh gaya. Iske baare mein aap aur kya janna chahte hain?"

    async def generate_response(
        self,
        user_text: str,
        cancel_token: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens asynchronously while respecting cooperative task cancellation."""
        if not user_text or not user_text.strip():
            return

        if cancel_token is not None and cancel_token.is_set():
            return

        if self.is_mock_mode:
            async for token in self._generate_mock_stream(user_text, cancel_token=cancel_token):
                yield token
            return

        try:
            async for token in self._generate_remote_stream(user_text, cancel_token=cancel_token):
                yield token
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Remote LLM connection failed (%s); falling back to mock stream", exc)
            async for token in self._generate_mock_stream(user_text, cancel_token=cancel_token):
                yield token

    async def _generate_mock_stream(
        self,
        user_text: str,
        cancel_token: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream mock response tokens with realistic pacing and cooperative cancellation."""
        full_text = self._select_mock_response(user_text)
        # Split into word tokens preserving trailing spaces
        tokens = re.findall(r"\S+\s*", full_text)
        if not tokens:
            tokens = [full_text]

        for token in tokens:
            if cancel_token is not None and cancel_token.is_set():
                logger.debug("Mock stream aborted by cancel_token")
                return

            if self.token_delay_s > 0:
                try:
                    await asyncio.sleep(self.token_delay_s)
                except asyncio.CancelledError:
                    logger.debug("Mock stream aborted by task cancellation")
                    raise

            if cancel_token is not None and cancel_token.is_set():
                logger.debug("Mock stream aborted by cancel_token post-sleep")
                return

            yield token

    async def _generate_remote_stream(
        self,
        user_text: str,
        cancel_token: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Gnani Evon / OpenAI-compatible completion endpoint."""
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        api_key = self.settings.llm_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-API-Key-ID": api_key,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Vartalaap, an ultra-fast, helpful Indic voice assistant.",
                },
                {"role": "user", "content": user_text},
            ],
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 128,
        }

        timeout = aiohttp.ClientTimeout(total=5.0, sock_read=2.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise aiohttp.ClientError(
                        f"LLM API returned status {resp.status}: {error_body[:200]}"
                    )

                async for raw_line in resp.content:
                    if cancel_token is not None and cancel_token.is_set():
                        logger.debug("Remote stream aborted by cancel_token")
                        return

                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def generate_reply(
        self,
        user_text: str,
        cancel_token: Optional[asyncio.Event] = None,
    ) -> str:
        """Generate a complete text reply, supporting cooperative cancellation."""
        tokens: List[str] = []
        async for token in self.generate_response(user_text, cancel_token=cancel_token):
            tokens.append(token)
        return "".join(tokens).strip()
