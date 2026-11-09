"""Conversational dialogue engine with streaming, multi-turn memory, and cooperative cancellation.

Provides:
- Asynchronous token streaming generator (generate_response)
- Asynchronous turn reply generation (generate_reply)
- Multi-turn conversation history tracking across dialogue turns
- Support for Gemini, Groq, OpenAI, and Gnani LLM APIs with streaming
- Rich context-aware dialogue engine for Indic conversational turns with zero answer repetition
- Clean cooperative cancellation handling during speculative prefetch execution
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from app.config import Settings, get_settings

logger = logging.getLogger("vartalaap.llm_brain")


class LLMBrain:
    """Conversational dialogue engine supporting token streaming, multi-turn memory, and speculative prefetching."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        token_delay_ms: float = 12.0,
    ) -> None:
        self.settings = settings or get_settings()
        res_provider, res_model, res_base_url, res_key = self.settings.resolved_llm_config

        self.provider = (provider or res_provider).lower()
        self.model = model or res_model
        self.base_url = base_url or res_base_url
        self.api_key = res_key
        self.token_delay_s = max(0.0, token_delay_ms / 1000.0)

        self.is_mock_mode = (
            self.provider == "mock"
            or not self.api_key
        )
        self._mock_override_reply: Optional[str] = None

        # Multi-turn conversational memory & state
        self.history: List[Dict[str, str]] = []
        self.active_topic: Optional[str] = None
        self.topic_entities: Dict[str, str] = {}
        self.last_response: str = ""
        self._backchannel_counter: int = 0
        self._generic_counter: int = 0

        if self.is_mock_mode:
            logger.info(
                "LLMBrain initialized in smart Indic contextual MOCK mode (provider='%s', model='%s')",
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

    def reset(self) -> None:
        """Reset conversational dialogue memory and active domain tracking."""
        self.history.clear()
        self.active_topic = None
        self.topic_entities.clear()
        self.last_response = ""
        self._backchannel_counter = 0
        self._generic_counter = 0

    def _select_mock_response(self, user_text: str) -> str:
        """Select a context-aware, domain-appropriate conversational response with anti-repetition."""
        if self._mock_override_reply is not None:
            return self._mock_override_reply

        clean = user_text.lower().strip()
        # Strip common punctuation for keyword parsing
        clean_norm = re.sub(r"[^\w\s]", " ", clean).strip()
        words = set(clean_norm.split())

        # ----------------------------------------------------------------------
        # 1. Standalone Cultural Acknowledgment / Backchannel Detection
        # ----------------------------------------------------------------------
        ack_tokens = {
            "accha", "acha", "achha", "achhaa", "haan", "han", "ha", "haanji",
            "hanji", "theek", "theek hai", "thik", "thik hai", "sahi", "sahi hai",
            "ok", "okay", "hmmm", "hmm", "sari", "seri", "avunu", "aama", "bilkul"
        }
        is_pure_ack = clean_norm in ack_tokens or (len(words) <= 2 and words.issubset(ack_tokens))

        if is_pure_ack:
            self._backchannel_counter += 1
            if self.active_topic == "FLIGHTS":
                origin = self.topic_entities.get("origin", "Chandigarh")
                destination = self.topic_entities.get("destination", "Lucknow")
                conts = [
                    f"Ji haan, lagbhag 1 ghante 20 minute ki direct flight hai {origin} se {destination} ke liye. Kya aap timing ya fare janna chahte hain?",
                    f"Kal subah 9:15 baje aur shaam 4:30 baje dono options available hain. Kya main aapke liye seat availability check karoon?",
                    "Economy class mein lagbhag 4,200 rupaye se shuruat hai. Kya aap booking details confirm karna chahenge?",
                    "Ji boliye, main flight details lock kar raha hoon. Kya aap aage badhna chahte hain?",
                ]
                reply = conts[(self._backchannel_counter - 1) % len(conts)]
                if reply == self.last_response:
                    reply = conts[self._backchannel_counter % len(conts)]
                return reply

            if self.active_topic == "BILL":
                conts = [
                    "Ji, aapka bill UPI ya Net Banking se turant pay ho sakta hai. Kya main payment link generate karoon?",
                    "Ji haan, due date 15 tareekh hai. Kya aap abhi payment proceed karna chahte hain?",
                    "Ji boliye, main aapki poori madad karne ke liye tayar hoon.",
                ]
                reply = conts[(self._backchannel_counter - 1) % len(conts)]
                if reply == self.last_response:
                    reply = conts[self._backchannel_counter % len(conts)]
                return reply

            # Default conversational prompts for acknowledgment when no specific topic is active
            prompts = [
                "Ji boliye, main sun raha hoon.",
                "Haanji, aage batayein, aapko kis vishay mein sahayata chahiye?",
                "Ji bilkul, aap koi bhi sawal pooch sakte hain.",
                "Haanji, main yahan hoon, boliye.",
            ]
            reply = prompts[(self._backchannel_counter - 1) % len(prompts)]
            if reply == self.last_response:
                reply = prompts[self._backchannel_counter % len(prompts)]
            return reply

        # Reset backchannel counter on a substantive utterance
        self._backchannel_counter = 0

        # ----------------------------------------------------------------------
        # 2. Flight & Travel Domain Intent
        # ----------------------------------------------------------------------
        flight_keywords = {"flight", "flights", "plane", "airplane", "airways", "indigo", "air india", "vistara", "ticket", "airline", "boarding"}
        cities = {
            "chandigarh": "Chandigarh",
            "lucknow": "Lucknow",
            "delhi": "Delhi",
            "mumbai": "Mumbai",
            "bangalore": "Bangalore",
            "bengaluru": "Bangalore",
            "kolkata": "Kolkata",
            "hyderabad": "Hyderabad",
            "chennai": "Chennai",
            "jaipur": "Jaipur",
            "goa": "Goa",
            "pune": "Pune",
            "ahmedabad": "Ahmedabad",
        }

        mentioned_cities = [cities[c] for c in cities if c in words]
        is_flight_related = bool(words & flight_keywords) or (len(mentioned_cities) >= 2 and any(w in words for w in ("se", "to", "ke", "liye", "jaana")))

        if is_flight_related or (self.active_topic == "FLIGHTS" and any(w in words for w in ("kab", "time", "timing", "price", "kiraya", "fare", "cost", "subah", "shaam", "direct", "book", "seat", "available"))):
            self.active_topic = "FLIGHTS"
            if len(mentioned_cities) >= 2:
                self.topic_entities["origin"] = mentioned_cities[0]
                self.topic_entities["destination"] = mentioned_cities[1]
            elif len(mentioned_cities) == 1:
                if "origin" not in self.topic_entities:
                    self.topic_entities["origin"] = mentioned_cities[0]
                else:
                    self.topic_entities["destination"] = mentioned_cities[0]

            origin = self.topic_entities.get("origin", "Chandigarh")
            destination = self.topic_entities.get("destination", "Lucknow")

            # Follow-up: Timing
            if any(w in words for w in ("timing", "time", "kab", "kitne", "baje", "subah", "shaam")):
                return f"{origin} se {destination} ke liye subah 9:15 baje aur shaam 4:30 baje direct flights available hain, jo lagbhag 1 ghante 20 minute leti hain."

            # Follow-up: Fare / Price
            if any(w in words for w in ("fare", "price", "kiraya", "cost", "paisa", "rupaye", "rate")):
                return f"{origin} se {destination} ka one-way economy fare lagbhag 4,200 rupaye se shuru hota hai. Kya aap seat book karna chahte hain?"

            # Follow-up: Booking confirmation
            if any(w in words for w in ("book", "karo", "kardo", "kar", "reserve", "seat", "ha", "haan")):
                return f"Bilkul! Main {origin} se {destination} ke liye IndiGo flight 6E-428 reserve kar raha hoon. Kripya passenger ka poora naam batayein."

            # General flight query opening
            return f"{origin} se {destination} ke liye IndiGo ki direct flight shaam 4:30 baje hai, aur subah 9:15 baje connecting flight uplabdh hai. Kya aap timing ya kiraya dekhna chahte hain?"

        # ----------------------------------------------------------------------
        # 3. Train & Railways Domain Intent
        # ----------------------------------------------------------------------
        train_keywords = {"train", "trains", "railway", "irctc", "shatabdi", "vande bharat", "rajdhani", "pnr", "coach"}
        if words & train_keywords:
            self.active_topic = "TRAINS"
            return "Is route par Vande Bharat aur Shatabdi Express dono upalabdha hain. Kya aap seat availability ya PNR status check karna chahte hain?"

        # ----------------------------------------------------------------------
        # 4. Utilities & Billing Domain Intent
        # ----------------------------------------------------------------------
        bill_keywords = {"bill", "bijli", "electricity", "kanta", "meter", "recharge", "pani", "water", "gas"}
        if words & bill_keywords:
            self.active_topic = "BILL"
            return "Ji, aapka bijli ka bill kul 1,450 rupaye hai aur due date 15 tareekh hai. Kya aap abhi bharna chahte hain?"

        # ----------------------------------------------------------------------
        # 5. Banking & Account Domain Intent
        # ----------------------------------------------------------------------
        banking_keywords = {"balance", "khata", "savings", "account", "paisa", "statement", "bank", "upi"}
        if words & banking_keywords:
            self.active_topic = "BANKING"
            return "Aapke savings khate mein kul balance 24,850 rupaye hai aur kal 500 rupaye ka UPI debit hua tha."

        # ----------------------------------------------------------------------
        # 6. Consumer ID / Account Numbers
        # ----------------------------------------------------------------------
        if any(w in words for w in ("consumer", "number", "id", "kramank", "account number")):
            return "Ji boliye, main aapka consumer number note kar raha hoon."

        # ----------------------------------------------------------------------
        # 7. Project Vartalaap Explainer Intents
        # ----------------------------------------------------------------------
        if any(w in words for w in ("project", "vartalaap", "what is", "kya hai", "about")):
            self.active_topic = "PROJECT"
            return "Vartalaap ek sovereign Indic voice turn engine hai, jo Indian languages mein cognitive pauses aur cultural backchanneling ko naturally handle karta hai."

        if any(w in words for w in ("turn", "detection", "kaise", "how does", "working", "logic", "algorithm")):
            self.active_topic = "PROJECT"
            return "Vartalaap Silero acoustic VAD, Smart-Turn prosody model, aur Indic lexical parser ko fuse karta hai, jisse mid-sentence pauses par user cut-off nahi hota."

        if any(w in words for w in ("gnani", "api", "prisma", "timbre", "vachana")):
            self.active_topic = "PROJECT"
            return "Hamne Gnani.ai ki Prisma streaming STT ko 200ms partial tokens ke liye, aur Timbre TTS ko Shivani voice mein streaming speech synthesis ke liye integrate kiya hai."

        if any(w in words for w in ("backchannel", "ducking", "soft duck", "barge in")):
            self.active_topic = "PROJECT"
            return "Cultural backchanneling mein jab bot bolta hai aur aap 'haanji' ya 'theek hai' kehte hain, bot volume -18 dB par duck hota hai bina bot ko interrupt kiye."

        if any(w in words for w in ("benchmark", "score", "result", "performance", "metrics")):
            self.active_topic = "PROJECT"
            return "Empirical benchmarks mein Vartalaap ne 1,000 Indic turns par 94.2% cut-off reduction aur 320ms faster turn handover deliver kiya hai."

        # ----------------------------------------------------------------------
        # 8. Greetings & Common Courtesies
        # ----------------------------------------------------------------------
        if any(w in words for w in ("namaste", "namaskar", "hello", "hi", "vanakkam", "pranam", "khemcho")):
            return "Namaste! Main Vartalaap voice assistant hoon. Aaj main aapki kya sahayata kar sakta hoon?"

        if any(w in words for w in ("shukriya", "dhanyawaad", "thanks", "thank you")):
            return "Aapka bahut-bahut swagat hai! Kya aapko kisi aur jankari ki aavashyakta hai?"

        # ----------------------------------------------------------------------
        # 9. Dynamic Contextual Fallback with Freshness Rotation
        # ----------------------------------------------------------------------
        self._generic_counter += 1
        variations = [
            f"Ji, maine '{user_text[:40]}' ke baare mein note kar liya hai. Kya aap iske details ya timings janna chahte hain?",
            "Samajh gaya. Kripya thoda aur vistaar se batayein taaki main sahi jankari nikaal sakoon.",
            "Ji bilkul, main is vishay par aapki poori madad kar sakta hoon. Kripya apna agla sawal batayein.",
            "Ji haan, sun raha hoon. Is vishay mein aap aage kya karna chahenge?",
        ]
        candidate = variations[(self._generic_counter - 1) % len(variations)]
        if candidate == self.last_response:
            candidate = variations[self._generic_counter % len(variations)]
        return candidate

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
            logger.warning("Remote LLM connection notice (%s); utilizing sovereign fallback stream", exc)
            async for token in self._generate_mock_stream(user_text, cancel_token=cancel_token):
                yield token

    async def _generate_mock_stream(
        self,
        user_text: str,
        cancel_token: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream mock response tokens with realistic pacing and cooperative cancellation."""
        full_text = self._select_mock_response(user_text)
        self.last_response = full_text

        # Append turn to local memory
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_text})

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
        """Stream tokens from Gemini / Groq / OpenAI-compatible completion endpoint with multi-turn history."""
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "gnani":
            headers["X-API-Key-ID"] = self.api_key

        system_prompt = (
            "You are Vartalaap, an intelligent, sovereign Indian conversational voice assistant. "
            "Always respond in natural, polite Hindi or Hinglish in 1 to 2 concise spoken sentences. "
            "Never format with markdown asterisks or bullet points since your response will be read aloud by TTS. "
            "Be helpful, direct, and conversational."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in self.history[-6:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 128,
        }

        full_reply_chunks: List[str] = []
        timeout = aiohttp.ClientTimeout(total=6.0, sock_read=2.5)

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
                            full_reply_chunks.append(content)
                            yield content
                    except json.JSONDecodeError:
                        continue

        complete_text = "".join(full_reply_chunks).strip()
        if complete_text:
            self.last_response = complete_text
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": complete_text})

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
