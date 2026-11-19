"""Language-model clients behind one small interface: stream(messages, ...) yields text."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Iterator

from .settings import GROQ_FALLBACK_MODELS, GROQ_PREFERRED, GROQ_RATE_LIMIT_FALLBACK

log = logging.getLogger(__name__)
Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class LLMError(Exception):
    """A model call failed. `kind` is one of auth, rate_limit, connection, model, other."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


class OfflineLLM:
    """Used when no API key is configured. The app then shows the best matching passages instead."""
    available = False
    provider = "offline"
    last_model = "offline"

    def list_models(self) -> list[str]:
        return []

    def stream(self, messages, model, temperature=0.2, max_tokens=900) -> Iterator[str]:
        raise LLMError("auth", "No language model is connected.")

    def complete(self, messages, model, temperature=0.0, max_tokens=200) -> str:
        raise LLMError("auth", "No language model is connected.")


_MODEL_CACHE: dict[str, tuple[float, list[str]]] = {}
_NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "embed", "playai", "transcribe", "safeguard")


def order_models(ids: list[str]) -> list[str]:
    """Chat models only, preferred ones first."""
    chat = sorted({i for i in ids if not any(bad in i.lower() for bad in _NOT_CHAT)})
    first = [m for m in GROQ_PREFERRED if m in chat]
    return first + [m for m in chat if m not in first]


class GroqLLM:
    available = True
    provider = "groq"

    def __init__(self, api_key: str, client=None):
        self.api_key = api_key
        self._client = client
        self.last_model = ""

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=45.0, max_retries=1)
        return self._client

    def list_models(self) -> list[str]:
        key = hashlib.sha256(self.api_key.encode()).hexdigest()
        cached = _MODEL_CACHE.get(key)
        if cached and time.time() - cached[0] < 3600:
            return cached[1]
        try:
            ids = [m.id for m in self.client.models.list().data]
            models = order_models(ids) or list(GROQ_FALLBACK_MODELS)
        except Exception as exc:
            log.warning("Could not list Groq models: %s", type(exc).__name__)
            return list(GROQ_FALLBACK_MODELS)
        _MODEL_CACHE[key] = (time.time(), models)
        return models

    def stream(self, messages: list[Message], model: str, temperature: float = 0.2,
               max_tokens: int = 900) -> Iterator[str]:
        import groq

        candidates = [model] + ([GROQ_RATE_LIMIT_FALLBACK] if model != GROQ_RATE_LIMIT_FALLBACK else [])
        last_error: LLMError | None = None
        for name in candidates:
            started = False
            try:
                response = self.client.chat.completions.create(
                    model=name, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=True,
                )
                self.last_model = name
                for event in response:
                    delta = event.choices[0].delta.content if event.choices else None
                    if delta:
                        started = True
                        yield delta
                return
            except groq.AuthenticationError:
                raise LLMError("auth", "The Groq API key was rejected. Check GROQ_API_KEY.") from None
            except groq.RateLimitError:
                last_error = LLMError("rate_limit", "Groq's free tier is busy right now. Wait a moment and try again.")
                if started:
                    raise last_error from None
                continue                                 # try the lighter fallback model
            except groq.APIConnectionError:
                raise LLMError("connection", "Could not reach Groq. Check the internet connection.") from None
            except groq.NotFoundError:
                last_error = LLMError("model", f"The model {name} is not available on this key.")
                if started:
                    raise last_error from None
                continue
            except groq.BadRequestError as exc:
                if "decommission" in str(exc).lower() or "not found" in str(exc).lower():
                    last_error = LLMError("model", f"The model {name} is no longer offered by Groq.")
                    continue
                raise LLMError("other", "Groq rejected the request. Try a shorter question.") from None
            except groq.APIStatusError as exc:
                raise LLMError("other", f"Groq returned an error ({exc.status_code}). Try again shortly.") from None
        raise last_error or LLMError("other", "No model could answer.")

    def complete(self, messages, model, temperature=0.0, max_tokens=200) -> str:
        return "".join(self.stream(messages, model, temperature, max_tokens)).strip()


class OllamaLLM:
    available = True
    provider = "ollama"

    def __init__(self, host: str, session=None):
        self.host = host.rstrip("/")
        self._http = session
        self.last_model = ""

    @property
    def http(self):
        if self._http is None:
            import requests

            self._http = requests.Session()
        return self._http

    def list_models(self) -> list[str]:
        try:
            r = self.http.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            return sorted({m["name"] for m in r.json().get("models", [])})
        except Exception:
            return []

    def stream(self, messages, model, temperature=0.2, max_tokens=900) -> Iterator[str]:
        import requests

        self.last_model = model
        try:
            with self.http.post(
                f"{self.host}/api/chat", stream=True, timeout=120,
                json={"model": model, "messages": messages, "stream": True,
                      "options": {"temperature": temperature, "num_predict": max_tokens}},
            ) as r:
                if r.status_code == 404:
                    raise LLMError("model", f"Ollama does not have the model {model}. Run: ollama pull {model}")
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    piece = json.loads(line).get("message", {}).get("content")
                    if piece:
                        yield piece
        except LLMError:
            raise
        except requests.ConnectionError:
            raise LLMError("connection", "Could not reach Ollama. Is `ollama serve` running?") from None
        except Exception:
            raise LLMError("other", "Ollama returned an error.") from None

    def complete(self, messages, model, temperature=0.0, max_tokens=200) -> str:
        return "".join(self.stream(messages, model, temperature, max_tokens)).strip()


def make_llm(settings):
    if settings.provider == "groq" and settings.groq_api_key:
        return GroqLLM(settings.groq_api_key)
    if settings.provider == "ollama":
        return OllamaLLM(settings.ollama_host)
    return OfflineLLM()
