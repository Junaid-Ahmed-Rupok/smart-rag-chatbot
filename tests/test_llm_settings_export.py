import json
from types import SimpleNamespace

import groq
import httpx
import pytest

from smartrag import llm as llm_mod
from smartrag.export import chat_to_markdown
from smartrag.llm import GroqLLM, LLMError, OllamaLLM, order_models
from smartrag.settings import load_settings


def event(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def groq_error(cls, status):
    req = httpx.Request("POST", "https://api.groq.com/x")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


class FakeClient:
    def __init__(self, script):
        self.script, self.used = script, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=i) for i in
            ["whisper-large-v3", "llama-3.1-8b-instant", "meta-llama/llama-guard-4", "llama-3.3-70b-versatile", "qwen3-32b"]]))

    def _create(self, model, **kw):
        self.used.append(model)
        action = self.script[model]
        if isinstance(action, Exception):
            raise action
        return iter([event(t) for t in action])


MSG = [{"role": "user", "content": "hi"}]


def test_groq_streams_text():
    c = FakeClient({"llama-3.3-70b-versatile": ["Hel", "lo", None, "!"]})
    assert "".join(GroqLLM("k", c).stream(MSG, "llama-3.3-70b-versatile")) == "Hello!"


def test_rate_limit_falls_back_to_lighter_model():
    c = FakeClient({"llama-3.3-70b-versatile": groq_error(groq.RateLimitError, 429),
                    "llama-3.1-8b-instant": ["ok"]})
    g = GroqLLM("k", c)
    assert "".join(g.stream(MSG, "llama-3.3-70b-versatile")) == "ok"
    assert g.last_model == "llama-3.1-8b-instant" and c.used == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_rate_limit_everywhere_is_a_friendly_error():
    c = FakeClient({m: groq_error(groq.RateLimitError, 429) for m in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")})
    with pytest.raises(LLMError) as e:
        list(GroqLLM("k", c).stream(MSG, "llama-3.3-70b-versatile"))
    assert e.value.kind == "rate_limit"


def test_bad_key_and_connection_errors():
    with pytest.raises(LLMError) as e:
        list(GroqLLM("k", FakeClient({"m": groq_error(groq.AuthenticationError, 401)})).stream(MSG, "m"))
    assert e.value.kind == "auth"
    req = httpx.Request("POST", "https://api.groq.com/x")
    with pytest.raises(LLMError) as e:
        list(GroqLLM("k", FakeClient({"m": groq.APIConnectionError(request=req)})).stream(MSG, "m"))
    assert e.value.kind == "connection"


def test_retired_model_moves_on_to_the_next():
    c = FakeClient({"old-model": groq_error(groq.NotFoundError, 404), "llama-3.1-8b-instant": ["fine"]})
    assert "".join(GroqLLM("k", c).stream(MSG, "old-model")) == "fine"


def test_model_list_is_filtered_ordered_and_cached(monkeypatch):
    llm_mod._MODEL_CACHE.clear()
    g = GroqLLM("key-1", FakeClient({}))
    assert g.list_models() == ["llama-3.3-70b-versatile", "qwen3-32b", "llama-3.1-8b-instant"]
    g.client.models = SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert g.list_models()[0] == "llama-3.3-70b-versatile"            # served from cache
    assert order_models(["whisper-x", "zzz", "llama-3.1-8b-instant"]) == ["llama-3.1-8b-instant", "zzz"]


def test_default_is_never_an_obscure_model():
    ids = ["allam-2-7b", "groq/compound", "openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b", "openai/gpt-oss-120b",
           "canopylabs/orpheus-tts"]
    ordered = order_models(ids)
    assert ordered[0] == "openai/gpt-oss-120b"
    assert set(ordered[-2:]) == {"allam-2-7b", "groq/compound"}
    assert "canopylabs/orpheus-tts" not in ordered


def test_light_fallback_is_chosen_from_what_exists():
    from smartrag.llm import light_fallback
    assert light_fallback(["openai/gpt-oss-120b", "openai/gpt-oss-20b"], "openai/gpt-oss-120b") == "openai/gpt-oss-20b"
    assert light_fallback(["x"], "llama-3.1-8b-instant") is None


def test_model_list_falls_back_when_provider_is_down():
    llm_mod._MODEL_CACHE.clear()
    c = FakeClient({})
    c.models = SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert GroqLLM("key-2", c).list_models() == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


class FakeResponse:
    status_code = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)


def test_ollama_stream_parsing():
    lines = [json.dumps({"message": {"content": "He"}}).encode(), b"", json.dumps({"message": {"content": "llo"}}).encode()]
    session = SimpleNamespace(post=lambda *a, **k: FakeResponse(lines))
    assert "".join(OllamaLLM("http://x", session).stream(MSG, "mistral")) == "Hello"


def test_settings_defaults_and_degradation(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert load_settings().provider == "offline"
    monkeypatch.setenv("LLM_PROVIDER", "groq")                          # asked for groq but no key: degrade, never crash
    assert load_settings().provider == "offline"
    monkeypatch.setenv("GROQ_API_KEY", "abc")
    assert load_settings().provider == "groq"


def test_settings_tolerate_bad_values(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    monkeypatch.setenv("CHUNK_OVERLAP", "99999")
    monkeypatch.setenv("RETRIEVAL_K", "500")
    s = load_settings()
    assert s.chunk_size == 900 and s.chunk_overlap <= s.chunk_size // 2 and s.top_k == 12


def test_export_markdown():
    md = chat_to_markdown(
        [{"role": "user", "content": "Q?"},
         {"role": "assistant", "content": "A [1]", "citations": [{"n": 1, "label": "a.pdf, p. 2", "text": "evidence text", "used": True},
                                                                  {"n": 2, "label": "b.pdf", "text": "skip", "used": False}]}],
        ["a.pdf"])
    assert "## You" in md and "[1] a.pdf, p. 2" in md and "b.pdf" not in md.split("**Sources**")[1]
