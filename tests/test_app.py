"""Drives the real Streamlit app (Main.py) with a fake embedder and a fake model."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FakeEmbedder, FakeLLM
from smartrag import embeddings
from smartrag.llm import OfflineLLM
from smartrag.ui import state

MAIN = str(Path(__file__).resolve().parent.parent / "Main.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    st.cache_resource.clear()
    embeddings.set_embedder(FakeEmbedder())
    return AppTest.from_file(MAIN, default_timeout=60)


def button(at, label=None, key=None):
    return next(b for b in at.button if (key and b.key == key) or (label and b.label == label))


def texts(at):
    return " ".join(m.value for m in at.markdown)


def test_first_load_has_no_errors_and_shows_welcome(app):
    at = app.run()
    assert not at.exception
    assert "Bring your documents" in texts(at) and button(at, key="sample_btn")
    assert at.chat_input[0].placeholder.startswith("Ask a question")


def test_sample_document_then_cited_answer(app, monkeypatch):
    llm = FakeLLM("Chunk overlap keeps a sentence whole at a boundary [1].")
    monkeypatch.setattr(state, "llm", lambda: llm)
    at = app.run()
    button(at, key="sample_btn").click().run()
    assert not at.exception
    assert state.kb.__name__ == "kb" and "how-rag-works.md" in at.session_state["kb"].files
    assert at.session_state["kb"].mode == "hybrid"
    assert "1 document ready" in texts(at)

    at.chat_input[0].set_value("What is chunk overlap?").run()
    assert not at.exception
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    a = msgs[1]
    assert a["kind"] == "answer" and "[1]" in a["content"]
    assert a["citations"][0]["used"] and "overlap" in a["citations"][0]["text"].lower()
    assert any("Sources" in e.label for e in at.expander)


def test_unrelated_question_is_declined(app, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(state, "llm", lambda: llm)
    at = app.run()
    button(at, key="sample_btn").click().run()
    at.chat_input[0].set_value("What is the population of Mars in 2050?").run()
    assert at.session_state["messages"][-1]["kind"] == "refusal" and llm.calls == []


def test_suggestion_chip_asks_the_question(app, monkeypatch):
    monkeypatch.setattr(state, "llm", lambda: FakeLLM("Because it finds more [1]."))
    at = app.run()
    button(at, key="sample_btn").click().run()
    button(at, "Why use hybrid search?").click().run()
    assert not at.exception and at.session_state["messages"][0]["content"] == "Why use hybrid search?"


def test_offline_mode_still_answers_with_passages(app, monkeypatch):
    monkeypatch.setattr(state, "llm", lambda: OfflineLLM())
    at = app.run()
    assert "Offline mode" in texts(at)
    button(at, key="sample_btn").click().run()
    at.chat_input[0].set_value("What is chunk overlap?").run()
    assert not at.exception and at.session_state["messages"][-1]["kind"] == "extractive"


def test_clear_chat_remove_document_and_reset(app, monkeypatch):
    monkeypatch.setattr(state, "llm", lambda: FakeLLM("ok [1]"))
    at = app.run()
    button(at, key="sample_btn").click().run()
    at.chat_input[0].set_value("What is chunk overlap?").run()
    button(at, "Clear chat").click().run()
    assert at.session_state["messages"] == [] and at.session_state["kb"].has_documents()
    button(at, key="rm_how-rag-works.md").click().run()
    assert not at.exception and not at.session_state["kb"].has_documents()
    assert "Bring your documents" in texts(at)


def test_founder_question(app, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(state, "llm", lambda: llm)
    at = app.run()
    at.chat_input[0].set_value("who built you?").run()
    assert "Sarder Junaid Ahmed" in at.session_state["messages"][-1]["content"] and llm.calls == []


def test_model_failure_does_not_break_the_page(app, monkeypatch):
    from smartrag.llm import LLMError
    monkeypatch.setattr(state, "llm", lambda: FakeLLM("Start", fail=LLMError("connection", "Could not reach Groq.")))
    at = app.run()
    button(at, key="sample_btn").click().run()
    at.chat_input[0].set_value("What is chunk overlap?").run()
    assert not at.exception and "Could not reach Groq." in at.session_state["messages"][-1]["content"]


def test_refusal_offers_a_general_answer_button(app, monkeypatch):
    llm = FakeLLM("Mars has no permanent population.")
    monkeypatch.setattr(state, "llm", lambda: llm)
    at = app.run()
    button(at, key="sample_btn").click().run()
    at.chat_input[0].set_value("What is the population of Mars in 2050?").run()
    assert at.session_state["messages"][-1]["kind"] == "refusal" and llm.calls == []
    button(at, "Answer from general knowledge").click().run()
    assert not at.exception and at.session_state["messages"][-1]["kind"] == "general" and len(llm.calls) == 1


def test_greeting_in_documents_mode_is_friendly(app, monkeypatch):
    monkeypatch.setattr(state, "llm", lambda: FakeLLM("Hi there! Ask me about your documents."))
    at = app.run()
    button(at, key="sample_btn").click().run()
    at.chat_input[0].set_value("Hello bro? How are you?").run()
    assert at.session_state["messages"][-1]["kind"] == "chat" and "Hi there" in at.session_state["messages"][-1]["content"]
