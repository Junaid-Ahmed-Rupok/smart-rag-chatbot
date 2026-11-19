import pytest

from conftest import FakeLLM, SAMPLE_MD
from smartrag.documents import load_file
from smartrag.engine import Engine
from smartrag.index import KnowledgeBase
from smartrag.llm import OfflineLLM
from smartrag.smalltalk import OFFLINE_REPLY, is_smalltalk

BIG = 10 * 1024 * 1024


@pytest.mark.parametrize("text", [
    "Hello bro? How are you?", "hi", "Hey!", "thanks", "Thank you so much", "good morning", "Assalamu alaikum",
    "who are you?", "what can you do?", "how does this work", "সালাম", "bye",
])
def test_greetings_are_small_talk(text):
    assert is_smalltalk(text)


@pytest.mark.parametrize("text", [
    "hello, what is the revenue growth?", "What is chunk overlap?", "How are the passages ranked?",
    "thanks, but what is his education?", "Summarize the document", "help me find the churn rate",
])
def test_real_questions_are_not_small_talk(text):
    assert not is_smalltalk(text)


def make(llm, settings):
    kb = KnowledgeBase()
    kb.add(load_file("guide.md", SAMPLE_MD, BIG))
    return Engine(kb, llm, settings)


def test_greeting_gets_a_chat_reply_even_with_documents_only(settings):
    llm = FakeLLM("Hi! I'm doing great. Ask me about guide.md.")
    turn = make(llm, settings).ask("Hello bro? How are you?", [], documents_only=True)
    text = "".join(turn.stream())
    ans = turn.result()
    assert ans.kind == "chat" and "doing great" in text and ans.citations == []
    assert "guide.md" in llm.calls[-1][0]["content"]            # the model knows which documents are loaded


def test_greeting_offline(settings):
    turn = make(OfflineLLM(), settings).ask("hi", [])
    assert "".join(turn.stream()) == OFFLINE_REPLY and turn.result().kind == "chat"


def test_refusal_names_the_documents(settings):
    turn = make(FakeLLM(), settings).ask("What is the population of Mars in 2050?", [])
    assert "guide.md" in "".join(turn.stream())
