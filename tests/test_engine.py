from conftest import FakeEmbedder, FakeLLM, SAMPLE_MD
from smartrag.documents import load_file
from smartrag.engine import Engine, REFUSAL
from smartrag.index import KnowledgeBase
from smartrag.llm import LLMError, OfflineLLM
from smartrag.about import is_founder_question

BIG = 10 * 1024 * 1024


def make(llm, settings, embedder=None, docs=True):
    kb = KnowledgeBase(embedder_factory=(lambda: embedder) if embedder else None)
    if docs:
        kb.add(load_file("guide.md", SAMPLE_MD, BIG))
    return Engine(kb, llm, settings)


def run(turn):
    text = "".join(turn.stream())
    return text, turn.result()


def test_grounded_answer_with_citations(settings):
    llm = FakeLLM("Overlap keeps context [1]. See also [9].")
    text, ans = run(make(llm, settings).ask("What is chunk overlap?", []))
    assert ans.kind == "answer" and ans.cited
    assert "[9]" not in ans.text and "[1]" in ans.text                 # invalid marker removed, valid kept
    assert [c.used for c in ans.citations][0] is True
    prompt = llm.calls[-1]
    assert "Context passages" in prompt[-1]["content"] and "guide.md, Chunking" in prompt[-1]["content"]
    assert "ONLY" in prompt[0]["content"]


def test_unrelated_question_is_refused_without_calling_the_model(settings):
    llm = FakeLLM()
    text, ans = run(make(llm, settings).ask("Who won the 1998 football world cup?", []))
    assert ans.kind == "refusal" and text == REFUSAL and llm.calls == []


def test_general_mode_answers_without_context(settings):
    llm = FakeLLM("Paris is the capital of France.")
    text, ans = run(make(llm, settings).ask("Capital of France?", [], documents_only=False))
    assert ans.kind == "general" and "Paris" in text and "Context passages" not in llm.calls[-1][-1]["content"]


def test_no_documents_uses_general_assistant(settings):
    llm = FakeLLM("Hello!")
    text, ans = run(make(llm, settings, docs=False).ask("hi", []))
    assert ans.kind == "general" and text == "Hello!"


def test_founder_question_never_reaches_the_model(settings):
    llm = FakeLLM()
    text, ans = run(make(llm, settings).ask("Who built you?", []))
    assert ans.kind == "founder" and "Sarder Junaid Ahmed" in text and llm.calls == []
    assert not is_founder_question("who wrote the methods section")


def test_offline_with_documents_shows_best_passages(settings):
    text, ans = run(make(OfflineLLM(), settings).ask("What is chunk overlap?", []))
    assert ans.kind == "extractive" and "overlap" in text.lower() and ans.citations[0].used


def test_offline_without_documents_explains_how_to_connect(settings):
    text, ans = run(make(OfflineLLM(), settings, docs=False).ask("hello", []))
    assert ans.kind == "offline" and "GROQ_API_KEY" in text


def test_model_failure_becomes_a_friendly_message(settings):
    llm = FakeLLM("Partial answer", fail=LLMError("rate_limit", "Groq is busy."))
    text, ans = run(make(llm, settings).ask("What is chunk overlap?", []))
    assert ans.kind == "error" and "Groq is busy." in ans.text and ans.text.startswith("Partial answer")


def test_unexpected_exception_does_not_crash(settings):
    llm = FakeLLM("x", fail=ValueError("weird"))
    _, ans = run(make(llm, settings).ask("What is chunk overlap?", []))
    assert ans.kind == "error" and "went wrong" in ans.text


def test_follow_up_is_rewritten_into_a_standalone_question(settings):
    llm = FakeLLM("Because of boundaries [1].", condense="what is chunk overlap")
    history = [{"role": "user", "content": "Tell me about chunking"}, {"role": "assistant", "content": "It splits text."}]
    _, ans = run(make(llm, settings).ask("why is it needed?", history))
    assert ans.standalone == "what is chunk overlap"
    assert len(llm.calls) == 2 and "standalone" in llm.calls[0][0]["content"].lower()


def test_no_rewrite_for_first_question_or_clear_long_question(settings):
    llm = FakeLLM()
    run(make(llm, settings).ask("What is chunk overlap?", []))
    assert len(llm.calls) == 1
    llm2 = FakeLLM()
    hist = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    run(make(llm2, settings).ask("Explain how reciprocal rank fusion merges the two ranked lists of passages", hist))
    assert len(llm2.calls) == 1


def test_memory_window_is_respected(settings):
    llm = FakeLLM()
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)]
    run(make(llm, settings).ask("What is chunk overlap in this guide for the documents please", hist))
    sent = llm.calls[-1]
    assert len(sent) <= 1 + settings.memory_turns * 2 + 1


def test_summary_questions_sample_the_whole_document(settings):
    llm = FakeLLM("A summary [1][2].")
    _, ans = run(make(llm, settings).ask("Summarize the document", []))
    assert ans.kind == "answer" and len(ans.citations) >= 2


def test_extractive_does_not_confuse_document_brackets_with_citations(settings):
    kb = KnowledgeBase()
    kb.add(load_file("p.txt", b"As shown in earlier work [7], overlap of chunks keeps context intact.", BIG))
    text, _ = run(Engine(kb, OfflineLLM(), settings).ask("chunk overlap context", []))
    assert "(7)" in text and "[7]" not in text
