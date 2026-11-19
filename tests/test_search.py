import pytest

from conftest import BrokenEmbedder, FakeEmbedder, SAMPLE_MD
from smartrag.bm25 import BM25, tokenize
from smartrag.chunking import chunk_documents
from smartrag.documents import IngestError, load_file
from smartrag.index import KnowledgeBase, is_overview_question
from langchain_core.documents import Document

BIG = 10 * 1024 * 1024


def kb_with(embedder=None, **kw):
    kb = KnowledgeBase(embedder_factory=(lambda: embedder) if embedder else None, **kw)
    kb.add(load_file("guide.md", SAMPLE_MD, BIG))
    return kb


def test_tokenize_handles_bengali_and_stopwords():
    assert "the" not in tokenize("What is the capital of France?")
    assert "বাংলাদেশের" in tokenize("বাংলাদেশের রাজধানী ঢাকা।")


def test_bm25_ranks_matching_text_first():
    docs = [tokenize(t) for t in ["cats sleep all day", "dogs bark loudly", "overlap keeps context"]]
    scores = BM25(docs).scores(tokenize("context overlap"))
    assert scores.argmax() == 2 and scores[0] == 0


def test_chunks_overlap_and_keep_metadata():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_documents([Document(page_content=text, metadata={"source": "a.pdf", "page": 4})], 300, 60)
    assert len(chunks) > 3 and all(c.page == 4 and c.source == "a.pdf" for c in chunks)
    assert chunks[0].label == "a.pdf, p. 4"
    assert set(chunks[0].text.split()[-4:]) & set(chunks[1].text.split()[:12])


def test_keyword_search_finds_the_right_section():
    kb = kb_with()
    top = kb.search("how does chunk overlap work", 2)[0]
    assert top.chunk.section == "Chunking" and top.coverage > 0.5 and kb.mode == "keyword"


def test_hybrid_mode_adds_semantic_scores():
    kb = kb_with(FakeEmbedder())
    hits = kb.search("reciprocal rank fusion", 3)
    assert kb.mode == "hybrid" and hits[0].chunk.section == "Retrieval" and hits[0].dense is not None


def test_unrelated_question_is_not_relevant():
    for kb in (kb_with(), kb_with(FakeEmbedder())):
        assert not kb.is_relevant(kb.search("who won the football world cup in 1998", 3))


def test_duplicate_and_replace_and_remove():
    kb = kb_with()
    n = len(kb)
    assert kb.add(load_file("guide.md", SAMPLE_MD, BIG)).name == "guide.md" and len(kb) == n      # same file: no-op
    with pytest.raises(IngestError, match="same content"):
        kb.add(load_file("copy.md", SAMPLE_MD, BIG))
    kb.add(load_file("guide.md", b"# New\nCompletely different content about volcanoes.", BIG))   # replaced
    assert [c.source for c in kb.chunks] == ["guide.md"] and "volcanoes" in kb.chunks[0].text
    kb.remove("guide.md")
    assert not kb.has_documents() and kb.search("volcanoes") == []


def test_remove_one_of_two_keeps_search_consistent():
    kb = kb_with(FakeEmbedder())
    kb.add(load_file("b.txt", b"Volcanoes erupt when magma rises through the crust.", BIG))
    kb.remove("guide.md")
    hit = kb.search("magma crust", 1)[0]
    assert hit.chunk.source == "b.txt" and kb.mode == "hybrid"


def test_passage_cap():
    kb = KnowledgeBase(max_chunks=2)
    with pytest.raises(IngestError, match="limit"):
        kb.add(load_file("big.txt", ("word " * 2000).encode(), BIG))


def test_failing_embedder_degrades_to_keyword_search():
    kb = kb_with(BrokenEmbedder())
    assert kb.mode == "keyword" and kb.search("optical character recognition", 1)[0].chunk.section == "Limits"


def test_overview_spreads_over_documents():
    kb = kb_with()
    kb.add(load_file("b.txt", b"Volcanoes erupt when magma rises. " * 80, BIG))
    sources = {h.chunk.source for h in kb.overview(6)}
    assert sources == {"guide.md", "b.txt"}


def test_overview_detection():
    assert is_overview_question("Can you summarize this?") and is_overview_question("what is this document about")
    assert not is_overview_question("what is chunk overlap")


def test_bengali_documents_are_searchable_by_keywords():
    text = "বাংলাদেশের রাজধানী ঢাকা। ঢাকা নদীমাতৃক দেশের বৃহত্তম শহর।\n\nকক্সবাজার বিশ্বের দীর্ঘতম সমুদ্র সৈকত।".encode()
    kb = KnowledgeBase()
    kb.add(load_file("bn.txt", text, BIG))
    hits = kb.search("বাংলাদেশের রাজধানী কী?", 2)
    assert hits and "ঢাকা" in hits[0].chunk.text and kb.is_relevant(hits)
    assert not kb.is_relevant(kb.search("who won the football world cup", 2))
