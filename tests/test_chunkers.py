"""Tests for SentenceChunker — sentence-boundary-aware chunking.

Fixed char windows cut sentences mid-thought; SentenceChunker packs whole
sentences into chunks. These tests pin down boundary preservation, oversized
sentence handling, and sentence-level overlap.
"""
from kssrag.core.chunkers import SentenceChunker


def test_chunks_respect_sentence_boundaries():
    text = (
        "Kosisochukwu is a software engineer from Nigeria. "
        "He builds RAG systems and LLM tooling. "
        "He studies at FUTO in Owerri!\n"
        "Anime? One Piece, Naruto, and Attack on Titan are his favorites. "
        "He supports Real Madrid and Al Nassr."
    )
    chunker = SentenceChunker(chunk_size=90, overlap=0)
    chunks = chunker.chunk(text, {"source": "s"})
    assert len(chunks) >= 2
    for chunk in chunks:
        # Every chunk ends exactly at a sentence boundary.
        assert chunk["content"].endswith((".", "!", "?"))


def test_oversized_sentence_emitted_alone():
    text = "Short sentence here. " + ("Y" * 250) + ". Then a normal ending sentence."
    chunker = SentenceChunker(chunk_size=100, overlap=0)
    chunks = chunker.chunk(text, {})
    big = [c for c in chunks if len(c["content"]) >= 250]
    assert len(big) == 1
    assert big[0]["content"].startswith("Y" * 250)


def test_overlap_carries_trailing_sentences():
    # s1(24) s2(40) s3(23) s4(27); chunk_size=70 packs (s1,s2) then (s2,s3) ...
    s1 = "A short first sentence."
    s2 = "A slightly longer second sentence here."
    s3 = "Third sentence content."
    s4 = "Fourth and final sentence."
    text = f"{s1} {s2} {s3} {s4}"
    chunker = SentenceChunker(chunk_size=70, overlap=30)
    chunks = chunker.chunk(text, {})
    assert chunks[0]["content"].startswith(s1)
    # Overlap: the second chunk re-opens with the previous chunk's tail.
    assert chunks[1]["content"].startswith(s2)
    assert len(chunks) >= 3
    assert chunks[2]["content"].startswith(s3)


def test_short_text_single_chunk():
    chunks = SentenceChunker(chunk_size=500, overlap=50).chunk(
        "Just one sentence here.", {"source": "t"}
    )
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_id"] == 0


def test_chunk_ids_are_sequential():
    text = "First sentence. " * 40  # plenty of text
    chunks = SentenceChunker(chunk_size=100, overlap=10).chunk(text, {})
    assert [c["metadata"]["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_empty_text_returns_empty_list():
    assert SentenceChunker().chunk("   \n\n ", {}) == []


def test_sentence_splitter_handles_paragraph_breaks():
    chunker = SentenceChunker()
    sentences = chunker._split_sentences(
        "First paragraph sentence.\n\nSecond paragraph sentence. Third here!"
    )
    assert len(sentences) == 3


def test_kssrag_uses_sentence_chunker_when_configured(tmp_path):
    from kssrag import KSSRAG
    from kssrag.config import Config, VectorStoreType, ChunkerType

    src = tmp_path / "bio.txt"
    src.write_text(
        "First sentence about building RAG systems. " * 8
        + "Final sentence that closes the document here.",
        encoding="utf-8",
    )
    cfg = Config(
        CHUNKER_TYPE=ChunkerType.SENTENCE,
        VECTOR_STORE_TYPE=VectorStoreType.BM25,
        ENABLE_CACHE=False,
        CHUNK_SIZE=100,
        CHUNK_OVERLAP=0,
        OPENROUTER_API_KEY="",
    )
    rag = KSSRAG(config=cfg)
    rag.load_document(str(src))
    # Sentence-aware chunks never cut mid-sentence.
    assert len(rag.documents) >= 2
    assert all(c["content"].endswith(".") for c in rag.documents)
