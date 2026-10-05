from __future__ import annotations

import numpy as np

from subarag.data.prepare import prepare_corpus
from subarag.retrieval import (
    AdaptiveRetriever,
    Bm25SparseIndex,
    Document,
    ExactDenseIndex,
    FixedDenseRetriever,
    HnswDenseIndex,
    RetrievalHit,
    build_hnsw_artifact,
    extract_supporting_spans,
    load_language_artifact,
    load_hnsw_artifact,
    reciprocal_rank_fusion,
    retrieval_cache_key,
    route_lexical,
    ranking_metrics,
    select_adaptive_k,
    unicode_tokens,
)


def source_row(query_id: int = 1):
    return {
        "query_id": query_id,
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "query": "गोवा की राजधानी क्या है?",
        "Answer": "पणजी",
        "Eng_Query": "What is Goa's capital?",
        "Eng_Answer": "Panaji",
        "passages": {
            "English_passages": ["Panaji is the capital of Goa.", "Mumbai is in Maharashtra."],
            "Translated_passages": ["पणजी गोवा की राजधानी है।", "मुंबई महाराष्ट्र में है।"],
            "is_selected": [1, 0],
        },
    }


def docs():
    return [
        Document("goa", "en", "Panaji is the capital of Goa. It sits beside the Mandovi."),
        Document("mumbai", "en", "Mumbai is in Maharashtra."),
        Document("numbers", "en", "Flight AI-202 leaves at 12:30."),
    ]


def test_artifact_loader_never_indexes_evaluation_rows(tmp_path):
    manifest = prepare_corpus(
        tmp_path, languages=["en", "hi"], splits=["validation"],
        reader=lambda source, revision: [source_row()],
    )
    documents, queries = load_language_artifact(tmp_path / "manifest.json", "hi")
    assert len(documents) == 2
    assert len(queries) == 1
    assert queries[0].query not in {document.text for document in documents}
    assert queries[0].relevant_passage_ids <= {document.id for document in documents}
    assert manifest.counts.evaluations == 2


def test_dense_indexes_share_ranked_contract():
    documents = docs()
    embeddings = np.asarray([[1, 0], [0, 1], [0.4, 0.6]], dtype=np.float32)
    query = np.asarray([1, 0], dtype=np.float32)
    exact = ExactDenseIndex(documents, embeddings)
    hnsw = HnswDenseIndex(documents, embeddings, m=2, ef_construction=20, ef_search=20)
    assert [hit.document.id for hit in exact.search(query, 3)] == ["goa", "numbers", "mumbai"]
    assert [hit.document.id for hit in hnsw.search(query, 3)] == ["goa", "numbers", "mumbai"]
    assert exact.size_bytes > 0 and hnsw.size_bytes > 0


def test_unicode_tokenizer_and_bm25_preserve_devanagari_words():
    assert unicode_tokens("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]
    documents = [
        Document("goa", "hi", "पणजी गोवा की राजधानी है।"),
        Document("mumbai", "hi", "मुंबई महाराष्ट्र में है।"),
    ]
    results = Bm25SparseIndex(documents).search("गोवा राजधानी", 2)
    assert results[0].document.id == "goa"
    assert results[0].score > results[1].score


def test_adaptive_k_uses_meaningful_gap_and_falls_back_when_scores_are_flat():
    documents = docs()
    separated = [RetrievalHit(document, score, rank, "dense") for rank, (document, score) in enumerate(
        zip(documents, [0.90, 0.87, 0.40], strict=True), 1
    )]
    flat = [RetrievalHit(document, score, rank, "dense") for rank, (document, score) in enumerate(
        zip(documents, [0.90, 0.88, 0.86], strict=True), 1
    )]
    assert select_adaptive_k(separated, min_relative_gap=0.1).k == 2
    assert select_adaptive_k(flat, min_relative_gap=0.1).k == 3


def test_conditional_lexical_route_and_rrf():
    documents = docs()
    dense = [RetrievalHit(documents[0], 0.81, 1, "dense"), RetrievalHit(documents[2], 0.80, 2, "dense")]
    lexical = [RetrievalHit(documents[2], 4.0, 1, "lexical"), RetrievalHit(documents[0], 1.0, 2, "lexical")]
    assert route_lexical("When does AI-202 leave?", dense).run_lexical
    fused = reciprocal_rank_fusion(dense, lexical)
    assert {hit.document.id for hit in fused[:2]} == {"goa", "numbers"}


def test_extract_supporting_spans_reduces_context_and_keeps_source_hit():
    hit = RetrievalHit(docs()[0], 0.9, 1, "dense")
    spans = extract_supporting_spans("capital Goa", [hit], max_sentences_per_document=1)
    assert spans == [(hit, "Panaji is the capital of Goa.")]
    assert len(spans[0][1]) < len(hit.document.text)


def test_retrieval_cache_keys_include_versions():
    one = retrieval_cache_key(" Goa? ", "en", corpus_version="c1", retriever_version="r1")
    equivalent = retrieval_cache_key("goa?", "en", corpus_version="c1", retriever_version="r1")
    changed = retrieval_cache_key("goa?", "en", corpus_version="c2", retriever_version="r1")
    assert one == equivalent
    assert one != changed


def test_retrieval_metrics_use_relevance_labels_not_candidate_membership():
    from subarag.retrieval.models import EvaluationQuery

    documents = docs()
    query = EvaluationQuery(
        id="q1", language="en", query="capital", relevant_passage_ids=frozenset({"goa"}),
        candidate_passage_ids=("goa", "mumbai"),
    )
    ranking = {
        "q1": [
            RetrievalHit(documents[1], 0.9, 1, "dense"),
            RetrievalHit(documents[0], 0.8, 2, "dense"),
        ]
    }
    metrics = ranking_metrics([query], ranking, k=10)
    assert metrics["recall@10"] == 1
    assert metrics["mrr@10"] == 0.5


def test_adaptive_retriever_runs_lexical_conditionally_and_caches():
    documents = docs()

    class FakeEmbedder:
        name = "fixture"
        dimension = 2
        calls = 0

        def embed_queries(self, texts):
            self.calls += 1
            return np.asarray([[1, 0]], dtype=np.float32)

    embedder = FakeEmbedder()
    retriever = AdaptiveRetriever(
        embedder=embedder,
        dense_index=ExactDenseIndex(documents, np.asarray([[1, 0], [0, 1], [0.4, 0.6]])),
        lexical_index=Bm25SparseIndex(documents),
        corpus_version="fixture-corpus", retriever_version="fixture-retriever",
        candidate_k=3, max_k=3,
    )
    first = retriever.retrieve("When does AI-202 leave?", "en")
    second = retriever.retrieve("when does ai-202 leave?", "en")
    assert first.telemetry.lexical_triggered
    assert second.telemetry.cache_hit
    assert embedder.calls == 1


def test_fixed_dense_retriever_is_the_cacheable_production_policy():
    documents = docs()

    class FakeEmbedder:
        name = "fixture"
        dimension = 2
        calls = 0

        def embed_queries(self, texts):
            self.calls += 1
            return np.asarray([[1, 0]], dtype=np.float32)

    embedder = FakeEmbedder()
    retriever = FixedDenseRetriever(
        embedder=embedder,
        dense_index=ExactDenseIndex(
            documents,
            np.asarray([[1, 0], [0, 1], [0.4, 0.6]], dtype=np.float32),
        ),
        corpus_version="fixture-corpus",
        retriever_version="e5-small-hnsw-k8-v1",
        k=2,
    )

    first = retriever.retrieve("What is Goa's capital?", "en")
    second = retriever.retrieve(" what IS goa's capital? ", "en")
    assert [hit.document.id for hit in first.hits] == ["goa", "numbers"]
    assert first.telemetry.selection_policy == "fixed"
    assert first.telemetry.selected_k == 2
    assert first.telemetry.lexical_triggered is False
    assert second.telemetry.cache_hit is True
    assert embedder.calls == 1


def test_hnsw_artifact_round_trip_and_checksum_verification(tmp_path):
    documents = docs()
    embeddings = np.asarray([[1, 0], [0, 1], [0.4, 0.6]], dtype=np.float32)
    manifest = build_hnsw_artifact(
        tmp_path / "artifact",
        documents=documents,
        embeddings=embeddings,
        corpus_version="corpus-sha",
        model_name="fixture/e5",
        retriever_version="e5-hnsw-k8-v1",
        language="en",
    )

    loaded = load_hnsw_artifact(manifest, expected_model="fixture/e5")
    assert loaded.corpus_version == "corpus-sha"
    assert loaded.retriever_version == "e5-hnsw-k8-v1"
    assert [hit.document.id for hit in loaded.index.search([1, 0], 2)] == [
        "goa",
        "numbers",
    ]

    index_path = manifest.parent / "index.faiss"
    data = bytearray(index_path.read_bytes())
    data[-1] ^= 1
    index_path.write_bytes(data)
    with np.testing.assert_raises_regex(ValueError, "checksum failed"):
        load_hnsw_artifact(manifest)
