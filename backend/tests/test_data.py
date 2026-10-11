from __future__ import annotations

import json
from pathlib import Path

import pytest

from subarag.data import load_manifest
from subarag.data.manifest import verify_manifest
from subarag.data.prepare import passage_id, prepare_corpus
from subarag.data.source import DEFAULT_REVISION, source_files


def row(language: str, query_id: int = 1, **overrides):
    value = {
        "query_id": query_id,
        "source_lang": "eng_Latn",
        "target_lang": {"hi": "hin_Deva", "mr": "mar_Deva"}[language],
        "query": f"{language} question that must never become a document?",
        "Answer": f"{language} answer that must never become a document",
        "Eng_Query": "An English question that must never become a document?",
        "Eng_Answer": "An English answer that must never become a document",
        "passages": {
            "English_passages": ["The first passage.", "An unselected passage."],
            "Translated_passages": [f"{language} first passage.", f"{language} unselected passage."],
            "is_selected": [1, 0],
        },
    }
    value.update(overrides)
    return value


def records(output: Path, manifest, kind: str):
    return [json.loads(line) for shard in manifest.shards if shard.kind == kind
            for line in (output / shard.path).read_text(encoding="utf-8").splitlines()]


def test_source_files_use_actual_parquet_names_and_validation_first():
    files = source_files(["en", "hi", "mr"], ["train", "validation"])
    assert [item.path for item in files] == ["validation/hinval.parquet", "validation/marval.parquet",
                                           "train/hintrain.parquet", "train/martrain.parquet"]
    assert [item.path for item in source_files(["en"], ["train"])] == ["train/hintrain.parquet"]


def test_all_passages_are_docs_and_queries_answers_relevance_are_isolated(tmp_path):
    manifest = prepare_corpus(tmp_path, reader=lambda source, revision: [row(source.language)])
    documents = records(tmp_path, manifest, "documents")
    evaluations = records(tmp_path, manifest, "evaluation")
    assert manifest.source.revision == DEFAULT_REVISION
    assert manifest.document_counts_by_language == {"en": 2, "hi": 2, "mr": 2}
    assert len(evaluations) == 3  # English copied across translations, train/validation overlap.
    assert manifest.counts.held_out_train_evaluations == 4
    assert {item["split"] for item in evaluations} == {"validation"}
    assert all(set(doc) == {"id", "language", "text", "source"} for doc in documents)
    assert all("question" not in doc["text"] and "answer" not in doc["text"] for doc in documents)
    assert any("unselected" in doc["text"] for doc in documents)
    assert all(len(item["candidate_passage_ids"]) == 2 and len(item["relevant_passage_ids"]) == 1 for item in evaluations)
    assert verify_manifest(tmp_path / "manifest.json") == manifest


def test_passage_ids_are_stable_normalized_and_language_specific():
    assert passage_id("en", " Café\n text ") == passage_id("en", "Cafe\u0301 text")
    assert passage_id("hi", "same") != passage_id("en", "same")
    assert passage_id("en", "Apple") != passage_id("en", "apple")


def test_sample_stops_without_consuming_an_extra_row_and_shards_roll_over(tmp_path):
    consumed = []

    def reader(source, revision):
        for query_id in range(20):
            consumed.append(query_id)
            yield row(source.language, query_id)

    manifest = prepare_corpus(tmp_path, languages=["hi"], splits=["train"], rows_per_source=3,
                              shard_rows=1, reader=reader)
    assert consumed == [0, 1, 2]
    assert manifest.counts.source_rows == 3
    assert manifest.counts.documents == 2
    assert manifest.counts.evaluations == 3
    assert all(shard.rows == 1 for shard in manifest.shards)


def test_bad_rows_are_quarantined_with_locations_without_source_content(tmp_path):
    bad = row("hi", 2, passages={"English_passages": ["SECRET"], "Translated_passages": [], "is_selected": [1]})
    manifest = prepare_corpus(tmp_path, languages=["hi"], splits=["train"],
                              reader=lambda source, revision: [bad, row("hi", 3)])
    assert manifest.counts.rejected_rows == 1
    rejection = records(tmp_path, manifest, "rejections")[0]
    assert rejection["source"]["row"] == 0
    assert "SECRET" not in json.dumps(rejection)


def test_json_encoded_passage_object_is_supported(tmp_path):
    example = row("hi")
    example["passages"] = json.dumps(example["passages"])
    manifest = prepare_corpus(tmp_path, languages=["hi"], splits=["train"],
                              reader=lambda source, revision: [example])
    assert manifest.counts.documents == 2
    assert manifest.counts.rejected_rows == 0


def test_invalid_validation_row_still_reserves_holdout_query_id(tmp_path):
    def reader(source, revision):
        if source.split == "validation":
            yield row("hi", query="")
            yield row("hi", 2)
        else:
            yield row("hi")

    manifest = prepare_corpus(tmp_path, languages=["hi"], reader=reader)
    assert manifest.counts.held_out_train_evaluations == 1
    assert [item["query_id"] for item in records(tmp_path, manifest, "evaluation")] == ["2"]


def test_failed_run_has_no_manifest_and_existing_output_is_never_overwritten(tmp_path):
    def failing_reader(source, revision):
        yield row(source.language)
        raise RuntimeError("source interrupted")

    with pytest.raises(RuntimeError, match="source interrupted"):
        prepare_corpus(tmp_path, reader=failing_reader)
    assert not (tmp_path / "manifest.json").exists()
    with pytest.raises(ValueError, match="must be empty"):
        prepare_corpus(tmp_path, reader=lambda source, revision: [])


def test_all_bad_rows_do_not_produce_a_completed_artifact(tmp_path):
    with pytest.raises(ValueError):
        prepare_corpus(tmp_path, reader=lambda source, revision: [{}])
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_verification_detects_missing_changed_and_escaping_shards(tmp_path):
    manifest = prepare_corpus(tmp_path, languages=["hi"], splits=["train"], reader=lambda source, revision: [row("hi")])
    path = tmp_path / "manifest.json"
    assert load_manifest(path, verify_files=True) == manifest
    shard = tmp_path / manifest.shards[0].path
    original = shard.read_bytes()
    # Same-size corruption evades metadata-only checking but not checksum validation.
    shard.write_bytes(original.replace(b"first", b"other", 1))
    assert load_manifest(path, verify_files=True) == manifest
    with pytest.raises(ValueError, match="verification failed"):
        verify_manifest(path)
    shard.unlink()
    assert load_manifest(path) == manifest
    with pytest.raises(ValueError, match="missing or size mismatch"):
        load_manifest(path, verify_files=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.jsonl"
    outside.write_bytes(original)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        load_manifest(path, verify_files=True)


@pytest.mark.parametrize("mutation", ["path", "count", "duplicate"])
def test_manifest_rejects_invalid_paths_counts_and_duplicate_shards(tmp_path, mutation):
    prepare_corpus(tmp_path, languages=["hi"], splits=["train"], reader=lambda source, revision: [row("hi")])
    path = tmp_path / "manifest.json"
    value = json.loads(path.read_text())
    if mutation == "path":
        value["shards"][0]["path"] = "../escaped.jsonl"
    elif mutation == "count":
        value["counts"]["documents"] += 1
    else:
        value["shards"].append(value["shards"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("options", [{"revision": "main"}, {"rows_per_source": 0}, {"languages": ["de"]}, {"shard_rows": 0}])
def test_invalid_configuration_fails_before_reading_source(tmp_path, options):
    def forbidden_reader(source, revision):
        pytest.fail("invalid configuration must not read source")

    with pytest.raises(ValueError):
        prepare_corpus(tmp_path, reader=forbidden_reader, **options)
