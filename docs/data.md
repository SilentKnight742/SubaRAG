# Corpus preparation

Step 1 produces passage files and separate evaluation files. It does not produce
embeddings, an index, retrieved results, or generated answers. Preparation runs
on remote compute; this VM needs only the code and synthetic tests.

## Source inspected

The public [MSMARCO-XI repository](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
was inspected at revision `bf5cdc1f26e581e519018e434db14edd1b77602b`.
Its actual files for our initial languages are:

| Language | Training | Validation |
| --- | --- | --- |
| Hindi | `train/hintrain.parquet` | `validation/hinval.parquet` |
| Marathi | `train/martrain.parquet` | `validation/marval.parquet` |
| English | `English_passages`, `Eng_Query`, and `Eng_Answer` within the above files | Same |

The card's named-language loading example and custom loader refer to JSONL paths
that do not match the files. The pipeline uses the standard Hugging Face Parquet
streaming reader with explicit, revision-pinned URLs. It executes no code from
the dataset repository. English-only runs use the Hindi file as their source.

The card and loader describe `passages.English_passages`,
`passages.Translated_passages`, and `passages.is_selected` as aligned arrays.
The preparation code validates these fields, query IDs, and language tags. It
accepts a nested passage object or a JSON-encoded object. Unexpected records go
to a rejection file with source location and reason, without copying their text.
The actual remote pilot still needs to run; published metadata inspection and
synthetic tests do not prove every real row follows this schema.

## Run the pilot in Colab

Open [Google Colab](https://colab.research.google.com/), sign in, and upload
`notebooks/prepare_data.ipynb`. Use a normal CPU runtime. No GPU or API key is
needed to read this public source.

The notebook asks you to upload `SubaRAG-source.zip`, a source-only archive of the
completed local branch. This works before anything is pushed to GitHub. A source
archive can be created from the repository with:

```sh
git archive --format=zip --prefix=SubaRAG/ --output=/tmp/SubaRAG-source.zip HEAD
```

`git archive` includes committed source only; ignored `.env` files, dependency
directories, and keys are excluded. Use the completed foundation commit, not an
older commit missing the data package.

Run notebook cells in order. It installs the package and runs:

```sh
python -m subarag.data prepare \
  --output /content/subarag-data-pilot \
  --languages en hi mr \
  --rows-per-source 1000
python -m subarag.data verify /content/subarag-data-pilot/manifest.json
```

The notebook uses fresh, uniquely named source and output folders on each run,
so retries preserve partial work. Command output is streamed into the notebook.

The bound is **1,000 rows per source file**, across four source files, not 1,000
passages or the full dataset. The reader uses 32-row batches, disables Arrow
pre-buffering, enables 64 KiB buffered streams, and selects only needed columns.
This requires `datasets>=4.2`. Small batches alone do not prevent eager reads:
footer inspection of the pinned Hindi files found a single row group in each,
about 1.16 GB uncompressed for validation and 9.73 GB for training. These options
reduce eager buffering, but are not a hard process-memory cap: Parquet pages,
dictionaries and filesystem caches still consume memory. The pilot is
deterministic first rows, not a representative random benchmark. Inspect counts
and rejection reasons before increasing scale. Do not assess retrieval quality
using this sample alone.

The last cell downloads only the small manifest. Keep the pilot shards in Colab
until we choose the artifact repository and upload them. Downloading the corpus
to this VM is unnecessary. Colab storage is temporary: do not rely on it for a
long full-corpus build or persistent artifacts. No cloud run or upload has been
performed by creating this notebook.

## Artifact contract

```text
output/
  manifest.json                  # written last, only on a completed preparation
  documents/en-00000.jsonl        # additional shards and languages as needed
  documents/hi-00000.jsonl
  documents/mr-00000.jsonl
  evaluation/en-validation-00000.jsonl
  evaluation/hi-train-00000.jsonl
  rejections/source-00000.jsonl   # only when there are invalid rows
  _work/dedup.sqlite3             # local preparation state; do not upload
```

Documents contain `id`, `language`, `text`, and first-occurrence source location.
Every candidate passage is eligible, including unselected passages. Queries,
answers, and relevance labels exist only in `evaluation/`, whose rows reference
document IDs. The corpus can contain validation passages: held-out **queries and
answers** are evaluation data, while their candidate passages must be retrievable.
Step 2 must index only `documents/`; it must not embed evaluation files.

Passage IDs hash language plus Unicode NFC and whitespace-normalized text. This
deduplicates repeated English content across language files without treating
different languages or text case as identical. A disk-backed SQLite key store
keeps corpus-sized sets out of memory. Source locations refer to zero-based row
and passage offsets in the pinned source file. Additional occurrences are
deduplicated; the retained provenance is the first occurrence, not an exhaustive
list of every source row containing that passage.

Validation files are processed first. A query ID seen in validation is excluded
from training evaluation output across all selected languages, including when
another field in its validation row is invalid. This policy covers the selected
rows: a bounded pilot does not inspect unseen validation rows. Repeated query
IDs in the same language keep their first evaluation record. Train and validation
evaluation files must never be mixed when tuning and reporting retrieval quality.

The schema-v1 manifest declares `artifact_kind: prepared_corpus`, pinned source
files, requested languages/splits, sample bound, preparation policies, aggregate
counts, and per-shard row count, byte count, and SHA-256. It makes no assertion
that an index is ready. `load_manifest(path)` validates metadata without reading
shards; `load_manifest(path, verify_files=True)` also checks containment,
existence, and byte sizes. The CLI `verify` streams every shard to check hashes
and row counts. File-stat verification alone does not detect same-size corruption.

Output directories must be empty. A failure leaves partial files without a
manifest. Use a fresh output path after an interrupted run; resume is not
implemented. The JSONL shards and manifest are the portable artifact; SQLite
state is excluded. Publishing those artifacts is a separate explicit action.

## Later full preparation

Continue on free Colab CPU sessions. Explicit `--full` replaces the sample bound;
it is not the default. Check the pilot's time and disk usage first: free sessions
are temporary and are not guaranteed to fit the full corpus. If necessary, split
preparation into bounded jobs in a later slice instead of renting compute.
Persist completed artifacts to a private Hugging Face dataset repository within
its free storage allowance, then verify checksums after transfer. Do not start
a paid job or enable billing.

The source card refers users to the original MS MARCO license terms. This code
does not assign a new license to derived data or authorize public redistribution.
Keep generated artifacts private pending that review, retain source attribution,
and never include API keys in the notebook, artifacts, or source archive.
