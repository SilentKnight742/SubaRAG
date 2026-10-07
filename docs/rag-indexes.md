# Colab serving-index build

This slice builds the existing E5/HNSW retrieval artifacts for the **entire
prepared pilot** (59,594 passages), not the full upstream dataset. It does not
change the live Render preview, call generation APIs, or promise that the existing
PyTorch runtime fits Render Free.

## Run remotely

1. Upload `notebooks/build_rag_indexes.ipynb` into a fresh free Colab runtime.
   CPU is supported; select a free T4 GPU if available for faster passage encoding.
   Do not purchase compute units. The notebook selects CUDA only when available.
2. Upload the new committed source archive when its first code cell prompts:

   ```sh
   git archive --format=zip --prefix=SubaRAG/ \
     --output=artifacts/colab-indexes/SubaRAG-index-source.zip HEAD
   ```

   Create the output directory first. Never archive the working directory or
   include `.env`, caches, keys, or existing generated artifacts. The notebook
   accepts Colab-renamed ZIPs and rejects the old data-preparation source ZIP.
3. Enter an HF token with read access to private `silentkirmada/SubaRAG-data`
   at the hidden prompt. It is used in memory and is not saved by a login command.
4. Run the build and verification cells. They read corpus commit
   `e29f5c990a4fc0d8e05771be4007f1c6560adfc2`, verify its shards, resolve and record
   an immutable model commit, then build English, Hindi, and Marathi sequentially.
   No source Parquet files or evaluation answers are embedded.
5. Share the downloaded `bundle.json`. Keep the Colab runtime open. If interrupted,
   rerun the build cell to reuse completed verified language indexes; incomplete
   index directories are not overwritten automatically.

The bundle contains three directories (`en`, `hi`, `mr`), each with
`retrieval-manifest.json`, `documents.jsonl`, and `index.faiss`. The report records
source archive, corpus manifest, model revision, build package versions, counts,
and file hashes. Verification reloads each saved index, checks checksums and
dimensions, and checks complete document counts against the prepared corpus.
This verifies artifacts, not answer quality or serving memory.

New retrieval manifests include the model commit. The runtime rejects mixed
model revisions and passes the pinned commit to the query embedder. Older
unpinned artifacts remain readable for local experiments. Before enabling the
offline RAG Docker target, its model cache must also be built with this exact
commit; the existing image's unpinned cache is not assumed compatible.

## Explicit upload, separate from deployment

After reviewing the report, the optional final notebook cell requires typing
`UPLOAD` and entering a dataset-scoped write token. It refuses a public dataset
repository, checks the files again, and uploads only the report and its nine
listed files to `retrieval/<bundle-sha256>/`. Existing root corpus files remain
unchanged. Save the printed commit SHA and path for the serving configuration.
Do not replace Render's read-only token with the upload token.

The live preview remains pinned to its existing corpus commit. No service
automatically consumes or deploys these new files.

## Following slice

Immutable retrieval-bundle loading is implemented, opt-in, using the three
`SUBARAG_RETRIEVAL_REPO`, `SUBARAG_RETRIEVAL_REVISION`, and
`SUBARAG_RETRIEVAL_PATH` settings. It also requires the existing read-only
`HF_TOKEN` and a configured `GEMINI_API_KEY`; it cannot be combined with local
retrieval manifest settings. With neither source configured, the API continues
to report RAG unavailable.

The verified pilot indexes were uploaded to private `silentkirmada/SubaRAG-data`:

- Commit: `9f6d55f53b9479930613e4c849e5e834a4230d0a`
- Path: `retrieval/e7a5c4e6bd668ccfeadbe8f2178d20f9a8570e51937d3592813af55ded2fac68`
- E5 model commit: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Documents: English 19,865; Hindi 19,862; Marathi 19,867.
- Total file bytes: 161,619,764 (154.1 MiB, excluding the bundle report).

Authenticated checks verified the uploaded report fingerprint and the three
language manifests. They did not download the full index bundle to the VM.
The loader verifies every file before publishing its local cache, and verifies
the cache again on reuse. It compares the index model/corpus metadata across the
report and each language manifest. When a prepared corpus is also configured,
its manifest fingerprint and remote provenance must match the bundle. Incomplete
configuration, corrupt data, or download failure prevents startup. Downloads
have a 256 MiB total file limit and five-minute timeout; errors do not expose
authorization headers or signed URLs.

The live Render configuration has **not** been changed. The default preview
image still has no embedding runtime; these settings must not be enabled there.
The source `.env.example` documents them as comments, not defaults.

Next, measure a lighter query-embedding runtime, including all three indexes and
realistic queries. The current
Sentence Transformers/PyTorch path has not passed the 512 MB Render Free budget.
Do not enable it there or buy a larger instance. If an optimized runtime cannot
fit with headroom while preserving retrieval quality, choose another genuinely
free serving arrangement before deployment. Larger-corpus and end-to-end answer
evaluation remain separate gates.
