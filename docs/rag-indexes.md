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

Connect immutable retrieval-bundle loading and measure a lighter query-embedding
runtime, including all three indexes and realistic queries. The current
Sentence Transformers/PyTorch path has not passed the 512 MB Render Free budget.
Do not enable it there or buy a larger instance. If an optimized runtime cannot
fit with headroom while preserving retrieval quality, choose another genuinely
free serving arrangement before deployment. Larger-corpus and end-to-end answer
evaluation remain separate gates.
