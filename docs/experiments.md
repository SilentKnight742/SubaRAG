# Retrieval experiments

This document records measured results, not target numbers. The raw reports and
embedding caches live under the ignored `artifacts/` directory; the compact
summary below is kept in source control so architecture decisions remain
reviewable.

## Method

- Source: MSMARCO-XI at revision
  `bf5cdc1f26e581e519018e434db14edd1b77602b`.
- Corpus: the first 1,000 validation source rows, producing roughly 10,000
  retrievable passages per language and 546 queries whose relevance labels are
  present inside each selected corpus slice.
- Leakage boundary: only `documents/` shards were indexed. Query, answer, and
  relevance rows stayed in `evaluation/` and were used only for scoring.
- Retrieval metrics: Recall@10, MRR@10, and nDCG@10.
- Latency metrics: warm-process p50, p95, and p99 on an 8-core x86_64 CPU with
  7.8 GiB RAM. They are local comparison numbers, not a hosting SLA.
- Selection rule: choose the lowest-latency candidate within 2% of the best
  nDCG@10, then validate policies on alternating calibration/evaluation rows.

The benchmark can be repeated with:

```sh
cd backend
.venv/bin/python -m subarag.experiments.retrieval \
  --manifest ../artifacts/corpus-validation-1000/manifest.json \
  --output ../artifacts/bench-e5-small-10k \
  --language en \
  --documents 10000 \
  --queries 1000
```

Use `hi` and `mr` for the other language runs. `subarag-rerank-benchmark`
repeats the English cross-encoder ablation after the retrieval benchmark has
written its embedding cache.

## Encoder screen

The comparable 2,000-document English screen contained 105 evaluable queries.

| Encoder | Dimensions | nDCG@10 | Query embedding p50 | Query embedding p95 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| multilingual-e5-small | 384 | 0.7204 | 15.1 ms | 48.6 ms | Selected |
| multilingual-e5-base | 768 | 0.7071 | 40.9 ms | 69.0 ms | Slower and lower quality |
| BGE-M3 | 1,024 | 0.6959 | 155.0 ms | 281.5 ms | Far too slow here |

E5-small was both the quality winner and the latency winner in this screen.
BGE-M3 also consumed about 2.9 GiB RSS after embeddings, compared with about
1.4 GiB for E5-small.

## 10K multilingual retrieval

| Language | Documents | Exact nDCG@10 | HNSW nDCG@10 | HNSW Recall@10 | HNSW search p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 9,941 | 0.7140 | 0.7123 | 0.9734 | 0.52 ms |
| Hindi | 9,942 | 0.6058 | 0.6058 | 0.8610 | 0.56 ms |
| Marathi | 9,941 | 0.5158 | 0.5159 | 0.7493 | 0.55 ms |

HNSW reproduced exact-search quality within 0.003 nDCG in every language while
keeping index lookup below one millisecond at p95. Query embedding, not ANN
lookup, is the retrieval bottleneck at this scale.

## Policy ablations

### Context size

Fixed top-8 was selected independently in all three 10K runs. On the calibration
halves, top-8 reached 99.7%, 99.6%, and 99.0% of top-12 nDCG for English, Hindi,
and Marathi respectively. The eligible adaptive policy averaged 10.7, 11.2, and
11.6 passages, so it used more context rather than less.

Decision: use fixed `k=8`. Keep adaptive-k in the experiment package; do not put
it on the serving path without new held-out evidence.

### Lexical and hybrid retrieval

| Language | Dense nDCG@10 | BM25 nDCG@10 | Calibrated route |
| --- | ---: | ---: | --- |
| English | 0.7123 | 0.5254 | Dense only |
| Hindi | 0.6058 | 0.3950 | Dense only |
| Marathi | 0.5159 | 0.3136 | Dense only |

Always-on reciprocal-rank fusion reduced quality in the 10K runs. The corrected
calibrator allowed “do nothing” as a candidate and chose dense-only in every
language. Unicode-safe BM25 and conditional fusion remain experimental tools,
not production dependencies.

### Cross-encoder reranking

On the 2K English screen, `ms-marco-MiniLM-L6-v2` raised nDCG@10 from 0.7204 to
0.7498. It cost 592.8 ms p50 and 1,453.7 ms p95 per 32-candidate query. A
margin-calibrated gate retained the gain on the held-out half but still fired on
61.5% of queries.

Decision: no default reranker. The tested reranker is also English-only; a future
multilingual reranker must demonstrate a materially better quality-per-millisecond
trade-off before entering the serving path.

### Compression, caching, and voice speculation

- Extractive spans reduced the token proxy by 37.6% in English, 44.1% in Hindi,
  and 68.3% in Marathi. This does not yet prove downstream answer quality, so
  compression preserves document identity and remains independently switchable.
- Versioned exact-query retrieval caching reduced the measured warm lookup to
  roughly 0.01 ms p95. Generated answers are deliberately not cached here.
- A text-prefix proxy showed the same top result only 77.5% of the time for an
  80%-complete English query, 70% for Hindi, and 67.5% for Marathi. At 40%-60%
  completion it was substantially worse.

Decision: final transcripts are authoritative. Do not generate from early voice
partials. A later streaming-ASR integration may prefetch retrieval only after a
stable late partial, and must invalidate it when the final normalized transcript
changes.

## Scale boundary and honest limitations

The local artifact contains about 10K documents per language. Real 100K and 1M
quality runs were not fabricated by duplicating embeddings: the available
machine does not have enough disk for the full 55+ GiB source files. The harness
accepts larger prepared artifacts, so those runs belong on remote compute after
artifact storage and licensing are settled.

ScaNN was not promoted into the implementation because HNSW lookup is already
about 0.5 ms p95 here; changing ANN libraries cannot remove the 16-86 ms query
embedding cost. Revisit index families only when a real larger-corpus trace shows
ANN search is a meaningful part of end-to-end p95.

Generation TTFT, factuality, citation precision, and end-to-end answer quality
remain unmeasured for a production candidate because no generator passed the
first local gate. Retrieval metrics must not be presented as answer-quality
metrics.

## Generator bake-off

The generator gate used labeled evidence rather than retrieved evidence so it
could test generation independently of retrieval errors. Each query was paired
with an answerable case containing relevant evidence plus distractors and a
no-answer case containing only distractors. The model had to return strict JSON,
use short evidence aliases, cite a relevant source, and refuse unsupported
questions.

`Qwen/Qwen2.5-0.5B-Instruct` failed both corrected English gate cases:

| Measure | Result |
| --- | ---: |
| Valid JSON/schema rate | 0% |
| Grounded citation rate | 0% |
| No-answer accuracy | 0% |
| Time to first token, p50 | 9.38 s |
| Total generation time, p50 | 13.76 s |
| Total generation time, p95 | 18.66 s |

It returned a refusal despite directly supporting evidence, then answered the
matched distractor-only case from outside knowledge. Both responses also broke
the citation schema. An initial multilingual pass showed the same format and
grounding failures and was stopped early rather than tuned against the test set.

The 1.5B candidate was eliminated by the staged latency gate before inference.
It uses the same Qwen2.5 decoder family with three times the parameters; once the
0.5B model's first token already took seconds rather than hundreds of
milliseconds on the target CPU class, the larger model could not satisfy the
project's quick-response requirement even if its answer quality improved. No
latency or quality number is claimed for the 1.5B model.

Decision: no local Transformers generator is selected and the application stays
fail-closed with `UnavailableRagEngine`. The next generator comparison should be
between low-latency hosted APIs or a deliberately optimized quantized runtime,
using the same frozen gate. A weak local model will not be wired merely to make
the demo appear complete.
