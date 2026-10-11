"""Low-cost extractive context reduction that preserves source identity."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .lexical import unicode_tokens
from .models import RetrievalHit


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\s+|\n+")


def _score_sentence(query_tokens: set[str], sentence: str) -> tuple[float, int]:
    tokens = unicode_tokens(sentence)
    if not tokens:
        return 0.0, 0
    overlap = len(query_tokens.intersection(tokens))
    return overlap / (len(set(tokens)) ** 0.5), len(tokens)


def extract_supporting_spans(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    max_sentences_per_document: int = 2,
    max_total_tokens: int = 500,
) -> list[tuple[RetrievalHit, str]]:
    query_tokens = set(unicode_tokens(query))
    selected: list[tuple[RetrievalHit, str]] = []
    used_tokens = 0
    for hit in hits:
        sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(hit.document.text)
                     if sentence.strip()]
        ranked = sorted(
            ((_score_sentence(query_tokens, sentence), index, sentence)
             for index, sentence in enumerate(sentences)),
            key=lambda item: (-item[0][0], item[1]),
        )
        keep = sorted(ranked[:max_sentences_per_document], key=lambda item: item[1])
        for (_, token_count), _, sentence in keep:
            if used_tokens and used_tokens + token_count > max_total_tokens:
                return selected
            selected.append((hit, sentence))
            used_tokens += token_count
    return selected
