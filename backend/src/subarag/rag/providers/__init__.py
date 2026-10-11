"""Hosted answer-generator adapters."""

from .gemini import GeminiAnswerGenerator, create_generation_http_client

__all__ = ["GeminiAnswerGenerator", "create_generation_http_client"]
