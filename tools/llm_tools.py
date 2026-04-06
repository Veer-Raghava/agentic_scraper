"""
tools/llm_tools.py — Thin, opinionated wrapper around the OpenAI Python SDK.

Why a wrapper instead of calling openai directly?
  • Centralises retry logic (tenacity) so every agent benefits automatically.
  • Provides a clean interface: complete() for text, embed() for vectors,
    complete_json() for structured extraction.
  • Makes it trivial to swap providers later — just change this file.
  • Normalises all errors into a single AgentError so callers don't need to
    handle openai-specific exceptions.

Usage (in any agent):
    from tools.llm_tools import LLMClient
    client = LLMClient()
    text   = client.complete(system="...", user="...")
    data   = client.complete_json(system="...", user="...")
    vecs   = client.embed(["text1", "text2"])
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from rich.console import Console

import config as cfg

console = Console()


# ── Custom exception ──────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""


# ── Retry decorator shared by all calls ───────────────────────────────────────
# Retries on RateLimitError and APIConnectionError with exponential backoff.
# After 4 attempts (~30 s total) it raises LLMError.

_RETRY = retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)


# ── JSON extraction helpers ───────────────────────────────────────────────────

def _parse_json_response(text: str) -> Any:
    """
    Robustly extract JSON from an LLM response that may contain:
      - Leading prose ("Here is the JSON: ...")
      - Markdown fences (```json ... ```)
      - Trailing explanation
      - Unquoted N/A values

    Returns the parsed Python object or raises ValueError.
    """
    # Strip markdown fences first
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Fix a very common LLM mistake: unquoted N/A
    text = re.sub(r":\s*N/A\s*([,\}\]])", r': "N/A"\1', text)

    # Find the first JSON structure
    for opener, closer in [("[", "]"), ("{", "}")]:
        si = text.find(opener)
        if si == -1:
            continue
        ei = text.rfind(closer)
        if ei > si:
            chunk = text[si : ei + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                # Try with closing bracket appended (truncated response)
                try:
                    return json.loads(chunk + closer)
                except json.JSONDecodeError:
                    continue

    raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")


# ── Main client class ─────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified LLM client for all agents.

    One instance is created at startup and shared — it is stateless so
    sharing is safe and avoids redundant HTTP connection setup.
    """

    def __init__(self) -> None:
        if not cfg.OPENAI_API_KEY:
            raise LLMError(
                "OPENAI_API_KEY is missing. Set it in .env or the environment."
            )
        self._client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)

    # ── Core completion ──────────────────────────────────────────────────────

    @_RETRY
    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Send a single-turn chat completion and return the response text.

        Args:
            system      — The system prompt (agent persona + task instructions).
            user        — The user message (the data/query to process).
            model       — Override the default extraction model.
            max_tokens  — Override the default token limit.
            temperature — Override the default temperature.

        Returns:
            The assistant's response as a plain string.

        Raises:
            LLMError if all retries fail.
        """
        _model = model or cfg.EXTRACTION_MODEL
        _max_t = max_tokens or cfg.MAX_TOKENS_EXTRACTION
        _temp  = temperature if temperature is not None else cfg.TEMPERATURE

        try:
            response = self._client.chat.completions.create(
                model=_model,
                temperature=_temp,
                max_tokens=_max_t,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content.strip()

        except (openai.RateLimitError, openai.APIConnectionError,
                openai.APITimeoutError):
            raise  # let tenacity retry
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

    def complete_fast(self, *, system: str, user: str) -> str:
        """
        Convenience method using the cheaper/faster model.
        Use for lightweight tasks: query generation, summaries, routing.
        """
        return self.complete(
            system=system,
            user=user,
            model=cfg.FAST_MODEL,
            max_tokens=cfg.MAX_TOKENS_FAST,
        )

    # ── JSON extraction ───────────────────────────────────────────────────────

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Like complete() but parses the response as JSON.

        The system prompt should instruct the LLM to return ONLY JSON.
        This method adds a safety reminder to the system prompt and robustly
        parses the response even if the LLM adds prose.

        Returns:
            Parsed Python object (list, dict, etc.)

        Raises:
            LLMError if the LLM call fails.
            ValueError if the response is not valid JSON after parsing.
        """
        # Add a hard reminder that enforces JSON-only output
        json_reminder = (
            "\n\nCRITICAL: Your response must be ONLY valid JSON. "
            "No preamble, no explanation, no markdown fences. "
            "Start your response with [ or { immediately."
        )
        raw = self.complete(
            system=system + json_reminder,
            user=user,
            model=model,
            max_tokens=max_tokens,
        )
        try:
            return _parse_json_response(raw)
        except ValueError as exc:
            if cfg.DEBUG:
                console.print(f"[dim red]JSON parse failed. Raw:\n{raw[:800]}[/dim red]")
            raise

    # ── Multi-turn chat ───────────────────────────────────────────────────────

    def chat(
        self,
        *,
        system: str,
        messages: list[dict],  # [{"role": "user"/"assistant", "content": "..."}]
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Multi-turn completion using a full message history.
        Used by the Orchestrator for the interactive chat loop.
        """
        _model = model or cfg.FAST_MODEL
        _max_t = max_tokens or cfg.MAX_TOKENS_FAST

        full_messages = [{"role": "system", "content": system}] + messages

        try:
            response = self._client.chat.completions.create(
                model=_model,
                temperature=cfg.TEMPERATURE,
                max_tokens=_max_t,
                messages=full_messages,
            )
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as exc:
            raise LLMError(f"Chat completion failed: {exc}") from exc

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """
        Get embeddings for a list of texts.

        Truncates each text to 8000 characters to stay within token limits.
        Returns None on failure so callers can gracefully fall back to
        structural scoring.

        Returns:
            List of embedding vectors, or None if embeddings are unavailable.
        """
        if not texts:
            return []
        try:
            truncated = [t[:8000] for t in texts]
            response = self._client.embeddings.create(
                input=truncated,
                model=cfg.EMBEDDING_MODEL,
            )
            return [item.embedding for item in response.data]
        except openai.OpenAIError as exc:
            console.print(
                f"  [dim yellow]⚠ Embeddings unavailable ({exc}), "
                "falling back to structural scoring[/dim yellow]"
            )
            return None

    # ── Cosine similarity (no scipy dependency) ───────────────────────────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Pure-Python cosine similarity between two vectors."""
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)