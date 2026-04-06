"""Minimal LLM client with safe fallbacks.

Works with OpenAI if configured, otherwise returns deterministic local fallbacks
so the CLI keeps functioning.
"""

from __future__ import annotations

import json
import re
from typing import Any

import config as cfg

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class LLMClient:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=cfg.OPENAI_API_KEY) if (OpenAI and cfg.OPENAI_API_KEY) else None

    def chat(self, system: str, messages: list[dict[str, str]], model: str, max_tokens: int = 300) -> str:
        if self._client:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=max_tokens,
                temperature=cfg.TEMPERATURE,
            )
            return (resp.choices[0].message.content or "").strip()
        # deterministic fallback
        return "Got it. Please type 'go' to run the pipeline, or 'help' to see commands."

    def complete_json(self, system: str, user: str, model: str, max_tokens: int = 400) -> Any:
        if self._client:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            txt = (resp.choices[0].message.content or "{}").strip()
            try:
                return json.loads(txt)
            except Exception:
                pass

        # fallback intent / suggestions
        u = user.lower()
        if "suggest 8-12 extraction columns" in u:
            topic = user.split('Topic: "')[-1].split('"')[0].lower()
            if "antibody" in topic or "adc" in topic:
                return [
                    "Antibody Name", "Antigen Name", "Linker", "Payload Name",
                    "DAR", "Indication", "Approval Year", "Source URL",
                ]
            return ["Name", "Description", "Year", "Source URL"]

        msg = user.replace("User message:", "").strip().lower()
        if msg in {"go", "yes", "ok", "all", "use all", "y"}:
            return {"intent": "confirm_columns"}
        if any(k in msg for k in ["export xlsx", "xlsx"]):
            return {"intent": "export", "export_format": "xlsx"}
        if any(k in msg for k in ["export csv", "csv"]):
            return {"intent": "export", "export_format": "csv"}
        if any(k in msg for k in ["show", "fill report", "results"]):
            return {"intent": "show_results"}
        if any(k in msg for k in ["find more", "more sources"]):
            n = re.search(r"(\d+)", msg)
            return {"intent": "more_sources", "source_limit": int(n.group(1)) if n else None}
        if any(k in msg for k in ["change topic", "new topic"]):
            return {"intent": "change_topic", "topic": msg}
        if any(k in msg for k in ["quit", "exit", "stop"]):
            return {"intent": "stop"}
        if "http://" in msg or "https://" in msg:
            return {"intent": "custom_urls"}
        if ".pdf" in msg:
            return {"intent": "local_pdfs"}

        cols = []
        m = re.search(r"columns?.*?:\s*(.+)$", msg)
        if m:
            cols = [c.strip().title() for c in m.group(1).split(",") if c.strip()]
        return {"intent": "start_pipeline", "topic": msg, "columns": cols or None}
