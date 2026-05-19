"""LLM client abstraction.

Four modes, picked in this order based on env vars:

  1. OpenRouter (`OPENROUTER_API_KEY`) — routes to
     `google/gemini-3-flash-preview` by default.  Higher rate limit than
     the Google free tier, so this is the preferred backend when both
     keys are present.
  2. Anthropic Claude (`ANTHROPIC_API_KEY`) — claude-sonnet-4-6 via the
     official SDK with prompt caching on long system blocks.
  3. Google Gemini direct (`GEMINI_API_KEY` / `GOOGLE_API_KEY`) — same
     model PKGPT uses, but free tier is capped at 20 req/day/model.
  4. Deterministic mode — rule-based decisions that encode the same
     pharmacometric heuristics the LLM prompts target.  Active when no
     key is configured.

Set EITHER `OPENROUTER_API_KEY` or `GEMINI_API_KEY` in a `.env` at the
project root (or in the shell environment) before running the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"


@dataclass
class AgentMessage:
    role: str             # "system", "user", "assistant"
    content: str
    cacheable: bool = False


@dataclass
class AgentRunLog:
    name: str
    inputs: dict
    output: Any
    notes: list[str] = field(default_factory=list)


class LLMClient:
    """Provider-agnostic LLM client with JSON-output helpers."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        # decide provider — OpenRouter first (higher rate limit), then
        # Anthropic direct, then Gemini direct, then deterministic.
        or_key = os.environ.get("OPENROUTER_API_KEY")
        anth_key = os.environ.get("ANTHROPIC_API_KEY")
        gem_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if HAS_REQUESTS and or_key:
            self.provider = "openrouter"
            self.model = model or DEFAULT_OPENROUTER_MODEL
            self.api_key = or_key
            self._client = None   # use requests directly, no SDK
            self.enabled = True
        elif HAS_ANTHROPIC and anth_key:
            self.provider = "anthropic"
            self.model = model or DEFAULT_ANTHROPIC_MODEL
            self.api_key = anth_key
            self._client = anthropic.Anthropic(api_key=anth_key)
            self.enabled = True
        elif HAS_GEMINI and gem_key:
            self.provider = "gemini"
            self.model = model or DEFAULT_GEMINI_MODEL
            self.api_key = gem_key
            genai.configure(api_key=gem_key)
            self._client = genai.GenerativeModel(self.model)
            self.enabled = True
        else:
            self.provider = "none"
            self.model = None
            self.enabled = False
            self._client = None

    def chat_json(self, system_blocks: list[str], user: str,
                  max_tokens: int = 2048, temperature: float = 0.2) -> dict:
        """Send a system+user prompt and parse the response as JSON."""
        if not self.enabled:
            return {}
        if self.provider == "openrouter":
            # OpenAI-compatible chat completions API at openrouter.ai.
            messages = [{"role": "system", "content": blk}
                        for blk in system_blocks]
            messages.append({"role": "user", "content": user})
            try:
                resp = _requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                              "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": messages,
                          "temperature": temperature,
                          "max_tokens": max_tokens,
                          "response_format": {"type": "json_object"}},
                    timeout=90,
                )
                if resp.status_code != 200:
                    return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return _parse_json_block(text)
            except Exception as e:
                return {"error": str(e)}
        if self.provider == "anthropic":
            system = []
            for blk in system_blocks:
                entry = {"type": "text", "text": blk}
                if len(blk) > 1024:
                    entry["cache_control"] = {"type": "ephemeral"}
                system.append(entry)
            try:
                resp = self._client.messages.create(
                    model=self.model, system=system,
                    max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                return _parse_json_block(text)
            except Exception as e:
                return {"error": str(e)}
        if self.provider == "gemini":
            combined = "\n\n".join(system_blocks) + "\n\n---\n\n" + user
            try:
                resp = self._client.generate_content(
                    combined,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                    },
                )
                text = resp.text
                return _parse_json_block(text)
            except Exception as e:
                return {"error": str(e)}
        return {}


    def chat_text(self, system_blocks: list[str], user: str,
                   max_tokens: int = 2048, temperature: float = 0.4) -> str:
        """Send a system+user prompt and return the raw text response.

        Used by agents that need free-form output (e.g. the manuscript
        writer that returns a full Markdown document, not JSON).
        """
        if not self.enabled:
            return ""
        if self.provider == "openrouter":
            messages = [{"role": "system", "content": blk}
                        for blk in system_blocks]
            messages.append({"role": "user", "content": user})
            try:
                resp = _requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                              "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": messages,
                          "temperature": temperature,
                          "max_tokens": max_tokens},
                    timeout=300,
                )
                if resp.status_code != 200:
                    return f"[HTTP {resp.status_code}: {resp.text[:300]}]"
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                return f"[error: {e}]"
        if self.provider == "anthropic":
            system = []
            for blk in system_blocks:
                entry = {"type": "text", "text": blk}
                if len(blk) > 1024:
                    entry["cache_control"] = {"type": "ephemeral"}
                system.append(entry)
            try:
                resp = self._client.messages.create(
                    model=self.model, system=system,
                    max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in resp.content
                                if hasattr(b, "text"))
            except Exception as e:
                return f"[error: {e}]"
        if self.provider == "gemini":
            combined = "\n\n".join(system_blocks) + "\n\n---\n\n" + user
            try:
                resp = self._client.generate_content(
                    combined,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                return resp.text
            except Exception as e:
                return f"[error: {e}]"
        return ""


def _parse_json_block(text: str) -> dict:
    """Extract the first JSON object from a markdown-fenced or plain response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw": text}


# A single shared client for the whole pipeline.
SHARED = LLMClient()
