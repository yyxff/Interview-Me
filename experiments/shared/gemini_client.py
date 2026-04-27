"""
Shared Gemini API client for eval scripts.

Usage:
    from shared.gemini_client import flash, pro

    response = flash("your prompt here")
    response = pro("your prompt here")
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from google import genai

# Search for .env in cwd, then experiments subdirs, then repo root
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())
from google.genai import types

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _call(model_name: str, prompt: str, system: str = "", json_mode: bool = False) -> str:
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    return response.text


def flash(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Cheap, fast — use for persona answer generation and bulk annotation."""
    model = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
    return _call(model, prompt, system, json_mode)


def pro(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Stronger — use for LLM-as-Judge."""
    model = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.5-pro-preview-06-05")
    return _call(model, prompt, system, json_mode)
