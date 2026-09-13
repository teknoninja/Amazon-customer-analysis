"""Shared Groq LLM client (OpenAI-compatible) used across the pipeline."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

# Free-tier friendly defaults (override via .env)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Free tier ≈ 30 RPM AND ≈ 8K TPM. Keep batches small so TPM survives the minute.
GROQ_BATCH_SIZE = int(os.environ.get("GROQ_RPM", os.environ.get("GROQ_BATCH_SIZE", "12")))
GROQ_WINDOW_SECONDS = float(os.environ.get("GROQ_WINDOW_SECONDS", "60"))
# Tiny pause between calls inside a batch (helps RPM burst detection)
GROQ_INTRA_BATCH_GAP = float(os.environ.get("GROQ_INTRA_BATCH_GAP", "1.0"))
MAX_RATE_LIMIT_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "8"))
# TPM 429s need a long cool-down; Retry-After is often only 1–3s and is too short
MIN_429_SLEEP = float(os.environ.get("GROQ_MIN_429_SLEEP", "25"))

T = TypeVar("T", bound=BaseModel)

_batch_start: Optional[float] = None
_batch_count: int = 0
_last_call_ts: float = 0.0


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Create a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=0)


def _reset_batch() -> None:
    global _batch_start, _batch_count
    _batch_start = None
    _batch_count = 0


def _wait_for_batch_slot() -> None:
    """
    Send up to GROQ_BATCH_SIZE requests, then wait out the rest of the 60s window.

    Example with batch=12: ~12 calls, then sleep until 60s from batch start, repeat.
    """
    global _batch_start, _batch_count, _last_call_ts

    now = time.monotonic()

    # Start a new window if needed / expired
    if _batch_start is None or (now - _batch_start) >= GROQ_WINDOW_SECONDS:
        _batch_start = now
        _batch_count = 0

    # Batch full → sleep until the minute window ends, then start fresh
    if _batch_count >= GROQ_BATCH_SIZE:
        sleep_for = GROQ_WINDOW_SECONDS - (now - _batch_start) + 0.1
        if sleep_for > 0:
            logger.info(
                "Batch complete (%s calls). Waiting %.1fs for next minute window...",
                GROQ_BATCH_SIZE,
                sleep_for,
            )
            time.sleep(sleep_for)
        _batch_start = time.monotonic()
        _batch_count = 0

    # Small gap between calls inside the batch
    gap = time.monotonic() - _last_call_ts
    if _last_call_ts and gap < GROQ_INTRA_BATCH_GAP:
        time.sleep(GROQ_INTRA_BATCH_GAP - gap)

    _batch_count += 1
    _last_call_ts = time.monotonic()


def _compact_schema(response_model: Type[BaseModel]) -> str:
    """Short schema text — full JSON Schema wastes free-tier TPM."""
    fields = []
    for name, field in response_model.model_fields.items():
        ann = getattr(field, "annotation", str)
        fields.append(f'  "{name}": <{ann}>')
    joined = ",\n".join(fields)
    return "{\n" + joined + "\n}"


def _retry_after_seconds(err: RateLimitError, attempt: int) -> float:
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    header_wait = 0.0
    if retry_after:
        try:
            header_wait = float(retry_after)
        except ValueError:
            header_wait = 0.0
    # Exponential floor; never shorter than MIN_429_SLEEP (TPM recovery)
    backoff = min(90.0, MIN_429_SLEEP * attempt)
    return max(header_wait, backoff, MIN_429_SLEEP)


def structured_completion(
    prompt: str,
    response_model: Type[T],
    *,
    temperature: float = 0.2,
    system: Optional[str] = None,
) -> T:
    """
    Call Groq and parse the response into a Pydantic model via JSON object mode.
    """
    client = get_openai_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                "Respond with a single JSON object matching this shape:\n"
                f"{_compact_schema(response_model)}"
            ),
        }
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        _wait_for_batch_slot()
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from Groq")
            return response_model.model_validate_json(content)
        except RateLimitError as e:
            last_err = e
            # Don't count failed calls against the batch budget
            global _batch_count
            _batch_count = max(0, _batch_count - 1)
            wait_s = _retry_after_seconds(e, attempt)
            logger.warning(
                "Groq rate limit (429). Cooling down %.1fs then retry %s/%s "
                "(likely tokens/min, not just requests/min)...",
                wait_s,
                attempt,
                MAX_RATE_LIMIT_RETRIES,
            )
            time.sleep(wait_s)
            _reset_batch()

    raise RuntimeError(
        f"Groq rate limit persisted after {MAX_RATE_LIMIT_RETRIES} retries. "
        "Wait a minute and re-run — golden set resumes from the last checkpoint."
    ) from last_err
