import os
import math
import random
import time
from typing import Any, Dict, List, Optional
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .config import (
    CONFIG,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
)

# cached singleton client instance
_client: Optional[OpenAI] = None
_active_provider: Optional[str] = None
_active_key: Optional[str] = None


def get_api_key_and_provider(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str, str]:
    # resolve provider and base url
    chosen_provider = (provider or CONFIG.get("provider", "groq")).lower()

    # if api key is provided directly
    if api_key:
        if chosen_provider == "groq":
            return api_key, "groq", GROQ_BASE_URL
        return api_key, "openrouter", OPENROUTER_BASE_URL

    # check environment variables
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if chosen_provider == "groq" and groq_key:
        return groq_key, "groq", GROQ_BASE_URL
    if chosen_provider == "openrouter" and openrouter_key:
        return openrouter_key, "openrouter", OPENROUTER_BASE_URL

    # fallback to whichever key is present
    if groq_key:
        return groq_key, "groq", GROQ_BASE_URL
    if openrouter_key:
        return openrouter_key, "openrouter", OPENROUTER_BASE_URL

    # nothing found — return empty key, caller will raise
    return "", chosen_provider, GROQ_BASE_URL


def get_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAI:
    # returns or initializes the openai client
    global _client, _active_provider, _active_key
    key, resolved_provider, base_url = get_api_key_and_provider(provider, api_key)

    if not key:
        env_var = "GROQ_API_KEY" if resolved_provider == "groq" else "OPENROUTER_API_KEY"
        raise RuntimeError(
            f"API key missing for provider '{resolved_provider}'.\n"
            f"Please set {env_var} in your .env file.\n"
            f"Get a free Groq key at: https://console.groq.com/keys"
        )

    # reuse existing client if same provider and no explicit override
    if _client is not None and _active_key == key and _active_provider == resolved_provider:
        return _client

    timeout = float(CONFIG.get("request_timeout_seconds", 90.0))
    _client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)
    _active_provider = resolved_provider
    _active_key = key
    return _client


def _retry_after(exc: Exception) -> Optional[float]:
    # parse retry-after header if provided by api
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        delay = float(raw) if raw is not None else None
        return delay if delay is not None and math.isfinite(delay) and delay >= 0 else None
    except (ValueError, TypeError):
        return None


def _is_retryable(exc: Exception) -> bool:
    # determine if an exception represents a transient failure worth retrying
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        return code == 429 or (code is not None and code >= 500)
    return False


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None,
    seed: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    # execute chat completion with exponential backoff and jitter on transient errors
    # returns the text content of the response
    result = call_llm_with_usage(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
        seed=seed,
        response_format=response_format,
    )
    return result["content"]


def call_llm_with_usage(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    client: Optional[OpenAI] = None,
    seed: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # execute chat completion with exponential backoff and jitter on transient errors
    # returns dict with 'content' (str) and 'usage' (prompt_tokens, completion_tokens, total_tokens)
    active_client = client if client is not None else get_client()

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = int(seed)
    if response_format is not None:
        kwargs["response_format"] = response_format

    max_retries = int(CONFIG.get("max_retries", 4))
    base_backoff = float(CONFIG.get("base_backoff_seconds", 3.0))
    max_backoff = float(CONFIG.get("max_backoff_seconds", 60.0))

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = active_client.chat.completions.create(**kwargs)
            content_text = (response.choices[0].message.content or "").strip()

            # if content is empty and we have retries left, wait and retry
            if not content_text and attempt < max_retries:
                time.sleep(min(max_backoff, base_backoff * (2 ** attempt)))
                continue

            # extract token usage if available (some providers omit it)
            usage = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return {"content": content_text, "usage": usage}

        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise

            wait = _retry_after(exc)
            if wait is None:
                wait = min(max_backoff, base_backoff * (2 ** attempt) + random.uniform(0.0, 1.0))
            print(f"  retrying in {wait:.1f}s after error: {type(exc).__name__}")
            time.sleep(wait)

    # only reached if all retries exhausted without raising
    raise RuntimeError(f"All {max_retries} retries exhausted: {last_exc}")
