"""Hugging Face Inference Providers integration."""

import logging
import os
from typing import Optional

import requests


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct:fastest"
DEFAULT_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful statistical methods assistant. Give concise, accurate "
    "answers, state important assumptions, and do not invent application "
    "features or research findings."
)


class HuggingFaceError(RuntimeError):
    """Raised when the configured AI provider cannot complete a request."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def is_ai_enabled() -> bool:
    """Return whether AI-backed features are enabled."""
    return os.environ.get("AI_ENHANCEMENT_ENABLED", "false").lower() == "true"


def get_huggingface_config() -> tuple[Optional[str], str]:
    """Return the configured token and Inference Providers model."""
    return (
        os.environ.get("HUGGINGFACE_API_KEY", "").strip() or None,
        os.environ.get("HUGGINGFACE_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL,
    )


def _timeout_seconds() -> float:
    raw_timeout = os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "45")
    try:
        return min(max(float(raw_timeout), 5.0), 55.0)
    except ValueError:
        return 45.0


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or error)[:300]
        if error:
            return str(error)[:300]
        if payload.get("message"):
            return str(payload["message"])[:300]
    return "The AI provider returned an error."


def call_huggingface_api(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate a chat response through Hugging Face Inference Providers."""
    if not is_ai_enabled():
        raise HuggingFaceError("AI features are currently disabled.", 503)

    api_key, configured_model = get_huggingface_config()
    if not api_key:
        raise HuggingFaceError(
            "HUGGINGFACE_API_KEY is required when AI features are enabled.",
            503,
        )

    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise ValueError("An AI prompt is required.")

    target_model = (model or configured_model).strip()
    api_url = os.environ.get("HUGGINGFACE_API_URL", DEFAULT_API_URL).strip()
    payload = {
        "model": target_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": cleaned_prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=(5, _timeout_seconds()),
        )
    except requests.exceptions.Timeout as exc:
        raise HuggingFaceError("The AI provider timed out.", 504) from exc
    except requests.exceptions.ConnectionError as exc:
        raise HuggingFaceError("Could not connect to the AI provider.", 503) from exc
    except requests.exceptions.RequestException as exc:
        raise HuggingFaceError("The AI provider request failed.", 502) from exc

    if not response.ok:
        provider_message = _error_message(response)
        logger.warning(
            "Hugging Face request failed with status %s: %s",
            response.status_code,
            provider_message,
        )
        if response.status_code in {402, 429}:
            message = "AI usage limits have been reached. Please try again later."
        elif response.status_code in {401, 403}:
            message = "The AI provider credentials or model access are invalid."
        else:
            message = "The AI provider could not complete the request."
        raise HuggingFaceError(message, response.status_code)

    try:
        result = response.json()
        choices = result["choices"]
        content = choices[0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected Hugging Face response shape.")
        raise HuggingFaceError(
            "The AI provider returned an unexpected response.", 502
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise HuggingFaceError("The AI provider returned an empty response.", 502)
    return content.strip()
