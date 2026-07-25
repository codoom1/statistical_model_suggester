"""Server-side OpenAI Responses API integration."""

import logging
import os
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful statistical methods assistant. Give concise, accurate "
    "answers, state important assumptions, and do not invent application "
    "features or research findings."
)


class OpenAIServiceError(RuntimeError):
    """Raised when OpenAI cannot complete an application request."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def is_ai_enabled() -> bool:
    """Return whether AI-backed features are enabled."""
    return os.environ.get("AI_ENHANCEMENT_ENABLED", "false").lower() == "true"


def get_openai_config() -> tuple[Optional[str], str]:
    """Return the configured OpenAI API key and model without exposing the key."""
    return (
        os.environ.get("OPENAI_API_KEY", "").strip() or None,
        os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    )


def _timeout_seconds() -> float:
    raw_timeout = os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "45")
    try:
        return min(max(float(raw_timeout), 5.0), 55.0)
    except ValueError:
        return 45.0


def _max_output_tokens() -> int:
    raw_limit = os.environ.get("AI_MAX_OUTPUT_TOKENS", "400")
    try:
        return min(max(int(raw_limit), 100), 1_500)
    except ValueError:
        return 400


def _reasoning_effort() -> str:
    configured = os.environ.get("OPENAI_REASONING_EFFORT", "low").strip().lower()
    return configured if configured in {"none", "low", "medium", "high"} else "low"


def call_openai_api(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    safety_identifier: Optional[str] = None,
    response_schema: Optional[dict[str, Any]] = None,
    schema_name: str = "structured_response",
) -> str:
    """Generate text through OpenAI's Responses API."""
    if not is_ai_enabled():
        raise OpenAIServiceError("AI features are currently disabled.", 503)

    api_key, configured_model = get_openai_config()
    if not api_key:
        raise OpenAIServiceError(
            "OPENAI_API_KEY is required when AI features are enabled.",
            503,
        )

    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise ValueError("An AI prompt is required.")

    target_model = (model or configured_model).strip()
    fallback_model = (
        os.environ.get("OPENAI_FALLBACK_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL
    )
    request_options = {
        "model": target_model,
        "instructions": system_prompt or DEFAULT_SYSTEM_PROMPT,
        "input": cleaned_prompt,
        "max_output_tokens": _max_output_tokens(),
        "reasoning": {"effort": _reasoning_effort()},
        "store": False,
    }
    if safety_identifier:
        request_options["safety_identifier"] = safety_identifier
    if response_schema:
        request_options["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": response_schema,
                "strict": True,
            }
        }

    client = OpenAI(
        api_key=api_key,
        timeout=_timeout_seconds(),
        max_retries=1,
    )
    candidate_models = [target_model]
    if fallback_model != target_model:
        candidate_models.append(fallback_model)

    response = None
    for candidate_model in candidate_models:
        request_options["model"] = candidate_model
        try:
            response = client.responses.create(**request_options)
            break
        except NotFoundError as exc:
            if candidate_model != candidate_models[-1]:
                logger.warning(
                    "Configured OpenAI model '%s' is unavailable; trying '%s'.",
                    candidate_model,
                    fallback_model,
                )
                continue
            raise OpenAIServiceError(
                "The configured OpenAI model is unavailable for this project. "
                "Set OPENAI_MODEL to an enabled model such as gpt-5-mini.",
                503,
            ) from exc
        except APITimeoutError as exc:
            raise OpenAIServiceError("The AI provider timed out.", 504) from exc
        except APIConnectionError as exc:
            raise OpenAIServiceError(
                "Could not connect to the AI provider.", 503
            ) from exc
        except RateLimitError as exc:
            raise OpenAIServiceError(
                "AI usage limits have been reached. Please try again later.",
                429,
            ) from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise OpenAIServiceError(
                "The OpenAI credentials or model access are invalid.",
                getattr(exc, "status_code", 401),
            ) from exc
        except APIStatusError as exc:
            logger.warning("OpenAI request failed with status %s.", exc.status_code)
            raise OpenAIServiceError(
                "The AI provider could not complete the request.",
                exc.status_code,
            ) from exc

    if response is None:
        raise OpenAIServiceError("The AI provider returned no response.", 502)
    content = response.output_text
    if not isinstance(content, str) or not content.strip():
        raise OpenAIServiceError("The AI provider returned an empty response.", 502)
    return content.strip()
