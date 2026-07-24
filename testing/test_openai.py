"""Manual smoke test for the configured OpenAI Responses API integration."""

import os
import sys

from dotenv import load_dotenv

from utils.ai_service import call_openai_api


def main() -> int:
    """Run one inexpensive request using local environment configuration."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not configured.")
        return 1

    os.environ.setdefault("AI_ENHANCEMENT_ENABLED", "true")
    try:
        response = call_openai_api(
            "In one sentence, name a model for a binary outcome."
        )
    except Exception as exc:
        print(f"OpenAI smoke test failed: {exc}")
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
