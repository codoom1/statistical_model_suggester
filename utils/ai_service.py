import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

class HuggingFaceError(Exception):
    """Custom exception for Hugging Face API errors."""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

def is_ai_enabled() -> bool:
    """Check if AI features are enabled via environment variable."""
    ai_enabled_raw = os.environ.get('AI_ENHANCEMENT_ENABLED', 'false')
    enabled = ai_enabled_raw.lower() == 'true'
    logger.debug(f"AI_ENHANCEMENT_ENABLED raw value: '{ai_enabled_raw}', Parsed: {enabled}")
    return enabled

def get_huggingface_config() -> tuple[str | None, str]:
    """Get Hugging Face API key and model from environment variables."""
    api_key = os.environ.get('HUGGINGFACE_API_KEY')
    model = os.environ.get('HUGGINGFACE_MODEL', DEFAULT_MODEL)
    if not api_key:
        logger.warning("No HUGGINGFACE_API_KEY environment variable set. Using public model access.")
    logger.debug(f"Hugging Face Config - API Key Present: {'Yes' if api_key else 'No'}, Model: {model}")
    return api_key, model

def call_huggingface_api(prompt: str, model: str | None = None) -> str:
    """
    Call the Hugging Face Inference API.

    Args:
        prompt (str): The prompt to send to the model.
        model (str, optional): The model ID to use. Defaults to HUGGINGFACE_MODEL env var or DEFAULT_MODEL.

    Returns:
        str: The generated text response from the model.

    Raises:
        HuggingFaceError: If there's an issue with the API call (connection, timeout, API error, bad response).
        ValueError: If AI features are disabled.
    """
    if not is_ai_enabled():
        logger.warning("Attempted to call Hugging Face API while AI features are disabled.")
        raise ValueError("AI features are currently disabled.")

    api_key, default_model = get_huggingface_config()
    target_model = model if model else default_model

    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    api_url = f"https://api-inference.huggingface.co/models/{target_model}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,  # Allow for longer responses
            "temperature": 0.9,    # Control randomness
            "top_p": 0.95,       # Control nucleus sampling
            "return_full_text": False # Only return the generated text
        }
    }

    logger.info(f"Making request to Hugging Face API: {api_url} with model: {target_model}")
    logger.debug(f"Payload keys: {list(payload.keys())}")
    logger.debug(f"Headers set: {list(headers.keys())}")

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=45) # Increased timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # Try to parse JSON response
        try:
            result = response.json()
            logger.info(f"Got successful response ({response.status_code}) from Hugging Face API: {type(result)}")

            # Extract the generated text (handle different response formats)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                elif isinstance(result[0], str): # Some models might return string directly in list
                     return result[0].strip()
            elif isinstance(result, dict) and "generated_text" in result: # Handle single dict response
                 return result["generated_text"].strip()

            # If we get here, we have an unexpected format
            logger.warning(f"Unexpected response format from Hugging Face API: {result}")
            raise HuggingFaceError("Unexpected response format received from AI service.", response.status_code)

        except ValueError as e: # JSONDecodeError inherits from ValueError
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response status code: {response.status_code}")
            logger.error(f"Response content: {response.text[:500]}...") # Log beginning of content
            raise HuggingFaceError(f"Invalid response received from AI service (Status {response.status_code}).", response.status_code) from e

    except requests.exceptions.Timeout as e:
        logger.error("Hugging Face API request timed out")
        raise HuggingFaceError("The request to the AI service timed out.") from e
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error when calling Hugging Face API: {e}")
        raise HuggingFaceError("Could not connect to the AI service.") from e
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_text = e.response.text[:500] # Limit error text length
        logger.error(f"Hugging Face API returned error status {status_code}: {error_text}")
        
        # Specific check for credit limit (common issue)
        if status_code == 402 or 'exceeded your monthly' in error_text.lower() or 'credits' in error_text.lower():
             raise HuggingFaceError("Hugging Face API credits exceeded. Please check your account.", status_code=402) from e
        
        # General HTTP error
        raise HuggingFaceError(f"AI service request failed with status {status_code}.", status_code=status_code) from e
    except Exception as e:
        logger.error(f"Unexpected error calling Hugging Face API: {e}", exc_info=True)
        raise HuggingFaceError(f"An unexpected error occurred while contacting the AI service: {str(e)}") from e 