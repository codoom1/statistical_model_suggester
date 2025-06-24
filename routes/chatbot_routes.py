"""
Chatbot Service Routes
This module provides routes for the AI chatbot feature,
allowing users to ask questions about the application content.
"""
from flask import Blueprint, request, jsonify
import os
import logging
# Import the new AI service and error class
from utils.ai_service import call_huggingface_api, is_ai_enabled, HuggingFaceError, get_huggingface_config
logger = logging.getLogger(__name__)
chatbot_bp = Blueprint('chatbot', __name__, url_prefix='/chatbot')
@chatbot_bp.route('/ask', methods=['POST'])
def ask_question():
    """Process a user question and return an AI-generated response"""
    data = request.json
    question = data.get('question', '')
    page_context = data.get('context', '')
    if not question:
        return jsonify({
            'success': False,
            'message': 'No question provided',
            'response': ''
        }), 400
    # Check if AI is enabled using the centralized function
    if not is_ai_enabled():
        logger.warning("AI features are disabled. Please set AI_ENHANCEMENT_ENABLED=true in your .env file.")
        return jsonify({
            'success': False,
            'message': 'AI features are currently disabled',
            'response': 'Sorry, the AI assistant is currently unavailable. Please try again later or contact the administrator.'
        }), 503
    # Get the configured model (used in prompt and potentially passed to API)
    _, configured_model = get_huggingface_config()
    try:
        # Format the prompt for the AI
        formatted_prompt = f"""<s>[INST] You are a helpful AI assistant for the Statistical Model Suggester application.
        You provide concise, accurate answers to questions about statistical models, data analysis, research methods, and the application's features.
        Current page context: {page_context}
        User question: {question}
        Provide a concise, helpful response. Limit your answer to 3-6 sentences unless a more detailed explanation is absolutely necessary.
        If you don't know something specific to this application, be honest about it. [/INST]</s>
        """
        # Call the centralized AI service
        response = call_huggingface_api(formatted_prompt, model=configured_model)
        if not response:
            logger.warning("Empty response from AI model")
            return jsonify({
                'success': False,
                'message': 'AI model returned empty response',
                'response': 'Sorry, I couldn\'t generate a response. Please try rephrasing your question.'
            }), 500
        return jsonify({
            'success': True,
            'message': 'Response generated successfully',
            'response': response
        })
    except HuggingFaceError as e:
        logger.error(f"HuggingFace API Error in chatbot: {e}")
        status_code = e.status_code or 500
        user_message = str(e)
        # Provide a more user-friendly message for credit limits
        if e.status_code == 402:
            user_message = 'I apologize, but our AI service has reached its usage limit for this month. The administrator has been notified. Basic functionality will continue to work, but AI-powered features may be limited until the next billing cycle.'
        return jsonify({'success': False, 'message': f'AI Error (Status: {status_code})', 'response': user_message}), status_code
    except ValueError as e: # Raised by call_huggingface_api if AI is disabled (should be caught earlier, but as safeguard)
        logger.error(f"ValueError in chatbot (AI likely disabled): {e}")
        return jsonify({'success': False, 'message': 'AI Service Error', 'response': 'The AI service is currently unavailable.'}), 503
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in chatbot AI call: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'response': 'Sorry, I encountered an error while processing your question. Please try again later.'
        }), 500
@chatbot_bp.route('/test-config', methods=['GET'])
def test_config():
    """Test endpoint to check AI configuration status"""
    try:
        # Check environment variables
        api_key, model = get_huggingface_config()
        enabled = is_ai_enabled()
        # Mask API key for security
        masked_api_key = "Not provided"
        if api_key:
            masked_api_key = api_key[:4] + '...' + api_key[-4:] if len(api_key) > 8 else '***'
        # Client status is implicitly tested by checking enablement and config
        client_status = "Enabled and Configured" if enabled and api_key else "Disabled or Not Configured"
        # Prepare response
        return jsonify({
            'success': True,
            'config': {
                'ai_enabled': enabled,
                'api_key': masked_api_key,
                'model': model,
                'client_status': client_status
            },
            'environment_variables': {
                'AI_ENHANCEMENT_ENABLED': os.environ.get('AI_ENHANCEMENT_ENABLED', 'false'),
                'HUGGINGFACE_API_KEY': 'Present' if api_key else 'Not set',
                'HUGGINGFACE_MODEL': model
            }
        })
    except Exception as e:
        logger.error(f"Error in test-config endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500