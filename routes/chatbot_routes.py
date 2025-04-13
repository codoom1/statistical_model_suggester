"""
Chatbot Service Routes

This module provides routes for the AI chatbot feature,
allowing users to ask questions about the application content.
"""

from flask import Blueprint, request, jsonify
import os
import logging
from utils.questionnaire_generator import get_huggingface_client

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
    
    # Check if AI is enabled with detailed logging
    ai_enabled_raw = os.environ.get('AI_ENHANCEMENT_ENABLED', 'false')
    ai_enabled = ai_enabled_raw.lower() == 'true'
    logger.info(f"AI_ENHANCEMENT_ENABLED raw value: '{ai_enabled_raw}'")
    logger.info(f"AI enabled status: {ai_enabled}")
    
    if not ai_enabled:
        logger.warning("AI features are disabled. Please set AI_ENHANCEMENT_ENABLED=true in your .env file.")
        return jsonify({
            'success': False,
            'message': 'AI features are currently disabled',
            'response': 'Sorry, the AI assistant is currently unavailable. Please try again later or contact the administrator.'
        }), 503
    
    # Get the Hugging Face client with detailed logging
    logger.info(f"Attempting to initialize Hugging Face client")
    logger.info(f"HUGGINGFACE_API_KEY exists: {'Yes' if os.environ.get('HUGGINGFACE_API_KEY') else 'No'}")
    logger.info(f"HUGGINGFACE_MODEL: {os.environ.get('HUGGINGFACE_MODEL', 'Not set')}")
    
    client = get_huggingface_client()
    if not client:
        logger.error("Failed to initialize Hugging Face client")
        return jsonify({
            'success': False,
            'message': 'Failed to initialize AI client',
            'response': 'Sorry, I encountered a technical issue. Please try again later.'
        }), 500
    
    try:
        # Format the prompt for the AI
        formatted_prompt = f"""<s>[INST] You are a helpful AI assistant for the Statistical Model Suggester application. 
        You provide concise, accurate answers to questions about statistical models, data analysis, research methods, and the application's features.
        
        Current page context: {page_context}
        
        User question: {question}
        
        Provide a concise, helpful response. Limit your answer to 3-4 sentences unless a more detailed explanation is absolutely necessary.
        If you don't know something specific to this application, be honest about it. [/INST]</s>
        """
        
        # Call the Hugging Face API
        response = client(formatted_prompt, model=os.environ.get('HUGGINGFACE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2'))
        
        # Check if the response contains a recognizable error about credits
        if response and ('exceeded your monthly' in response or 'Error: Received status code 402' in response):
            logger.warning("Hugging Face API credits exceeded")
            return jsonify({
                'success': False,
                'message': 'API credits exceeded',
                'response': 'I apologize, but our AI service has reached its usage limit for this month. The administrator has been notified. Basic functionality will continue to work, but AI-powered features may be limited until the next billing cycle.'
            }), 402
        
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
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
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
        api_key = os.environ.get('HUGGINGFACE_API_KEY', 'Not set')
        model = os.environ.get('HUGGINGFACE_MODEL', 'Not set')
        ai_enabled = os.environ.get('AI_ENHANCEMENT_ENABLED', 'false').lower() == 'true'
        
        # Mask API key for security
        masked_api_key = "Not provided"
        if api_key and api_key != 'Not set':
            masked_api_key = api_key[:4] + '...' + api_key[-4:] if len(api_key) > 8 else '***' 
        
        # Test client initialization
        client = get_huggingface_client()
        client_status = "Initialized successfully" if client else "Failed to initialize"
        
        # Prepare response
        return jsonify({
            'success': True,
            'config': {
                'ai_enabled': ai_enabled,
                'api_key': masked_api_key,
                'model': model,
                'client_status': client_status
            },
            'environment_variables': {
                'AI_ENHANCEMENT_ENABLED': os.environ.get('AI_ENHANCEMENT_ENABLED', 'Not set'),
                'HUGGINGFACE_API_KEY': 'Present' if api_key and api_key != 'Not set' else 'Not set',
                'HUGGINGFACE_MODEL': model
            }
        })
    except Exception as e:
        logger.error(f"Error in test-config endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 