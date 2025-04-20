#!/usr/bin/env python3
"""
Test script for Hugging Face API connection
Run this script directly to test if your API key is valid and connection works
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_huggingface_api():
    # Get API key from environment
    api_key = os.environ.get('HUGGINGFACE_API_KEY')
    if not api_key:
        print("ERROR: No HUGGINGFACE_API_KEY found in environment variables")
        return False

    # Get model from environment or use default
    model = os.environ.get('HUGGINGFACE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')
    
    # Set up headers and API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    
    # Simple test prompt
    payload = {
        "inputs": "What is 2 + 2?",
        "parameters": {
            "max_length": 50,
            "temperature": 0.7,
            "top_p": 0.95
        }
    }
    
    print(f"Testing Hugging Face API with model: {model}")
    print(f"API key (first 4 chars): {api_key[:4]}...")
    print(f"API URL: {api_url}")
    
    try:
        # Make the API request
        print("Sending request to Hugging Face API...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        # Check status code
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"ERROR: API returned non-200 status code: {response.status_code}")
            print(f"Response body: {response.text}")
            return False
        
        # Parse response
        try:
            result = response.json()
            print("Response JSON structure:")
            print(json.dumps(result, indent=2))
            
            # Extract generated text
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                generated_text = result[0]["generated_text"].strip()
                print(f"Generated text: {generated_text}")
                return True
            elif isinstance(result[0], str):
                generated_text = result[0].strip()
                print(f"Generated text: {generated_text}")
                return True
            
            print("ERROR: Unexpected response format")
            return False
            
        except ValueError as e:
            print(f"ERROR: Could not parse JSON response: {e}")
            print(f"Response content: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        return False
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Connection error")
        return False
        
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

if __name__ == "__main__":
    print("=== Hugging Face API Test ===")
    print("Environment variables:")
    print(f"AI_ENHANCEMENT_ENABLED: {os.environ.get('AI_ENHANCEMENT_ENABLED', 'Not set')}")
    print(f"HUGGINGFACE_API_KEY: {'Set' if os.environ.get('HUGGINGFACE_API_KEY') else 'Not set'}")
    print(f"HUGGINGFACE_MODEL: {os.environ.get('HUGGINGFACE_MODEL', 'Not set (will use default)')}")
    print("")
    
    success = test_huggingface_api()
    
    if success:
        print("\n✅ TEST PASSED: Hugging Face API is working correctly!")
    else:
        print("\n❌ TEST FAILED: There was an issue with the Hugging Face API connection")