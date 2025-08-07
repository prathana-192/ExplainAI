import os
import sys

# Add the parent directory to the system path to allow imports from 'models' and 'config'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm import get_llm_model
import config.config as app_config

def test_llm_model(provider: str, model_name: str = None):
    """
    Attempts to initialize an LLM model and make a simple call to test the API key.
    """
    print(f"\n--- Testing {provider} Model ---")
    try:
        # Check if API key is present in config
        if provider == "Groq" and not app_config.GROQ_API_KEY:
            print(f"Skipping {provider}: GROQ_API_KEY not found in .env file.")
            return
        elif provider == "Google Gemini" and not app_config.GOOGLE_API_KEY:
            print(f"Skipping {provider}: GOOGLE_API_KEY not found in .env file.")
            return
        # Add OpenAI check if you implement it
        # elif provider == "OpenAI" and not app_config.OPENAI_API_KEY:
        #     print(f"Skipping {provider}: OPENAI_API_KEY not found in .env file.")
        #     return

        chat_model = get_llm_model(provider, model_name)
        print(f"Successfully initialized {provider} model: {model_name if model_name else 'default'}")

        # Make a simple test call
        test_message = "Hello, how are you today?"
        print(f"Sending test message: '{test_message}'")
        response = chat_model.invoke(test_message)
        print(f"Received response from {provider}: {response.content[:100]}...") # Print first 100 chars
        print(f"Test for {provider} successful!")

    except Exception as e:
        print(f"Test for {provider} failed: {e}")

if __name__ == "__main__":
    # Ensure .env file is loaded
    # This is handled by config.config, but explicitly calling load_dotenv here for clarity in test script
    from dotenv import load_dotenv
    load_dotenv()

    # Test Groq
    test_llm_model("Groq", "llama-3.1-8b-instant") # You can specify a model or let it use default
    
    # Test Google Gemini
    test_llm_model("Google Gemini", "gemini-1.5-flash") # You can specify a model or let it use default

    print("\n--- API Key Tests Complete ---")
