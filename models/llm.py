import os
import sys
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config.config as app_config 

def get_chatgroq_model(model_name: str = None):
    try:
        api_key = app_config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found.")

        if model_name is None:
            model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

        return ChatGroq(api_key=api_key, model=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")

def get_chatgoogle_model(model_name: str = None):
    try:
        api_key = app_config.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        if model_name is None:
            model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")

        return ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Google Gemini model: {str(e)}")

def get_llm_model(provider: str, model_name: str = None):
    if provider == "Groq":
        return get_chatgroq_model(model_name)
    elif provider == "Google Gemini":
        return get_chatgoogle_model(model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
