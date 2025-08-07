import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY=os.getenv("SERPAPI_API_KEY")

def is_web_search_enabled():
    return bool(SERPAPI_API_KEY)
