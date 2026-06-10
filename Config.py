import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Smart RAG Chatbot"
APP_ICON = "🤖"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
