import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")

PDF_PATH = "data/agentic_ai_ebook.pdf"

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 3072

LLM_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
MIN_RELEVANCE = 0.30