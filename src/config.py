import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned", "moroccan_law_paper.md")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

# الموديلات قابلة للتعديل من ملف .env دون التعديل على الكود
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# الموديل الأساسي والقائمة الاحتياطية
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("LLM_FALLBACK_MODELS", "gemini-flash-latest,gemini-3.1-flash-lite").split(",")
    if m.strip()
]

COLLECTION_NAME = "legal_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5
