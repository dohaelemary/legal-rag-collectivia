import time
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import (
    DATA_PATH, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    GEMINI_API_KEY, EMBEDDING_MODEL, COLLECTION_NAME,
)

def get_embeddings():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)

def _open_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=get_embeddings(),
    )

def _load_chunks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(markdown_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(sections)

def build_or_get_vector_store(force_rebuild: bool = False):
    store = _open_store()
    has_data = len(store.get(limit=1)["ids"]) > 0
    
    if has_data and not force_rebuild:
        print("Existing vector store found. Loading...")
        return store

    if has_data:
        store.delete_collection()
        store = _open_store()

    chunks = _load_chunks()
    print(f"Split into {len(chunks)} chunks. Embedding...")
    
    # إرسال 20 عنصر فقط في المرة بدلاً من 50 مع تأخير ثانيتين لمنع الـ Rate Limit
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        store.add_documents(chunks[i:i + batch_size])
        time.sleep(2)  # انتظر ثانيتين بين كل batch والآخر
        
    print("Vector store created successfully.")
    return store

if __name__ == "__main__":
    build_or_get_vector_store(force_rebuild=True)
    