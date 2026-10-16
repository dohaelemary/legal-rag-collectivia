import json
import os
import time
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    VECTOR_STORE_DIR, 
    GEMINI_API_KEY, EMBEDDING_MODEL, COLLECTION_NAME,
)

def get_embeddings():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)

def build_or_get_vector_store():
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=get_embeddings(),
    )
    
    documents = []
    
    # 1. المواد
    articles_path = os.path.join("data", "cleaned", "egyptian_civil_code_articles2.json")
    if os.path.exists(articles_path):
        with open(articles_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                art_num = item.get("article", "")
                art_text = item.get("context", "")
                documents.append(Document(
                    page_content=f"المادة {art_num}: {art_text}",
                    metadata={"type": "article", "article_number": str(art_num)}
                ))

    # 2. الـ QA
    qa_path = os.path.join("data", "cleaned", "egyptian_civil_qa_test.json")
    if os.path.exists(qa_path):
        with open(qa_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                q = item.get("question", item.get("q", ""))
                a = item.get("answer", item.get("a", ""))
                if q or a:
                    documents.append(Document(
                        page_content=f"سؤال قانوني: {q}\nالإجابة القانونية: {a}",
                        metadata={"type": "qa"}
                    ))

    print(f"Total documents expected: {len(documents)}")
    
    existing_data = store.get()
    existing_count = len(existing_data["ids"])
    print(f"Already embedded and stored: {existing_count} documents.")

    if existing_count >= len(documents):
        print("Vector store is already fully complete!")
        return store

    remaining_docs = documents[existing_count:]
    print(f"Resuming embedding for remaining {len(remaining_docs)} documents...")

    batch_size = 5
    for i in range(0, len(remaining_docs), batch_size):
        batch = remaining_docs[i:i + batch_size]
        store.add_documents(batch)
        current_batch_num = (existing_count + i) // batch_size + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"Uploaded batch {current_batch_num} / {total_batches}")
        time.sleep(6)

    print("Vector store updated and completed successfully!")
    return store

if __name__ == "__main__":
    build_or_get_vector_store()