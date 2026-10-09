from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.config import GEMINI_API_KEY, TOP_K_RESULTS, LLM_MODEL, LLM_FALLBACK_MODELS
from src.vector_builder import build_or_get_vector_store

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""أنت مساعد قانوني خبير متخصص في تحليل النصوص والقوانين المغربية.
أجب عن سؤال المستخدم بناءً **فقط** على السياق النصي المرفق أدناه.
إذا كانت الإجابة غير موجودة في النص المرفق، أجب بوضوح: "المعلومة المطلوبة غير متوفرة في الوثيقة المتاحة".
احرص على الاستشهاد بالربط بين المواد والقوانين المذكورة إن وجدت.

السياق المتاح:
{context}

سؤال المستخدم:
{question}

الإجابة القانونية الدقيقة:""",
)

@lru_cache(maxsize=1)
def get_retriever():
    store = build_or_get_vector_store()
    return store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

@lru_cache(maxsize=1)
def get_llm():
    primary = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)
    fallbacks = [
        ChatGoogleGenerativeAI(model=m, google_api_key=GEMINI_API_KEY)
        for m in LLM_FALLBACK_MODELS
    ]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary

def reset_caches():
    """Call after rebuilding the vector store."""
    get_retriever.cache_clear()
    get_llm.cache_clear()

def _as_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)

def answer_legal_query(query: str):
    docs = get_retriever().invoke(query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    response = (PROMPT | get_llm()).invoke({"context": context, "question": query})
    return {"answer": _as_text(response.content), "sources": docs}

if __name__ == "__main__":
    print(answer_legal_query("What was the Context Precision for Architecture B?")["answer"])
    