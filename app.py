import streamlit as st
from src.rag_engine import answer_legal_query, reset_caches
from src.vector_builder import build_or_get_vector_store
from src.config import LLM_MODEL, EMBEDDING_MODEL

st.set_page_config(
    page_title="Legal Assistant - RAG System",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal AI Assistant")

with st.sidebar:
    st.header("⚙️ Control Panel")
    st.info(f"Powered by {LLM_MODEL} & {EMBEDDING_MODEL} with ChromaDB.")
    st.markdown("---")
    if st.button("🔄 Rebuild Vector Database"):
        with st.spinner("Processing documents & rebuilding vector store..."):
            try:
                build_or_get_vector_store(force_rebuild=True)
                reset_caches()
                st.success("Vector store rebuilt successfully!")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your legal question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            try:
                result = answer_legal_query(prompt)
                st.markdown(result["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                