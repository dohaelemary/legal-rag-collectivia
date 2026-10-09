# ⚖️ CollectivIA: Two-Pipeline Multilingual Legal RAG System

CollectivIA is an advanced **Retrieval-Augmented Generation (RAG)** decision-support web application tailored for Moroccan territorial governance law. The system enables users to query complex legislative corpora in **French, Modern Standard Arabic (MSA), and Moroccan Darija**, providing accurate, grounded answers accompanied by explicit legal citations.

---

## 🌟 Key Features

- **Multilingual Query Support:** Handles queries seamlessly across French, Arabic, and Moroccan Darija via query normalization pipelines.
- **Robust Legal Retrieval:** Implements vector search to locate precise article-level legal references before generating standard responses.
- **Strict Citation Enforcement:** Ensures the language model generates responses anchored strictly in the retrieved text to avoid legal hallucinations.
- **Interactive UI:** Built using **Streamlit** to offer an intuitive user experience for legal inquiry and analysis.

---

## 🏗️ Technical Architecture

- **LLM Engine:** Google Gemini (`gemini-3.5-flash` / `gemini-2.5-flash` with dynamic fallback mechanisms)
- **Embeddings:** `gemini-embedding-001`
- **Vector Database:** ChromaDB
- **Frameworks & Libraries:** LangChain, Streamlit, PyMuPDF, Pandas
- **Deployment:** Streamlit Community Cloud

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/dohaelemary/legal-rag-collectivia.git](https://github.com/dohaelemary/legal-rag-collectivia.git)
cd legal-rag-collectivia
```

### 2. Set Up Virtual Environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

```
### 3. Install Dependencies

pip install -r requirements.txt

```
### 4. Configure Environment Variables
Create a .env file in the root directory and add your Google Gemini API key:

GEMINI_API_KEY=your_actual_gemini_api_key


### 5. Run the Application

streamlit run app.py
```env
