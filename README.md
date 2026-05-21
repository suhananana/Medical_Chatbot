# 🏥 MediBot — Medical RAG Chatbot (Streamlit)

A fully interactive Medical AI Chatbot powered by **LangChain + LangGraph + FAISS + TinyLlama**.

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy**

> **Tip:** Add your `HUGGINGFACEHUB_API_TOKEN` as a Secret in the Streamlit Cloud dashboard so users don't need to enter it manually.

To use a pre-set token from secrets, add this near the top of `app.py`:
```python
import streamlit as st
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
```

---

## 🔑 Hugging Face Token

Get a **free** token at https://huggingface.co/settings/tokens  
You need **Read** access. No payment required for TinyLlama.

---

## 🧱 Architecture

```
User Query
    │
    ▼
[Streamlit UI]
    │
    ▼
[LangGraph ReAct Agent]
    │
    ├──► Tool 1: medical_rag_retriever  (FAISS local KB from Wikipedia)
    ├──► Tool 2: wikipedia_live_tool    (live Wikipedia fallback)
    └──► Tool 3: symptom_checker        (rule-based symptom mapper)
    │
    ▼
[TinyLlama-1.1B-Chat] (local inference via HuggingFace transformers)
    │
    ▼
[Chat Response → Streamlit UI]
```

---

## 📦 Stack

| Component     | Technology                               |
|---------------|------------------------------------------|
| 🖥️ UI         | Streamlit                                |
| 🤖 LLM        | TinyLlama/TinyLlama-1.1B-Chat-v1.0       |
| 🧠 Embeddings | sentence-transformers/all-MiniLM-L6-v2   |
| 📚 Vector DB  | FAISS (in-memory)                        |
| 🌐 Knowledge  | Wikipedia (50 medical topics)            |
| 🔗 Orchestration | LangGraph ReAct Agent                 |

---

## ⚠️ Disclaimer

MediBot is for **educational purposes only**.  
Always consult a licensed healthcare professional for medical advice.
