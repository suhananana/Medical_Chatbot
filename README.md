# 🏥 MediBot — Medical RAG Chatbot

**LangChain · LangGraph · Gemini 1.5 Flash · FAISS · Wikipedia · Streamlit**

---

## 🗂️ Project Structure

```
medibot_streamlit/
├── app.py                  ← Streamlit UI
├── agent.py                ← LangGraph agent + tools + vector store
├── requirements.txt
├── .gitignore
└── .streamlit/
    ├── config.toml         ← Theme & server settings
    └── secrets.toml        ← (optional) pre-fill API key
```

---

## ⚙️ Local Setup

### 1. Clone / copy this folder

```bash
cd medibot_streamlit
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Gemini API key (free)

👉 https://aistudio.google.com/app/apikey

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🌐 Deploy on Streamlit Cloud (free)

1. Push this folder to a **GitHub repository**.
2. Go to https://share.streamlit.io → **New app**.
3. Point it at `app.py` in your repo.
4. In **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
5. Click **Deploy** — done!

---

## 🤖 Architecture

```
User Input
    │
    ▼
┌─────────────┐
│  LangGraph  │  ← ReAct loop (max 15 iterations)
│   Agent     │
└──────┬──────┘
       │  LLM decides which tool(s) to call
       ▼
┌──────────────────────────────────────────┐
│              Tool Node                   │
│  1. medical_rag_retriever  (FAISS)       │
│  2. wikipedia              (live search) │
│  3. symptom_checker        (rule-based)  │
└──────────────────┬───────────────────────┘
                   │ results injected back
                   ▼
              Final Answer
```

### Components

| Component     | Technology                              |
|---------------|-----------------------------------------|
| LLM           | Google Gemini 1.5 Flash                 |
| Embeddings    | `all-MiniLM-L6-v2` (HuggingFace)       |
| Vector Store  | FAISS (in-memory)                       |
| Knowledge Base| 12 medical Wikipedia topics (~24 pages) |
| Orchestration | LangGraph StateGraph                    |
| UI            | Streamlit                               |

---

## 🔧 Tools

| Tool                   | Trigger                                  |
|------------------------|------------------------------------------|
| `medical_rag_retriever`| Any medical question (tried first)       |
| `wikipedia`            | Extra detail / up-to-date info           |
| `symptom_checker`      | User describes symptoms                  |

---

## 💬 Example Questions

- "What is diabetes mellitus and how is it treated?"
- "I have fever, cough, and fatigue — what could it be?"
- "How does the immune system fight cancer?"
- "What are the side effects of antibiotics?"
- "Explain hypertension and its risk factors."

---

> ⚠️ **Disclaimer:** For educational purposes only.  
> Always consult a qualified healthcare professional.
