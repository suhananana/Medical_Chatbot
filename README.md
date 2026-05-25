# 🏥 MediBot — Medical AI Assistant (Streamlit)

A clean, chat-style Streamlit app powered by your Medical RAG Chatbot notebook.

## Features
- Hybrid RAG (FAISS semantic + BM25 keyword) via Reciprocal Rank Fusion
- 6 tools: Hybrid RAG, FAISS RAG, BM25 RAG, Live Wikipedia, Symptom Checker, Drug Info
- Multi-LLM support: Gemini 2.5 Flash · GPT-4o-mini · Groq Llama
- Tool attribution badges on every response
- Multi-turn conversation memory

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key(s)
Copy `.env.example` to `.env` and fill in at least one key:
```bash
cp .env.example .env
# then edit .env
```

**Priority order:** Gemini → OpenAI → Groq (first available key is used)

| Provider | Get Key | Free Tier |
|----------|---------|-----------|
| Google Gemini | https://aistudio.google.com/app/apikey | 1500 req/day |
| OpenAI | https://platform.openai.com/api-keys | Pay-per-use |
| Groq | https://console.groq.com | 500K TPD (8b model) |

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> **Note:** First launch takes ~60 seconds to load the Wikipedia knowledge base (~60 articles) and build the FAISS index. Subsequent runs are instant thanks to Streamlit's `@st.cache_resource`.

## File Structure
```
medibot_app/
├── app.py          # Streamlit UI
├── backend.py      # RAG pipeline, tools, LangGraph agent
├── requirements.txt
├── .env.example    # API key template
└── README.md
```

## Disclaimer
MediBot is for **educational purposes only**. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
