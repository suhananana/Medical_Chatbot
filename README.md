# 🏥 MediBot — Medical RAG Chatbot

A clean, seamless ChatGPT-style medical chatbot.  
No setup screens. Just open and chat.

---

## 🚀 Run locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add your HF token
cp .env.example .env
# edit .env → paste your token

# 3. Launch
streamlit run app.py
```

Open **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud (free)

1. Push folder to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → pick repo → `app.py`
3. Under **Settings → Secrets**, add:
   ```
   HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
   ```
4. Deploy

---

## 🔑 HF Token

Free token → https://huggingface.co/settings/tokens (Read access is enough)

---

## ⚠️ Disclaimer
Educational purposes only. Always consult a licensed healthcare professional.
