import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical AI Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #f0f4f8; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Top header bar */
.medibot-header {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
    border-radius: 16px;
    padding: 24px 28px 20px;
    margin-bottom: 20px;
    color: white;
    display: flex;
    align-items: center;
    gap: 16px;
}
.medibot-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
.medibot-header p  { margin: 4px 0 0; font-size: 0.85rem; opacity: 0.85; }

/* Chat container */
.chat-wrap {
    background: white;
    border-radius: 16px;
    padding: 20px;
    min-height: 420px;
    max-height: 520px;
    overflow-y: auto;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 12px;
}

/* Message bubbles */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 14px;
}
.msg-user .bubble {
    background: #1a73e8;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    max-width: 75%;
    font-size: 0.93rem;
    line-height: 1.5;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 14px;
    gap: 10px;
    align-items: flex-start;
}
.bot-avatar {
    background: #e8f0fe;
    border-radius: 50%;
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.msg-bot .bubble {
    background: #f8f9fa;
    color: #1a1a2e;
    border-radius: 4px 18px 18px 18px;
    padding: 10px 16px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.6;
    border: 1px solid #e8eaf0;
}

/* Tool badge strip */
.tool-strip {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.tool-badge {
    background: #e8f0fe;
    color: #1a73e8;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid #c5d9fb;
}

/* Disclaimer */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #5d4037;
    margin-bottom: 14px;
}

/* Loading spinner text */
.thinking {
    color: #1a73e8;
    font-size: 0.85rem;
    font-style: italic;
    padding: 8px 0;
}

/* Input area */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #c5d9fb !important;
    padding: 10px 16px !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.15) !important;
}
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 46px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load backend (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading Medical Knowledge Base — takes ~60s on first run…")
def load_backend():
    from backend import build_rag_pipeline, build_agent, SYSTEM_PROMPT
    pipeline = build_rag_pipeline()
    agent    = build_agent(pipeline)
    return agent, pipeline

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, tools}
if "history" not in st.session_state:
    st.session_state.history  = []          # LangChain message objects

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="medibot-header">
  <div style="font-size:2.4rem">🏥</div>
  <div>
    <h1>MediBot — Medical AI Assistant</h1>
    <p>Hybrid RAG (FAISS + BM25) &nbsp;·&nbsp; 6 Tools &nbsp;·&nbsp; Wikipedia Knowledge Base</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Educational use only.</strong>
  MediBot provides general health information. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

# ── Load backend ──────────────────────────────────────────────────────────────
try:
    agent, pipeline = load_backend()
except Exception as e:
    st.error(f"❌ Failed to load backend: {e}")
    st.info("Make sure your `.env` file has at least one of: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY`.")
    st.stop()

# ── Chat display ──────────────────────────────────────────────────────────────
TOOL_LABELS = {
    "hybrid_rag_retriever": "Hybrid RAG",
    "faiss_rag_retriever":  "FAISS Semantic",
    "bm25_rag_retriever":   "BM25 Keyword",
    "wikipedia_live_tool":  "Wikipedia Live",
    "symptom_checker":      "Symptom Checker",
    "drug_information":     "Drug Info DB",
}

chat_html = '<div class="chat-wrap">'

if not st.session_state.messages:
    chat_html += """
    <div style="text-align:center;color:#9aa0b0;padding:60px 0 40px;">
      <div style="font-size:3rem">💬</div>
      <div style="margin-top:10px;font-size:0.95rem;">Ask me anything about symptoms, diseases, medications, or anatomy.</div>
      <div style="margin-top:20px;font-size:0.83rem;color:#b0b8c8;">
        Try: &nbsp;<em>What is diabetes?</em> &nbsp;·&nbsp;
        <em>I have fever and cough</em> &nbsp;·&nbsp;
        <em>Side effects of ibuprofen?</em>
      </div>
    </div>"""
else:
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_html += f"""
            <div class="msg-user">
              <div class="bubble">{m["content"]}</div>
            </div>"""
        else:
            tools_html = ""
            if m.get("tools"):
                badges = "".join(
                    f'<span class="tool-badge">✓ {TOOL_LABELS.get(t, t)}</span>'
                    for t in m["tools"]
                )
                tools_html = f'<div class="tool-strip">{badges}</div>'

            # Render newlines as <br> for display
            content = m["content"].replace("\n", "<br>")
            chat_html += f"""
            <div class="msg-bot">
              <div class="bot-avatar">🏥</div>
              <div>
                <div class="bubble">{content}</div>
                {tools_html}
              </div>
            </div>"""

chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([7, 1.2, 1.2])

with col1:
    user_input = st.text_input(
        "question",
        placeholder="Type your medical question…",
        label_visibility="collapsed",
        key="user_input",
    )
with col2:
    send = st.button("Send 💬", use_container_width=True, type="primary")
with col3:
    clear = st.button("Clear 🗑️", use_container_width=True)

# ── Clear ─────────────────────────────────────────────────────────────────────
if clear:
    st.session_state.messages = []
    st.session_state.history  = []
    st.rerun()

# ── Send ──────────────────────────────────────────────────────────────────────
if (send or user_input) and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🤔 MediBot is thinking…"):
        try:
            from backend import ask_agent
            reply, tools_used = ask_agent(agent, st.session_state.history, question)
        except Exception as e:
            reply = f"Sorry, an error occurred: {e}"
            tools_used = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "tools": tools_used,
    })
    st.rerun()
