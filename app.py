"""
app.py — MediBot Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="MediBot 🏥",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent import ask_medibot, build_agent, build_vectorstore

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Chat bubbles ── */
.user-bubble {
    background: #e3f2fd;
    border-left: 4px solid #1e88e5;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    font-family: system-ui, sans-serif;
}
.bot-bubble {
    background: #f1f8e9;
    border-left: 4px solid #43a047;
    padding: 14px 16px;
    border-radius: 8px;
    margin: 8px 0;
    font-family: system-ui, sans-serif;
    line-height: 1.75;
}
/* ── Tool badge strip ── */
.tool-strip {
    background: #fff8e1;
    border-left: 4px solid #fdd835;
    padding: 8px 14px;
    border-radius: 6px;
    margin: 4px 0 16px 0;
    font-size: 13px;
    font-family: system-ui, sans-serif;
}
.badge {
    display: inline-block;
    background: #e8f5e9;
    border: 1px solid #43a047;
    border-radius: 14px;
    padding: 3px 12px;
    margin: 2px 4px 2px 0;
    font-size: 13px;
}
/* ── Disclaimer banner ── */
.disclaimer {
    background: #fff3e0;
    border: 1px solid #ff9800;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    color: #e65100;
    font-family: system-ui, sans-serif;
}
/* ── Sidebar metric cards ── */
.metric-card {
    background: #f5f5f5;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    font-family: system-ui, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── Tool display labels ────────────────────────────────────────────────────
TOOL_LABELS = {
    "medical_rag_retriever": "📚 RAG Retriever (FAISS)",
    "wikipedia":             "🌐 Live Wikipedia",
    "symptom_checker":       "🩺 Symptom Checker",
}

DEMO_QUESTIONS = [
    "What is diabetes mellitus and how is it treated?",
    "I have fever, cough, and fatigue. What could it be?",
    "How does the immune system fight cancer?",
    "What are the side effects of antibiotics?",
    "Explain hypertension and its risk factors.",
    "I have chest pain, shortness of breath, and dizziness.",
]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "agent"          not in st.session_state: st.session_state.agent          = None
if "vectorstore"    not in st.session_state: st.session_state.vectorstore    = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []   # BaseMessage list
if "display_msgs"   not in st.session_state: st.session_state.display_msgs   = []   # dicts for UI
if "initialized"    not in st.session_state: st.session_state.initialized    = False
if "total_queries"  not in st.session_state: st.session_state.total_queries  = 0
if "tools_counter"  not in st.session_state: st.session_state.tools_counter  = {}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=70)
    st.title("MediBot 🏥")
    st.caption("Medical AI · LangChain · LangGraph · Gemini")

    st.divider()

    # ── API Key ──────────────────────────────────────────────────────────────
    st.subheader("🔑 Configuration")
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    # ── Init button ──────────────────────────────────────────────────────────
    init_btn = st.button(
        "🚀 Initialize MediBot",
        disabled=st.session_state.initialized,
        use_container_width=True,
        type="primary",
    )

    if init_btn:
        if not gemini_key.strip():
            st.error("Please enter your Gemini API key first.")
        else:
            # Build vector store with progress bar
            st.info("⏳ Loading Wikipedia articles...")
            prog_bar  = st.progress(0.0, text="Starting…")
            prog_text = st.empty()

            def on_progress(topic, loaded, done, total):
                pct = done / total
                prog_bar.progress(pct, text=f"Loading: {topic}")
                prog_text.caption(f"Articles loaded so far: {loaded}")

            try:
                vs = build_vectorstore(progress_callback=on_progress)
                st.session_state.vectorstore = vs
                prog_bar.progress(1.0, text="Vector store ready ✅")
                prog_text.empty()

                with st.spinner("🔧 Compiling LangGraph agent…"):
                    agent, tools = build_agent(gemini_key, vs)
                    st.session_state.agent       = agent
                    st.session_state.initialized = True

                st.success("✅ MediBot is ready!")
                st.rerun()

            except Exception as e:
                st.error(f"Initialization failed:\n{e}")

    st.divider()

    # ── Stats ────────────────────────────────────────────────────────────────
    if st.session_state.initialized:
        st.subheader("📊 Session Stats")
        st.markdown(
            f'<div class="metric-card">💬 Queries: <b>{st.session_state.total_queries}</b></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.tools_counter:
            st.markdown("**🔧 Tool usage:**")
            for tname, cnt in st.session_state.tools_counter.items():
                label = TOOL_LABELS.get(tname, tname)
                st.markdown(
                    f'<div class="metric-card">{label}: <b>{cnt}×</b></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Demo questions ────────────────────────────────────────────────
        st.subheader("💡 Demo Questions")
        for q in DEMO_QUESTIONS:
            if st.button(q, key=f"demo_{q[:30]}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.divider()

        # ── Controls ──────────────────────────────────────────────────────
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history  = []
            st.session_state.display_msgs  = []
            st.session_state.total_queries = 0
            st.session_state.tools_counter = {}
            st.rerun()

    # ── Architecture info ────────────────────────────────────────────────────
    with st.expander("ℹ️ Architecture"):
        st.markdown("""
**LLM:** Google Gemini 1.5 Flash  
**Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)  
**Vector Store:** FAISS (in-memory)  
**Orchestration:** LangGraph ReAct  

**Tools:**
- 📚 RAG Retriever
- 🌐 Live Wikipedia
- 🩺 Symptom Checker

**Flow:**
```
User → Agent → Tool(s)? → Agent → Answer
```
        """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏥 MediBot — Medical AI Assistant")
st.markdown(
    '<div class="disclaimer">⚠️ <b>Disclaimer:</b> MediBot provides information for '
    '<b>educational purposes only</b>. It is not a substitute for professional medical '
    'advice, diagnosis, or treatment. Always consult a qualified healthcare professional.</div>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── Not yet initialized ───────────────────────────────────────────────────────
if not st.session_state.initialized:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nPaste your **Gemini API key** in the sidebar.")
    with col2:
        st.info("**Step 2**\n\nClick **Initialize MediBot** — it will download medical articles and build the vector store.")
    with col3:
        st.info("**Step 3**\n\nStart chatting! Ask about diseases, symptoms, treatments, and more.")
    st.stop()

# ── Chat display ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.display_msgs:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 <b>You:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Bot reply
            reply_html = msg["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="bot-bubble">🤖 <b>MediBot:</b><br><br>{reply_html}</div>',
                unsafe_allow_html=True,
            )
            # Tool badges
            tools_used = msg.get("tools_used", [])
            if tools_used:
                badges = "".join(
                    f'<span class="badge">{TOOL_LABELS.get(t, t)}</span>'
                    for t in tools_used
                )
                st.markdown(
                    f'<div class="tool-strip"><b>🔧 Tools Used:</b> {badges}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="tool-strip"><b>🔧 Tools Used:</b> <i>None — answered from LLM knowledge</i></div>',
                    unsafe_allow_html=True,
                )

# ── Input area ────────────────────────────────────────────────────────────────
st.divider()

# Handle demo-button injection
pending = st.session_state.pop("pending_question", None)

with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([8, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask MediBot",
            value=pending or "",
            placeholder="e.g. What are the symptoms of asthma?",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Send 🏥", use_container_width=True, type="primary")

if submitted and user_input.strip():
    question = user_input.strip()

    # Add user bubble immediately
    st.session_state.display_msgs.append({"role": "user", "content": question})

    with st.spinner("🤔 MediBot is thinking…"):
        try:
            result = ask_medibot(
                st.session_state.agent,
                question,
                st.session_state.chat_history,
            )
            reply      = result["reply"]
            tools_used = result["tools_used"]
        except Exception as e:
            reply      = f"❌ Error: {e}"
            tools_used = []

    # Add bot bubble
    st.session_state.display_msgs.append({
        "role":       "assistant",
        "content":    reply,
        "tools_used": tools_used,
    })

    # Update stats
    st.session_state.total_queries += 1
    for t in tools_used:
        st.session_state.tools_counter[t] = st.session_state.tools_counter.get(t, 0) + 1

    st.rerun()
