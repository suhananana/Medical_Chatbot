"""
🏥 MediBot — Medical RAG Chatbot
Streamlit deployment of the LangChain + LangGraph + FAISS Medical AI Assistant
"""

import os
import time
import streamlit as st
from typing import List, Annotated
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Header banner ── */
.medibot-header {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
}
.medibot-header h1 { margin: 0; font-size: 2.2rem; }
.medibot-header p  { margin: 6px 0 0; opacity: .85; font-size: 1rem; }

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 14px; }

.bubble-user {
    align-self: flex-end;
    background: #1a73e8;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 78%;
    font-size: .97rem;
    line-height: 1.5;
    box-shadow: 0 2px 6px rgba(26,115,232,.3);
}

.bubble-bot {
    align-self: flex-start;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    max-width: 86%;
    font-size: .96rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    color: #1a1a2e;
}

.bubble-bot .tools-badge {
    margin-top: 10px;
    font-size: .78rem;
    color: #666;
    background: #f5f5f5;
    border-radius: 8px;
    padding: 5px 10px;
    display: inline-block;
}

/* ── Info card ── */
.info-card {
    background: #e8f4fd;
    border-left: 4px solid #1a73e8;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: .9rem;
    color: #0d47a1;
    margin-bottom: 16px;
}

/* ── Status pill ── */
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: .82rem;
    font-weight: 600;
}
.status-ready   { background: #e6f4ea; color: #2d7a3a; }
.status-loading { background: #fff3e0; color: #e65100; }
.status-error   { background: #fce8e6; color: #c5221f; }

/* ── Sidebar sections ── */
.sidebar-section {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
    font-size: .88rem;
}
.sidebar-section h4 { margin: 0 0 8px; color: #1a73e8; font-size: .95rem; }

/* ── Spinner override ── */
div[data-testid="stSpinner"] > div { color: #1a73e8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCE INITIALISATION
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(_embeddings):
    import wikipediaapi
    from time import sleep
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    MEDICAL_TOPICS = [
        # Diseases
        "Diabetes mellitus", "Hypertension", "COVID-19", "Cancer", "Asthma",
        "Tuberculosis", "Pneumonia", "Influenza", "Dengue fever", "Malaria",
        "Heart disease", "Stroke", "Kidney failure", "Liver disease", "Obesity",
        "Anemia", "Arthritis", "Migraine", "Epilepsy",
        "Parkinson's disease", "Alzheimer's disease",
        # Anatomy
        "Human body", "Brain", "Heart", "Lung", "Kidney", "Liver",
        "Digestive system", "Nervous system", "Immune system",
        "Respiratory system", "Circulatory system", "Endocrine system",
        # Symptoms
        "Fever", "Cough", "Chest pain", "Headache", "Fatigue",
        "Shortness of breath", "Abdominal pain", "Diarrhea", "Weight loss",
        # Medicines & Treatments
        "Antibiotic", "Vaccination", "Insulin", "Chemotherapy",
        "Paracetamol", "Ibuprofen", "Pain management",
        # Health
        "Nutrition", "Vitamin", "Mental health", "Exercise", "Public health",
    ]

    wiki = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/1.0")
    docs = []
    progress = st.progress(0, text="Loading medical knowledge base…")

    for i, topic in enumerate(MEDICAL_TOPICS):
        try:
            page = wiki.page(topic)
            if page.exists():
                docs.append(Document(
                    page_content=page.text[:5000],
                    metadata={"title": page.title, "source": page.fullurl},
                ))
        except Exception:
            pass
        sleep(0.25)
        progress.progress((i + 1) / len(MEDICAL_TOPICS),
                          text=f"Loading: {topic} ({i+1}/{len(MEDICAL_TOPICS)})")

    progress.empty()

    if not docs:
        st.error("❌ Could not load any Wikipedia articles. Check your internet connection.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks   = splitter.split_documents(docs)
    vs       = FAISS.from_documents(documents=chunks, embedding=_embeddings)
    return vs, len(docs), vs.index.ntotal


@st.cache_resource(show_spinner=False)
def load_llm(hf_token: str):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    from transformers import pipeline
    from langchain_huggingface.llms import HuggingFacePipeline
    from langchain_huggingface import ChatHuggingFace

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=512,
        temperature=0.2,
        device_map="auto",
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=hf_pipeline)
    return llm


@st.cache_resource(show_spinner=False)
def build_agent(hf_token: str, _vectorstore):
    from langchain_core.tools import create_retriever_tool, tool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.documents import Document
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    import wikipediaapi

    llm = load_llm(hf_token)

    # ── Tool 1: RAG Retriever ──────────────────────────────────
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool  = create_retriever_tool(
        retriever,
        name="medical_rag_retriever",
        description=(
            "Search the local medical knowledge base built from Wikipedia. "
            "Use this FIRST for any medical question about diseases, symptoms, "
            "treatments, medications, anatomy, or health conditions."
        ),
    )

    # ── Tool 2: Live Wikipedia ─────────────────────────────────
    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base."""
        wiki_client = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/1.0")
        try:
            page = wiki_client.page(query)
            if page.exists():
                return page.text[:3000]
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f"Wikipedia search error: {e}"

    # ── Tool 3: Symptom Checker ────────────────────────────────
    @tool
    def symptom_checker(symptoms: str) -> str:
        """Given a comma-separated list of symptoms, return possible medical conditions."""
        symptom_map = {
            "fever":               ["Influenza", "COVID-19", "Malaria", "Typhoid"],
            "cough":               ["Asthma", "COVID-19", "Tuberculosis", "Bronchitis"],
            "chest pain":          ["Heart disease", "Angina", "Pneumonia", "GERD"],
            "headache":            ["Migraine", "Hypertension", "Tension headache", "Meningitis"],
            "fatigue":             ["Anemia", "Diabetes", "Hypothyroidism", "Depression"],
            "shortness of breath": ["Asthma", "Heart failure", "COVID-19", "Pulmonary embolism"],
            "frequent urination":  ["Diabetes mellitus", "UTI", "Diabetes insipidus"],
            "weight loss":         ["Diabetes", "Cancer", "Tuberculosis", "Hyperthyroidism"],
            "joint pain":          ["Arthritis", "Gout", "Lupus", "Fibromyalgia"],
            "rash":                ["Eczema", "Psoriasis", "Allergic reaction", "Lupus"],
            "nausea":              ["Gastritis", "Food poisoning", "Pregnancy", "Migraine"],
            "dizziness":           ["Hypertension", "Anemia", "Inner ear disorder", "Dehydration"],
            "vomiting":            ["Gastritis", "Food poisoning", "Appendicitis", "Migraine"],
            "abdominal pain":      ["Gastritis", "Appendicitis", "IBS", "Kidney stones"],
            "back pain":           ["Muscle strain", "Herniated disc", "Kidney stones", "Sciatica"],
        }
        entered = [s.strip().lower() for s in symptoms.split(",")]
        result  = {}
        for sym in entered:
            for key, conds in symptom_map.items():
                if key in sym:
                    result[key] = conds

        if not result:
            return ("No specific matches found. Please describe symptoms more clearly "
                    "or consult a healthcare professional.")
        lines = ["⚠️ Possible conditions (educational only):\n"]
        for sym, conds in result.items():
            lines.append(f"  • {sym.capitalize()}: {', '.join(conds)}")
        lines.append("\n🩺 Always consult a licensed physician for proper diagnosis.")
        return "\n".join(lines)

    tools          = [rag_tool, wikipedia_live_tool, symptom_checker]
    llm_with_tools = llm.bind_tools(tools)
    tool_node      = ToolNode(tools)

    SYSTEM_PROMPT = """
You are MediBot, an expert Medical AI Assistant.

Your job is to answer medical questions clearly, accurately, and in detail.

You have access to these tools:
1. medical_rag_retriever  → search the local FAISS knowledge base
2. wikipedia_live_tool    → search Wikipedia live
3. symptom_checker        → map symptoms to conditions

STRICT RULES:
- ALWAYS use medical_rag_retriever first.
- Use symptom_checker if symptoms are mentioned.
- Use wikipedia_live_tool for extra details.
- Give structured and detailed answers.
- Mention tools used at the end.

Always end with:
⚠️ For informational purposes only. Always consult a healthcare professional.
"""

    class AgentState(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    def call_model(state: AgentState):
        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def run_tools(state: AgentState):
        last = state["messages"][-1]
        used = list(state.get("tools_used", []))
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get("name", "unknown")
                if name not in used:
                    used.append(name)
        result = tool_node.invoke(state)
        result["tools_used"] = used
        return result

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    agent = graph.compile()

    return agent


# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of {role, content, tools}
if "conv_messages" not in st.session_state:
    st.session_state.conv_messages = []         # LangChain message objects
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

TOOL_LABELS = {
    "medical_rag_retriever": "📚 RAG Retriever",
    "wikipedia_live_tool":   "🌐 Wikipedia Live",
    "symptom_checker":       "🩺 Symptom Checker",
}

EXAMPLE_QUESTIONS = [
    "What is diabetes mellitus and how is it treated?",
    "I have fever, cough, and fatigue — what could it be?",
    "Explain hypertension causes and prevention",
    "What are the side effects of antibiotics?",
    "How does the immune system fight infections?",
    "What is the difference between Type 1 and Type 2 diabetes?",
]

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediBot Setup")

    # ── HF Token ──
    st.markdown("### 🔑 Hugging Face Token")
    hf_input = st.text_input(
        "Enter your HF token",
        type="password",
        value=st.session_state.hf_token,
        placeholder="hf_...",
        help="Get a free token at https://huggingface.co/settings/tokens",
    )
    if hf_input:
        st.session_state.hf_token = hf_input

    st.caption("[Get a free token →](https://huggingface.co/settings/tokens)")

    st.divider()

    # ── Init button ──
    init_btn = st.button("🚀 Initialise MediBot", use_container_width=True,
                         disabled=not st.session_state.hf_token)

    if init_btn:
        if not st.session_state.hf_token:
            st.error("Please enter a Hugging Face token first.")
        else:
            with st.spinner("Loading embedding model…"):
                embeddings = load_embeddings()
            with st.spinner("Building knowledge base from Wikipedia (≈ 2 min)…"):
                vs, n_docs, n_vecs = build_vectorstore(embeddings)
            with st.spinner("Loading TinyLlama LLM…"):
                build_agent(st.session_state.hf_token, vs)
            st.session_state.kb_ready  = True
            st.session_state.n_docs    = n_docs
            st.session_state.n_vecs    = n_vecs
            st.success("✅ MediBot is ready!")

    st.divider()

    # ── Status card ──
    if st.session_state.kb_ready:
        st.markdown(f"""
        <div class="sidebar-section">
            <h4>📊 Knowledge Base</h4>
            📄 Articles loaded: <b>{st.session_state.get('n_docs', '—')}</b><br>
            🔢 Vectors indexed: <b>{st.session_state.get('n_vecs', '—')}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-section">
            <h4>📊 Status</h4>
            ⚠️ Not initialised yet.<br>Enter token and click <b>Initialise</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🔧 Tools Available</h4>
        📚 RAG Retriever (FAISS)<br>
        🌐 Live Wikipedia<br>
        🩺 Symptom Checker
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.conv_messages = []
        st.rerun()

    st.markdown("""
    <div class="sidebar-section">
        <h4>⚠️ Disclaimer</h4>
        MediBot is for <b>educational purposes only</b>.<br>
        Always consult a licensed healthcare professional.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN AREA — HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="medibot-header">
    <h1>🏥 MediBot — Medical AI Assistant</h1>
    <p>Powered by LangChain · LangGraph · FAISS · TinyLlama · Wikipedia</p>
</div>
""", unsafe_allow_html=True)

# ── Welcome / not-ready message ──
if not st.session_state.kb_ready:
    st.markdown("""
    <div class="info-card">
        👋 Welcome! Enter your <b>Hugging Face token</b> in the sidebar and click
        <b>Initialise MediBot</b> to load the knowledge base and start chatting.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Example questions you can ask:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        cols[i % 2].markdown(f"- {q}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="info-card">
            👋 MediBot is ready! Ask any medical question below.
        </div>
        """, unsafe_allow_html=True)

    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f'<div class="bubble-user">🧑 {turn["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            tools_html = ""
            if turn.get("tools"):
                tool_str  = " · ".join(TOOL_LABELS.get(t, t) for t in turn["tools"])
                tools_html = f'<div class="tools-badge">🔧 Tools used: {tool_str}</div>'
            st.markdown(
                f'<div class="bubble-bot">🤖 <b>MediBot:</b><br>{turn["content"]}{tools_html}</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────
# INPUT ROW
# ─────────────────────────────────────────────────────────────
st.divider()

# Quick-question buttons
st.markdown("**💡 Quick questions:**")
qcols = st.columns(3)
quick_clicked = None
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if qcols[i % 3].button(q, key=f"quick_{i}"):
        quick_clicked = q

col_input, col_send = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Your question",
        value=quick_clicked or "",
        placeholder="e.g. What is hypertension? / I have fever and cough…",
        label_visibility="collapsed",
        key="user_input_box",
    )
with col_send:
    send_btn = st.button("Ask 🏥", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────────────────────────
question = user_input.strip() if send_btn and user_input else (quick_clicked or "")

if question:
    from langchain_core.messages import HumanMessage, AIMessage

    # Append user bubble immediately
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.conv_messages.append(HumanMessage(content=question))

    with st.spinner("⏳ MediBot is thinking…"):
        try:
            embeddings = load_embeddings()
            vs, _, _   = build_vectorstore(embeddings)
            agent      = build_agent(st.session_state.hf_token, vs)

            result = agent.invoke(
                {"messages": st.session_state.conv_messages, "tools_used": []},
                config={"recursion_limit": 20},
            )

            ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
            if ai_msgs:
                final_msg  = ai_msgs[-1]
                reply      = (final_msg.content
                              if isinstance(final_msg.content, str)
                              else str(final_msg.content))
                tools_used = result.get("tools_used", [])

                st.session_state.conv_messages.append(final_msg)
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": reply,
                    "tools":   tools_used,
                })
            else:
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": "❌ No response received. Please try again.",
                    "tools":   [],
                })

        except Exception as e:
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": f"❌ Error: {e}",
                "tools":   [],
            })

    st.rerun()
