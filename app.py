"""
🏥 MediBot — Medical RAG Chatbot (Clean Chatbot UI)
No setup screens. Everything initialises silently in the background.
Put HUGGINGFACEHUB_API_TOKEN in .env  OR  Streamlit secrets.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Page config — must be first ──────────────────────────────
st.set_page_config(
    page_title="MediBot",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS — full-screen chat like ChatGPT / Gemini ─────────────
st.markdown("""
<style>
/* hide streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

/* page background */
body, .stApp { background: #0f1117; color: #e8eaed; }

/* ── top bar ── */
.topbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 999;
    background: #1a1d27;
    border-bottom: 1px solid #2e3045;
    padding: 14px 28px;
    display: flex; align-items: center; gap: 12px;
}
.topbar-logo { font-size: 1.5rem; }
.topbar-title { font-size: 1.15rem; font-weight: 700; color: #e8eaed; }
.topbar-sub   { font-size: .78rem; color: #8b8fa8; margin-left: 4px; }
.topbar-badge {
    margin-left: auto;
    font-size: .73rem; padding: 3px 12px;
    border-radius: 20px;
    background: #1e3a5f; color: #5aabff;
    font-weight: 600;
}

/* ── chat scroll area ── */
.chat-area {
    margin-top: 70px;
    margin-bottom: 110px;
    padding: 0 8px;
    display: flex; flex-direction: column; gap: 18px;
}

/* ── message rows ── */
.msg-row { display: flex; gap: 10px; align-items: flex-end; }
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.bot   { flex-direction: row; }

/* avatars */
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.avatar.user { background: #2563eb; }
.avatar.bot  { background: #1e3a5f; }

/* bubbles */
.bubble {
    max-width: 76%; padding: 13px 17px;
    line-height: 1.65; font-size: .95rem;
    border-radius: 18px;
}
.bubble.user {
    background: #2563eb; color: #fff;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: #1a1d27; color: #e0e4f0;
    border: 1px solid #2e3045;
    border-bottom-left-radius: 4px;
    white-space: pre-wrap;
}

/* tools pill inside bot bubble */
.tools-pill {
    margin-top: 10px;
    display: inline-flex; gap: 6px; flex-wrap: wrap;
}
.tools-pill span {
    background: #12213a; color: #5aabff;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2px 10px; font-size: .72rem;
}

/* ── welcome screen ── */
.welcome {
    text-align: center;
    padding: 60px 20px 20px;
    color: #8b8fa8;
}
.welcome h2 { font-size: 2rem; color: #e8eaed; margin-bottom: 8px; }
.welcome p  { font-size: .95rem; margin-bottom: 32px; }
.chip-grid  { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
.chip {
    background: #1a1d27; border: 1px solid #2e3045;
    border-radius: 24px; padding: 9px 18px;
    font-size: .85rem; color: #c4c9e0; cursor: pointer;
    transition: border-color .2s;
}
.chip:hover { border-color: #2563eb; color: #fff; }

/* ── bottom input bar ── */
.input-bar {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
    background: #0f1117;
    border-top: 1px solid #1e2030;
    padding: 14px 20px 18px;
    display: flex; gap: 10px; align-items: center;
}
.input-bar input[type="text"] {
    flex: 1; background: #1a1d27 !important;
    border: 1px solid #2e3045 !important;
    border-radius: 28px !important;
    padding: 13px 20px !important;
    color: #e8eaed !important; font-size: .95rem !important;
    outline: none !important;
}
.input-bar input[type="text"]:focus { border-color: #2563eb !important; }
.input-bar input[type="text"]::placeholder { color: #4a4f6a !important; }

/* style the streamlit text_input wrapper to blend in */
[data-testid="stTextInput"] > div > div { padding: 0 !important; border: none !important; background: transparent !important; }
[data-testid="stTextInput"] input { color: #e8eaed !important; }

/* send button */
div[data-testid="stButton"] > button {
    background: #2563eb !important; color: white !important;
    border: none !important; border-radius: 50% !important;
    width: 46px !important; height: 46px !important;
    font-size: 1.2rem !important; padding: 0 !important;
    cursor: pointer !important;
    transition: background .2s !important;
}
div[data-testid="stButton"] > button:hover { background: #1d4ed8 !important; }

/* disclaimer */
.disclaimer {
    text-align: center; font-size: .72rem;
    color: #3a3f5c; padding-bottom: 2px;
}

/* typing indicator */
.typing {
    display: flex; gap: 5px; align-items: center;
    padding: 12px 16px;
    background: #1a1d27; border: 1px solid #2e3045;
    border-radius: 18px; border-bottom-left-radius: 4px;
    width: fit-content;
}
.dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #5aabff;
    animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce {
    0%,80%,100% { transform: translateY(0); opacity:.4; }
    40%          { transform: translateY(-6px); opacity:1; }
}

/* remove default streamlit padding */
.block-container { padding: 0 !important; max-width: 820px !important; }
</style>
""", unsafe_allow_html=True)


# ── Resolve HF token (env / secrets / fallback) ──────────────
HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
    if hasattr(st, "secrets") else ""
)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# ── Silent cached resource loaders ───────────────────────────

@st.cache_resource(show_spinner=False)
def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def _vectorstore(_emb):
    import wikipediaapi
    from time import sleep
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    TOPICS = [
        "Diabetes mellitus","Hypertension","COVID-19","Cancer","Asthma",
        "Tuberculosis","Pneumonia","Influenza","Dengue fever","Malaria",
        "Heart disease","Stroke","Kidney failure","Liver disease","Obesity",
        "Anemia","Arthritis","Migraine","Epilepsy","Parkinson's disease",
        "Alzheimer's disease","Human body","Brain","Heart","Lung","Kidney",
        "Liver","Digestive system","Nervous system","Immune system",
        "Respiratory system","Circulatory system","Endocrine system",
        "Fever","Cough","Chest pain","Headache","Fatigue",
        "Shortness of breath","Abdominal pain","Diarrhea","Weight loss",
        "Antibiotic","Vaccination","Insulin","Chemotherapy",
        "Paracetamol","Ibuprofen","Pain management",
        "Nutrition","Vitamin","Mental health","Exercise","Public health",
    ]
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/1.0")
    docs = []
    for topic in TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                docs.append(Document(
                    page_content=page.text[:5000],
                    metadata={"title": page.title, "source": page.fullurl},
                ))
        except Exception:
            pass
        sleep(0.2)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks   = splitter.split_documents(docs)
    return FAISS.from_documents(documents=chunks, embedding=_emb)


@st.cache_resource(show_spinner=False)
def _agent(_vs):
    import wikipediaapi
    from typing import List, Annotated
    from typing_extensions import TypedDict
    from transformers import pipeline as hf_pipeline
    from langchain_huggingface.llms import HuggingFacePipeline
    from langchain_huggingface import ChatHuggingFace
    from langchain_core.tools import create_retriever_tool, tool
    from langchain_core.messages import SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    pipe = hf_pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=512,
        temperature=0.2,
        device_map="auto",
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    rag_tool = create_retriever_tool(
        _vs.as_retriever(search_kwargs={"k": 5}),
        name="medical_rag_retriever",
        description=(
            "Search the local medical knowledge base built from Wikipedia. "
            "Use this FIRST for any medical question about diseases, symptoms, "
            "treatments, medications, anatomy, or health conditions."
        ),
    )

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base."""
        wc = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/1.0")
        try:
            p = wc.page(query)
            return p.text[:3000] if p.exists() else f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f"Wikipedia error: {e}"

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Given comma-separated symptoms, return possible medical conditions."""
        MAP = {
            "fever":               ["Influenza","COVID-19","Malaria","Typhoid"],
            "cough":               ["Asthma","COVID-19","Tuberculosis","Bronchitis"],
            "chest pain":          ["Heart disease","Angina","Pneumonia","GERD"],
            "headache":            ["Migraine","Hypertension","Tension headache","Meningitis"],
            "fatigue":             ["Anemia","Diabetes","Hypothyroidism","Depression"],
            "shortness of breath": ["Asthma","Heart failure","COVID-19","Pulmonary embolism"],
            "frequent urination":  ["Diabetes mellitus","UTI","Diabetes insipidus"],
            "weight loss":         ["Diabetes","Cancer","Tuberculosis","Hyperthyroidism"],
            "joint pain":          ["Arthritis","Gout","Lupus","Fibromyalgia"],
            "rash":                ["Eczema","Psoriasis","Allergic reaction","Lupus"],
            "nausea":              ["Gastritis","Food poisoning","Pregnancy","Migraine"],
            "dizziness":           ["Hypertension","Anemia","Inner ear disorder","Dehydration"],
            "vomiting":            ["Gastritis","Food poisoning","Appendicitis","Migraine"],
            "abdominal pain":      ["Gastritis","Appendicitis","IBS","Kidney stones"],
            "back pain":           ["Muscle strain","Herniated disc","Kidney stones","Sciatica"],
        }
        entered = [s.strip().lower() for s in symptoms.split(",")]
        result  = {k: v for k, v in MAP.items() if any(k in s for s in entered)}
        if not result:
            return "No specific matches found. Please describe symptoms more clearly."
        lines = ["⚠️ Possible conditions (educational only):\n"]
        for sym, conds in result.items():
            lines.append(f"  • {sym.capitalize()}: {', '.join(conds)}")
        lines.append("\n🩺 Always consult a licensed physician for proper diagnosis.")
        return "\n".join(lines)

    tools          = [rag_tool, wikipedia_live_tool, symptom_checker]
    llm_with_tools = llm.bind_tools(tools)
    tool_node      = ToolNode(tools)

    SYSTEM = """You are MediBot, an expert Medical AI Assistant.
Answer all medical questions clearly, accurately, and in detail.
Tools available:
1. medical_rag_retriever — search local FAISS knowledge base (use FIRST)
2. wikipedia_live_tool   — live Wikipedia fallback
3. symptom_checker       — symptom → condition mapper

Rules: always use medical_rag_retriever first. Give structured, detailed answers.
End every reply with: ⚠️ For informational purposes only. Always consult a healthcare professional."""

    class State(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    def call_model(state: State):
        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM)] + msgs
        return {"messages": [llm_with_tools.invoke(msgs)]}

    def run_tools(state: State):
        last = state["messages"][-1]
        used = list(state.get("tools_used", []))
        if hasattr(last, "tool_calls"):
            for tc in last.tool_calls:
                name = tc.get("name", "")
                if name and name not in used:
                    used.append(name)
        res = tool_node.invoke(state)
        res["tools_used"] = used
        return res

    def route(state: State):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", run_tools)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# ── Session state ─────────────────────────────────────────────
for key, default in [
    ("messages", []),        # [{role, content, tools}]
    ("lc_messages", []),     # LangChain message objects
    ("booting", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

TOOL_LABELS = {
    "medical_rag_retriever": "📚 Knowledge Base",
    "wikipedia_live_tool":   "🌐 Wikipedia",
    "symptom_checker":       "🩺 Symptom Checker",
}

SUGGESTIONS = [
    "What is diabetes and how is it treated?",
    "I have fever, cough and fatigue — what could it be?",
    "Explain hypertension causes and prevention",
    "What are the side effects of antibiotics?",
    "How does the immune system fight infections?",
    "Difference between Type 1 and Type 2 diabetes?",
]

# ── Top bar ───────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <span class="topbar-logo">🏥</span>
    <span class="topbar-title">MediBot</span>
    <span class="topbar-sub">Medical AI Assistant</span>
    <span class="topbar-badge">● Online</span>
</div>
""", unsafe_allow_html=True)


# ── Silently boot the pipeline on first load ──────────────────
if st.session_state.booting:
    with st.spinner(""):          # invisible – just keeps the thread alive
        emb = _embeddings()
        vs  = _vectorstore(emb)
        _agent(vs)
    st.session_state.booting = False
    st.rerun()

# resolve cached objects
emb   = _embeddings()
vs    = _vectorstore(emb)
agent = _agent(vs)


# ── Chat area ────────────────────────────────────────────────
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    # Welcome screen with suggestion chips
    chip_html = "".join(
        f'<div class="chip" onclick="document.querySelector(\'input[type=text]\').value=\'{q}\'">{q}</div>'
        for q in SUGGESTIONS
    )
    st.markdown(f"""
    <div class="welcome">
        <h2>🏥 MediBot</h2>
        <p>Your AI-powered Medical Assistant.<br>Ask me anything about symptoms, diseases, treatments or medications.</p>
        <div class="chip-grid">{chip_html}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑" if role == "user" else "🤖"
        a_cls  = "user" if role == "user" else "bot"
        b_cls  = "user" if role == "user" else "bot"

        tools_html = ""
        if role == "assistant" and msg.get("tools"):
            pills = "".join(
                f'<span>{TOOL_LABELS.get(t, t)}</span>'
                for t in msg["tools"]
            )
            tools_html = f'<div class="tools-pill">{pills}</div>'

        st.markdown(f"""
        <div class="msg-row {a_cls}">
            <div class="avatar {a_cls}">{avatar}</div>
            <div class="bubble {b_cls}">{msg["content"]}{tools_html}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Input bar ────────────────────────────────────────────────
st.markdown('<div class="input-bar">', unsafe_allow_html=True)

col_txt, col_btn = st.columns([10, 1])
with col_txt:
    user_input = st.text_input(
        "msg",
        placeholder="Ask a medical question…",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("➤", key="send_btn")

st.markdown("""
<div class="disclaimer">
    ⚠️ MediBot is for educational purposes only — not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Process message ──────────────────────────────────────────
question = user_input.strip() if (send and user_input) else ""

if question:
    from langchain_core.messages import HumanMessage, AIMessage

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.lc_messages.append(HumanMessage(content=question))

    with st.spinner("MediBot is thinking…"):
        try:
            result = agent.invoke(
                {"messages": st.session_state.lc_messages, "tools_used": []},
                config={"recursion_limit": 20},
            )
            ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
            if ai_msgs:
                final   = ai_msgs[-1]
                reply   = final.content if isinstance(final.content, str) else str(final.content)
                used    = result.get("tools_used", [])
                st.session_state.lc_messages.append(final)
                st.session_state.messages.append({"role": "assistant", "content": reply, "tools": used})
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't generate a response. Please try again.",
                    "tools": [],
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Something went wrong: {e}",
                "tools": [],
            })

    st.rerun()
