import os, streamlit as st
from collections import defaultdict
from typing import List, Annotated
from typing_extensions import TypedDict
from time import sleep

# ── API Keys — paste your actual keys here ───────────────────────────────────
os.environ["GOOGLE_API_KEY"] = "your_gemini_key_here"
os.environ["GROQ_API_KEY"]   = "your_groq_key_here"
# os.environ["OPENAI_API_KEY"] = "your_openai_key_here"  # optional

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lora:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; color: #e8e8e8; }

section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] * { color: #c8d6e5 !important; }

.medibot-header {
    text-align: center;
    padding: 1.4rem 0 0.8rem;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 1.2rem;
}
.medibot-header h1 {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #4fc3f7;
    margin: 0;
    letter-spacing: -0.5px;
}
.medibot-header p {
    color: #4a7a96;
    font-size: 0.76rem;
    margin: 0.25rem 0 0;
    font-family: 'DM Mono', monospace;
}

.chat-user {
    background: #1a2535;
    border: 1px solid #2a3f55;
    border-radius: 14px 14px 4px 14px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 0.3rem 4rem;
    color: #c8d6e5;
    font-size: 0.92rem;
    line-height: 1.6;
}
.chat-bot {
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 14px 14px 14px 4px;
    padding: 0.9rem 1.1rem;
    margin: 0.3rem 4rem 0.3rem 0;
    color: #dde6f0;
    font-size: 0.92rem;
    line-height: 1.65;
}

.tool-box {
    background: #0b1a26;
    border: 1px solid #14334d;
    border-left: 3px solid #00bcd4;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 0.9rem;
    margin: 0.3rem 4rem 0.8rem 0;
    font-family: 'DM Mono', monospace;
}
.tool-box .tb-header {
    color: #00bcd4;
    font-size: 0.67rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.tool-badge {
    display: inline-block;
    background: #0d1f30;
    border: 1px solid #1a4060;
    color: #81d4fa;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 4px 4px 0;
    font-size: 0.7rem;
}
.tool-row {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    margin: 5px 0;
    font-size: 0.73rem;
    color: #6b9ab8;
}
.tool-row .chk { color: #26c6da; flex-shrink: 0; }
.tool-row em   { color: #3a6a84; font-style: italic; }

.no-tool {
    background: #0b1a26;
    border: 1px solid #14334d;
    border-left: 3px solid #2d4a5a;
    border-radius: 0 8px 8px 0;
    padding: 0.45rem 0.85rem;
    margin: 0.3rem 4rem 0.8rem 0;
    font-size: 0.73rem;
    color: #3a5a70;
    font-family: 'DM Mono', monospace;
}

.stTextInput input {
    background: #161b27 !important;
    border: 1px solid #2a3f55 !important;
    color: #c8d6e5 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,0.1) !important;
}

.stButton button {
    background: #1565c0 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
.stButton button:hover { background: #1976d2 !important; }

.stSelectbox > div > div {
    background: #161b27 !important;
    border: 1px solid #2a3f55 !important;
    color: #c8d6e5 !important;
    border-radius: 8px !important;
}

.chip-btn button {
    background: #0d1520 !important;
    border: 1px solid #1e3a50 !important;
    color: #4fc3f7 !important;
    font-size: 0.78rem !important;
    border-radius: 20px !important;
    padding: 0.2rem 0.8rem !important;
}

.disclaimer {
    background: #1a1209;
    border: 1px solid #3d2800;
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    font-size: 0.73rem;
    color: #9a7040;
    margin-top: 0.8rem;
}

.metric-box {
    background: #0d1520;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.4rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.4rem;
    font-weight: 600;
    color: #4fc3f7;
    font-family: 'DM Mono', monospace;
}
.metric-box .lbl {
    font-size: 0.67rem;
    color: #4a7a96;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Tool metadata ──────────────────────────────────────────────────────────────
TOOL_LABELS = {
    'hybrid_rag_retriever': '🔀 Hybrid RAG',
    'faiss_rag_retriever':  '🧠 FAISS Semantic',
    'bm25_rag_retriever':   '🔑 BM25 Keyword',
    'wikipedia_live_tool':  '🌐 Wikipedia Live',
    'symptom_checker':      '🩺 Symptom Checker',
    'drug_information':     '💊 Drug Info DB',
}
TOOL_WHY = {
    'hybrid_rag_retriever': 'Primary retriever — FAISS + BM25 for broadest medical coverage',
    'faiss_rag_retriever':  'Conceptual query detected — dense semantic search applied',
    'bm25_rag_retriever':   'Exact term / drug name found — keyword match preferred',
    'wikipedia_live_tool':  'Topic absent from local KB — expanded to live Wikipedia',
    'symptom_checker':      'Symptoms mentioned — urgency assessment triggered',
    'drug_information':     'Drug name detected — dosage, interactions & side effects fetched',
}

# ── Session state ──────────────────────────────────────────────────────────────
defaults = dict(
    initialized=False, chat_history=[], conv_history=[],
    current_model=None, available_models={}, agent=None,
    tools=None, query_count=0, tool_counts=defaultdict(int),
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# Core builders (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_kb():
    import wikipediaapi
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever

    wiki = wikipediaapi.Wikipedia(language='en', user_agent='MediBot-Streamlit/3.0')
    TOPICS = [
        'Diabetes mellitus','Hypertension','COVID-19','Cancer','Asthma',
        'Tuberculosis','Pneumonia','Influenza','Dengue fever','Malaria',
        'Heart disease','Stroke','Kidney failure','Liver disease','Obesity',
        'Anemia','Arthritis','Migraine','Epilepsy',
        "Parkinson's disease","Alzheimer's disease",
        'Human body','Brain','Heart','Lung','Kidney','Liver',
        'Digestive system','Nervous system','Immune system',
        'Respiratory system','Circulatory system','Endocrine system',
        'Fever','Cough','Chest pain','Headache','Fatigue',
        'Shortness of breath','Abdominal pain','Diarrhea','Weight loss',
        'Antibiotic','Vaccination','Insulin','Chemotherapy',
        'Paracetamol','Ibuprofen','Pain management','Metformin',
        'Aspirin','Amoxicillin',
        'Nutrition','Vitamin','Mental health','Exercise','Public health',
    ]
    docs = []
    for t in TOPICS:
        try:
            p = wiki.page(t)
            if p.exists():
                docs.append(Document(page_content=p.text[:5000],
                                     metadata={'title': p.title, 'source': p.fullurl}))
            sleep(0.2)
        except Exception:
            pass

    splitter  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks    = splitter.split_documents(docs)
    emb       = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device':'cpu'},
                                       encode_kwargs={'normalize_embeddings':True})
    vs        = FAISS.from_documents(chunks, emb)
    faiss_ret = vs.as_retriever(search_kwargs={'k':4})
    bm25_ret  = BM25Retriever.from_documents(chunks)
    bm25_ret.k = 4
    return faiss_ret, bm25_ret


def make_ensemble(faiss_ret, bm25_ret):
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from pydantic import ConfigDict

    def rrf(results, k=60):
        scores, docs = defaultdict(float), {}
        for lst in results:
            for rank, doc in enumerate(lst):
                did = (doc.page_content, doc.metadata.get('source'))
                docs.setdefault(did, doc)
                scores[did] += 1 / (k + rank)
        return [docs[i] for i in sorted(scores, key=lambda x: -scores[x])]

    class EnsembleRetriever(BaseRetriever):
        retrievers: list
        model_config = ConfigDict(arbitrary_types_allowed=True)
        def _get_relevant_documents(self, query: str) -> List[Document]:
            return rrf([r.invoke(query) for r in self.retrievers])
        async def _aget_relevant_documents(self, query):
            raise NotImplementedError

    return EnsembleRetriever(retrievers=[faiss_ret, bm25_ret])


def make_tools(faiss_ret, bm25_ret, ensemble):
    from langchain_core.tools import tool
    import wikipediaapi

    @tool
    def hybrid_rag_retriever(query: str) -> str:
        """Hybrid RAG: FAISS semantic + BM25 keyword via Reciprocal Rank Fusion. Use this FIRST as the PRIMARY retriever for all medical questions."""
        return str(ensemble.invoke(query))

    @tool
    def faiss_rag_retriever(query: str) -> str:
        """Semantic/dense vector search in local FAISS medical knowledge base. Best for conceptual medical questions about diseases, treatments, and anatomy."""
        return str(faiss_ret.invoke(query))

    @tool
    def bm25_rag_retriever(query: str) -> str:
        """Keyword BM25 search in local medical knowledge base. Best when exact medical terms, drug names, or specific condition names are mentioned."""
        return str(bm25_ret.invoke(query))

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base. Use as fallback for rare or recent conditions."""
        w = wikipediaapi.Wikipedia(language='en', user_agent='MediBot-Streamlit/3.0')
        try:
            p = w.page(query)
            return f"[{p.fullurl}]\n\n{p.text[:3000]}" if p.exists() else f"No page found for '{query}'."
        except Exception as e:
            return f'Wikipedia error: {e}'

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Analyze comma-separated symptoms and return possible conditions with urgency levels. Always use when user mentions symptoms."""
        MAP = {
            'fever':               (['Influenza','COVID-19','Malaria','Typhoid','Dengue'], 'Medium'),
            'cough':               (['Asthma','COVID-19','Tuberculosis','Bronchitis'], 'Low-Medium'),
            'chest pain':          (['Heart disease','Angina','Pneumonia','GERD'], '🚨 HIGH — Seek immediate care'),
            'headache':            (['Migraine','Hypertension','Tension headache','Meningitis'], 'Low-Medium'),
            'fatigue':             (['Anemia','Diabetes','Hypothyroidism','Depression'], 'Low'),
            'shortness of breath': (['Asthma','Heart failure','COVID-19','Pulmonary embolism'], '🚨 HIGH'),
            'frequent urination':  (['Diabetes mellitus','UTI','Prostate issues'], 'Medium'),
            'weight loss':         (['Diabetes','Cancer','Tuberculosis','Hyperthyroidism'], 'Medium-High'),
            'joint pain':          (['Arthritis','Gout','Lupus','Fibromyalgia'], 'Low-Medium'),
            'rash':                (['Eczema','Psoriasis','Allergic reaction','Lupus'], 'Low'),
            'nausea':              (['Gastritis','Food poisoning','Pregnancy','Migraine'], 'Low-Medium'),
            'dizziness':           (['Hypertension','Anemia','Inner ear disorder','Stroke'], 'Medium'),
            'abdominal pain':      (['Gastritis','Appendicitis','IBS','Kidney stones'], 'Medium-High'),
            'back pain':           (['Muscle strain','Herniated disc','Kidney stones','Sciatica'], 'Low-Medium'),
            'blurred vision':      (['Diabetes','Hypertension','Glaucoma','Stroke'], '🚨 HIGH'),
            'confusion':           (['Stroke','Hypoglycemia','Dementia','Encephalitis'], '🚨 HIGH'),
        }
        entered = [s.strip().lower() for s in symptoms.split(',')]
        found   = {k: v for k, v in MAP.items() if any(k in s for s in entered)}
        if not found:
            return 'No matches found. Describe symptoms more clearly or consult a healthcare professional.'
        lines = ['Symptom Analysis (Educational Only):\n']
        for sym, (conds, urg) in found.items():
            lines += [f'  • {sym.capitalize()}',
                      f'    Possible: {", ".join(conds)}',
                      f'    Urgency: {urg}']
        lines.append('\nEducational only. Always consult a licensed physician.')
        return '\n'.join(lines)

    @tool
    def drug_information(drug_name: str) -> str:
        """Get drug details: class, uses, dosage, side effects, contraindications, interactions. Use when user asks about a specific medication."""
        DB = {
            'paracetamol': ('Analgesic/Antipyretic','Fever, mild-moderate pain','500-1000mg every 4-6h, max 4g/day','Rare; overdose → liver damage','Severe liver disease','Warfarin (high doses)'),
            'ibuprofen':   ('NSAID','Pain, fever, inflammation','200-400mg every 4-6h','GI upset, ulcers, increased BP','Peptic ulcer, kidney disease','Aspirin, warfarin, ACE inhibitors'),
            'amoxicillin': ('Penicillin Antibiotic','Bacterial infections','250-500mg every 8h','Diarrhea, nausea, rash','Penicillin allergy','Warfarin, oral contraceptives'),
            'metformin':   ('Biguanide Antidiabetic','Type 2 diabetes','500mg twice daily, max 2550mg/day','GI upset; rare lactic acidosis','Kidney failure, liver disease','Alcohol, contrast agents'),
            'aspirin':     ('Salicylate NSAID','Pain, fever; cardio protection','Pain: 325-650mg; Cardio: 75-100mg/day','GI irritation, tinnitus, bleeding','Children <16, peptic ulcer','Warfarin, ibuprofen, SSRIs'),
            'insulin':     ('Injectable Antidiabetic','Type 1 & 2 DM','Individualized — doctor supervision','Hypoglycemia, weight gain','Active hypoglycemia','Beta-blockers, corticosteroids'),
        }
        key = drug_name.lower().strip()
        m   = next((k for k in DB if k in key or key in k), None)
        if not m:
            return f"No local data for '{drug_name}'. Try hybrid_rag_retriever or wikipedia_live_tool."
        c,u,d,s,ci,i = DB[m]
        return (f"{m.capitalize()}\n  Class: {c}\n  Uses: {u}\n  Dosage: {d}\n"
                f"  Side Effects: {s}\n  Contraindications: {ci}\n  Interactions: {i}\n\n"
                f"Follow your doctor's / pharmacist's instructions.")

    return [hybrid_rag_retriever, faiss_rag_retriever, bm25_rag_retriever,
            wikipedia_live_tool, symptom_checker, drug_information]


def get_llm(name):
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:    from langchain_openai import ChatOpenAI
    except: ChatOpenAI = None
    try:    from langchain_groq import ChatGroq
    except: ChatGroq = None

    if name == 'gemini':
        return ChatGoogleGenerativeAI(model='gemini-2.0-flash',
            google_api_key=os.environ['GOOGLE_API_KEY'],
            temperature=0.2, max_output_tokens=1024,
            convert_system_message_to_human=True)
    if name == 'openai' and ChatOpenAI:
        return ChatOpenAI(model='gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'],
                          temperature=0.2, max_tokens=1024)
    if name == 'groq' and ChatGroq:
        return ChatGroq(model='llama-3.1-8b-instant', api_key=os.environ['GROQ_API_KEY'],
                        temperature=0.2, max_tokens=1024)
    raise ValueError(f"Model {name} unavailable")


def build_agent(model_name, tools):
    from langchain_core.messages import SystemMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    SYSTEM = """You are MediBot, an expert Medical AI Assistant with a Hybrid RAG system.

Tools: hybrid_rag_retriever (PRIMARY for all medical questions), faiss_rag_retriever,
bm25_rag_retriever, wikipedia_live_tool (fallback), symptom_checker (when symptoms mentioned),
drug_information (when a drug is named).

Rules:
- Always start with hybrid_rag_retriever for medical questions.
- Call symptom_checker when symptoms are described.
- Call drug_information when a drug/medication is named.
- Use wikipedia_live_tool for rare or recent topics.
- Use clear headings and bullet points in responses.
- End with a medical disclaimer.

*Medical Disclaimer: For educational purposes only. Always consult a qualified healthcare professional.*"""

    class State(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    llm            = get_llm(model_name)
    llm_with_tools = llm.bind_tools(tools)
    tool_node      = ToolNode(tools)

    def call_model(state):
        msgs = list(state['messages'])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {'messages': [response]}

    def run_tools(state):
        last = state['messages'][-1]
        used = list(state.get('tools_used', []))
        if hasattr(last, 'tool_calls'):
            for tc in last.tool_calls:
                n = tc.get('name', 'unknown')
                if n not in used:
                    used.append(n)
        res = tool_node.invoke(state)
        res['tools_used'] = used
        return res

    def should_continue(state):
        last = state['messages'][-1]
        # Only continue to tools if the LAST message is an AIMessage with tool_calls
        # AND we haven't already run tools for this turn (prevents loops)
        if not hasattr(last, 'tool_calls'):
            return END
        if not last.tool_calls:
            return END
        # Count how many AIMessages with tool_calls exist — if too many, stop
        tool_call_rounds = sum(
            1 for m in state['messages']
            if hasattr(m, 'tool_calls') and m.tool_calls
        )
        if tool_call_rounds > 3:  # max 3 tool rounds then force answer
            return END
        return 'tools' 

    g = StateGraph(State)
    g.add_node('agent', call_model)
    g.add_node('tools', run_tools)
    g.set_entry_point('agent')
    g.add_conditional_edges('agent', should_continue, {'tools':'tools', END:END})
    g.add_edge('tools','agent')
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Auto-initialise from env vars on first run
# ══════════════════════════════════════════════════════════════════════════════
MODEL_META = {
    'gemini': {'label':'✨ Gemini 2.5 Flash', 'env':'GOOGLE_API_KEY'},
    'openai': {'label':'⚡ GPT-4o-mini',      'env':'OPENAI_API_KEY'},
    'groq':   {'label':'🦙 Llama-3.3-70b',    'env':'GROQ_API_KEY'},
}

if not st.session_state.initialized:
    available = {k: v for k, v in MODEL_META.items() if os.environ.get(v['env'])}
    if available:
        with st.spinner("⏳ Loading MediBot — building knowledge base…"):
            try:
                faiss_ret, bm25_ret  = load_kb()
                ensemble             = make_ensemble(faiss_ret, bm25_ret)
                tools                = make_tools(faiss_ret, bm25_ret, ensemble)
                first                = list(available.keys())[0]
                agent                = build_agent(first, tools)

                st.session_state.available_models = available
                st.session_state.current_model    = first
                st.session_state.tools            = tools
                st.session_state.agent            = agent
                st.session_state.initialized      = True
                st.rerun()
            except Exception as e:
                st.error(f"Startup error: {e}")
                st.stop()
    else:
        st.error("⚠️ No API keys found. Set GOOGLE_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY as environment variables (or in a .env file).")
        st.stop()

# Rebuild agent on model switch
if st.session_state.agent is None and st.session_state.initialized:
    with st.spinner("🔄 Switching model…"):
        st.session_state.agent = build_agent(
            st.session_state.current_model, st.session_state.tools)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — model picker + stats only
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 Model")
    model_display = {v['label']: k for k, v in st.session_state.available_models.items()}
    chosen_label  = st.selectbox("LLM", list(model_display.keys()),
                                 label_visibility="collapsed")
    chosen_key = model_display[chosen_label]
    if chosen_key != st.session_state.current_model:
        st.session_state.current_model = chosen_key
        st.session_state.agent = None
        st.rerun()

    st.divider()
    st.markdown("## 📊 Session")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric-box'><div class='val'>{st.session_state.query_count}</div><div class='lbl'>Queries</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='val'>{len(st.session_state.chat_history)}</div><div class='lbl'>Messages</div></div>", unsafe_allow_html=True)

    if st.session_state.tool_counts:
        st.markdown("<p style='font-size:0.68rem;color:#4a7a96;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;'>Tool usage</p>", unsafe_allow_html=True)
        for tool, cnt in sorted(st.session_state.tool_counts.items(), key=lambda x: -x[1]):
            st.markdown(f"<p style='font-size:0.73rem;color:#4a7a96;margin:2px 0;font-family:DM Mono,monospace;'>{TOOL_LABELS.get(tool,tool)} ×{cnt}</p>", unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conv_history = []
        st.session_state.query_count  = 0
        st.session_state.tool_counts  = defaultdict(int)
        st.rerun()

    st.divider()
    st.markdown("""<div style='font-size:0.69rem;color:#2d4a5a;line-height:1.7;'>
    <b style='color:#3a6a84;'>6 Tools active</b><br>
    🔀 Hybrid RAG (FAISS+BM25)<br>
    🧠 FAISS Semantic Search<br>
    🔑 BM25 Keyword Search<br>
    🌐 Live Wikipedia<br>
    🩺 Symptom Checker<br>
    💊 Drug Information DB
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main chat area
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class='medibot-header'>
    <h1>🏥 MediBot</h1>
    <p>Hybrid RAG · 6 Tools · MultiModel · Tool Attribution</p>
</div>""", unsafe_allow_html=True)

# Render history
for msg in st.session_state.chat_history:
    if msg['role'] == 'user':
        st.markdown(f"<div class='chat-user'>🧑&nbsp; {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>{msg['content']}</div>", unsafe_allow_html=True)
        tools_used = msg.get('tools_used', [])
        if tools_used:
            badges = "".join(f"<span class='tool-badge'>{TOOL_LABELS.get(t,t)}</span>" for t in tools_used)
            rows   = "".join(f"""<div class='tool-row'>
                <span class='chk'>✓</span>
                <div><strong style='color:#81d4fa;'>{TOOL_LABELS.get(t,t)}</strong>
                &nbsp;—&nbsp;<em>{TOOL_WHY.get(t,'')}</em></div>
            </div>""" for t in tools_used)
            st.markdown(f"""<div class='tool-box'>
                <div class='tb-header'>🔧 Tools invoked</div>
                <div style='margin-bottom:6px;'>{badges}</div>{rows}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='no-tool'>ℹ️ No RAG tools invoked — answered from model knowledge</div>", unsafe_allow_html=True)

# Input row — wrapped in a form so Enter key and Send button both work exactly once
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([9, 1])
    with col_in:
        user_input = st.text_input("q", placeholder="Ask a medical question…",
                                   label_visibility="collapsed")
    with col_btn:
        send = st.form_submit_button("Send", use_container_width=True)

# Process — only fires on explicit Send click, never on rerun
if send and str(user_input).strip():
    query = str(user_input).strip()
    st.session_state.chat_history.append({'role':'user','content':query})

    with st.spinner("🔍 Thinking…"):
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state.conv_history.append(HumanMessage(content=query))
        try:
            trimmed_history = st.session_state.conv_history[-10:]
            res        = st.session_state.agent.invoke(
                {'messages': trimmed_history, 'tools_used': []},
                config={'recursion_limit': 10})
            ai_msgs    = [m for m in res['messages'] if isinstance(m, AIMessage)]
            tools_used = res.get('tools_used', [])
            reply      = ai_msgs[-1].content if ai_msgs else "No response."
            if isinstance(reply, list):
                reply = " ".join(p.get('text','') for p in reply if isinstance(p, dict))

            st.session_state.conv_history.append(ai_msgs[-1] if ai_msgs else AIMessage(content=reply))
            st.session_state.query_count += 1
            for t in tools_used:
                st.session_state.tool_counts[t] += 1
            st.session_state.chat_history.append(
                {'role':'bot','content':reply,'tools_used':tools_used})
        except Exception as e:
            st.session_state.chat_history.append(
                {'role':'bot','content':f"⚠️ Error: {e}",'tools_used':[]})
    st.rerun()

st.markdown("""<div class='disclaimer'>
⚠️ <b>Medical Disclaimer:</b> MediBot is for educational purposes only and does not constitute
medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.
</div>""", unsafe_allow_html=True)
