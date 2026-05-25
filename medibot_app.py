import os
import streamlit as st
from collections import defaultdict
from typing import List, Annotated
from typing_extensions import TypedDict
from time import sleep

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] * {
    color: #c8d6e5 !important;
}

/* ── Header ── */
.medibot-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 1.5rem;
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
    color: #6b8299;
    font-size: 0.8rem;
    margin: 0.25rem 0 0;
    font-family: 'DM Mono', monospace;
}

/* ── Chat bubbles ── */
.chat-user {
    background: #1a2535;
    border: 1px solid #2a3f55;
    border-radius: 14px 14px 4px 14px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #c8d6e5;
    font-size: 0.92rem;
    line-height: 1.55;
}
.chat-bot {
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 14px 14px 14px 4px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #dde6f0;
    font-size: 0.92rem;
    line-height: 1.6;
}
.chat-bot strong {
    color: #4fc3f7;
}

/* ── Tool badge ── */
.tool-used-box {
    background: #0d1f2d;
    border: 1px solid #1a3347;
    border-left: 3px solid #00bcd4;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-top: 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
}
.tool-used-box .tool-header {
    color: #00bcd4;
    font-weight: 500;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
}
.tool-badge {
    display: inline-block;
    background: #0f2233;
    border: 1px solid #1a4060;
    color: #81d4fa;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
}
.tool-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin: 4px 0;
    color: #7fa8c0;
    font-size: 0.74rem;
}
.tool-row .check {
    color: #26c6da;
    flex-shrink: 0;
    margin-top: 1px;
}
.tool-row .why {
    color: #4a7a96;
    font-style: italic;
}

/* ── Input box ── */
.stTextInput input {
    background: #161b27 !important;
    border: 1px solid #2a3f55 !important;
    color: #c8d6e5 !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,0.12) !important;
}

/* ── Buttons ── */
.stButton button {
    background: #1565c0 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: background 0.2s !important;
}
.stButton button:hover {
    background: #1976d2 !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #161b27 !important;
    border: 1px solid #2a3f55 !important;
    color: #c8d6e5 !important;
    border-radius: 8px !important;
}

/* ── Status / disclaimer ── */
.disclaimer {
    background: #1a1209;
    border: 1px solid #3d2800;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.75rem;
    color: #a0825a;
    margin-top: 0.5rem;
}

/* ── Loading text ── */
.thinking {
    color: #4fc3f7;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    padding: 0.5rem;
    animation: pulse 1.2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Metrics in sidebar ── */
.metric-box {
    background: #0d1520;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.5rem;
    font-weight: 600;
    color: #4fc3f7;
    font-family: 'DM Mono', monospace;
}
.metric-box .lbl {
    font-size: 0.7rem;
    color: #4a7a96;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Hide streamlit default header decorations */
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

TOOL_DESCRIPTIONS = {
    'hybrid_rag_retriever': 'FAISS + BM25 via Reciprocal Rank Fusion — best recall for medical questions',
    'faiss_rag_retriever':  'Dense vector similarity search in local Wikipedia knowledge base',
    'bm25_rag_retriever':   'TF-IDF keyword matching — great for exact drug/condition names',
    'wikipedia_live_tool':  'Live Wikipedia API — used for rare or recent medical topics',
    'symptom_checker':      'Rule-based symptom mapper with urgency levels',
    'drug_information':     'Local drug DB covering dosage, side effects, contraindications',
}

TOOL_WHY = {
    'hybrid_rag_retriever': 'Primary retriever — combines semantic + keyword for broadest coverage',
    'faiss_rag_retriever':  'Conceptual question detected — semantic search gives best context',
    'bm25_rag_retriever':   'Exact medical term / drug name found — keyword match preferred',
    'wikipedia_live_tool':  'Topic not found in local KB — expanded to live Wikipedia',
    'symptom_checker':      'Symptoms were mentioned — urgency assessment triggered',
    'drug_information':     'Drug name detected — fetching dosage, interactions & side effects',
}

# ── Initialise session state ───────────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized     = False
    st.session_state.chat_history    = []   # list of {role, content, tools_used}
    st.session_state.conv_history    = []   # LangChain message objects
    st.session_state.current_model   = None
    st.session_state.available_models= {}
    st.session_state.agent           = None
    st.session_state.init_error      = None
    st.session_state.query_count     = 0
    st.session_state.tool_counts     = defaultdict(int)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — API keys & model selection
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔑 API Keys")
    st.markdown("<p style='font-size:0.75rem;color:#4a7a96;'>Enter at least one key to start</p>", unsafe_allow_html=True)

    gemini_key = st.text_input("Google Gemini key", type="password", placeholder="AIza...")
    openai_key = st.text_input("OpenAI key (optional)", type="password", placeholder="sk-...")
    groq_key   = st.text_input("Groq key (optional)", type="password", placeholder="gsk_...")

    init_btn = st.button("🚀 Initialise MediBot", use_container_width=True)

    st.divider()
    st.markdown("## 🤖 Active Model")
    if st.session_state.available_models:
        model_display = {v['label']: k for k, v in st.session_state.available_models.items()}
        chosen_label = st.selectbox(
            "LLM",
            options=list(model_display.keys()),
            index=0,
            label_visibility="collapsed"
        )
        chosen_key = model_display[chosen_label]
        if chosen_key != st.session_state.current_model and st.session_state.initialized:
            st.session_state.current_model = chosen_key
            st.session_state.agent = None  # trigger rebuild on next query
    else:
        st.markdown("<p style='color:#4a7a96;font-size:0.8rem;'>No models yet — enter keys above</p>", unsafe_allow_html=True)

    st.divider()

    # ── Stats ──
    if st.session_state.initialized:
        st.markdown("## 📊 Session Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class='metric-box'>
                <div class='val'>{st.session_state.query_count}</div>
                <div class='lbl'>Queries</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-box'>
                <div class='val'>{len(st.session_state.chat_history)}</div>
                <div class='lbl'>Messages</div></div>""", unsafe_allow_html=True)

        if st.session_state.tool_counts:
            st.markdown("<p style='font-size:0.7rem;color:#4a7a96;text-transform:uppercase;letter-spacing:0.06em;'>Tool usage</p>", unsafe_allow_html=True)
            for tool, count in sorted(st.session_state.tool_counts.items(), key=lambda x: -x[1]):
                label = TOOL_LABELS.get(tool, tool)
                st.markdown(f"<p style='font-size:0.75rem;color:#7fa8c0;margin:2px 0;font-family:DM Mono,monospace;'>{label} ×{count}</p>", unsafe_allow_html=True)

        st.divider()
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conv_history = []
            st.session_state.query_count  = 0
            st.session_state.tool_counts  = defaultdict(int)
            st.rerun()

    st.divider()
    st.markdown("""<div style='font-size:0.7rem;color:#3a5a70;'>
    <b style='color:#4a7a96;'>Tools available:</b><br>
    🔀 Hybrid RAG (FAISS+BM25)<br>
    🧠 FAISS Semantic Search<br>
    🔑 BM25 Keyword Search<br>
    🌐 Live Wikipedia<br>
    🩺 Symptom Checker<br>
    💊 Drug Information DB
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Initialisation logic
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Load Wikipedia medical KB — cached so it only runs once."""
    import wikipediaapi
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever

    wiki = wikipediaapi.Wikipedia(language='en', user_agent='MediBot-Streamlit/3.0')

    MEDICAL_TOPICS = [
        'Diabetes mellitus', 'Hypertension', 'COVID-19', 'Cancer', 'Asthma',
        'Tuberculosis', 'Pneumonia', 'Influenza', 'Dengue fever', 'Malaria',
        'Heart disease', 'Stroke', 'Kidney failure', 'Liver disease', 'Obesity',
        'Anemia', 'Arthritis', 'Migraine', 'Epilepsy',
        "Parkinson's disease", "Alzheimer's disease",
        'Human body', 'Brain', 'Heart', 'Lung', 'Kidney', 'Liver',
        'Digestive system', 'Nervous system', 'Immune system',
        'Respiratory system', 'Circulatory system', 'Endocrine system',
        'Fever', 'Cough', 'Chest pain', 'Headache', 'Fatigue',
        'Shortness of breath', 'Abdominal pain', 'Diarrhea', 'Weight loss',
        'Antibiotic', 'Vaccination', 'Insulin', 'Chemotherapy',
        'Paracetamol', 'Ibuprofen', 'Pain management', 'Metformin',
        'Aspirin', 'Amoxicillin',
        'Nutrition', 'Vitamin', 'Mental health', 'Exercise', 'Public health',
    ]

    all_docs = []
    for topic in MEDICAL_TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                all_docs.append(Document(
                    page_content=page.text[:5000],
                    metadata={'title': page.title, 'source': page.fullurl}
                ))
            sleep(0.2)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore   = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_ret     = vectorstore.as_retriever(search_kwargs={'k': 4})
    bm25_ret      = BM25Retriever.from_documents(chunks)
    bm25_ret.k    = 4

    return faiss_ret, bm25_ret, chunks


def build_ensemble(faiss_ret, bm25_ret):
    from collections import defaultdict
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from pydantic import ConfigDict

    def reciprocal_rank_fusion(results, k=60):
        fused_scores = defaultdict(float)
        unique_docs  = {}
        for result_list in results:
            for rank, doc in enumerate(result_list):
                doc_id = (doc.page_content, doc.metadata.get('source'))
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
                fused_scores[doc_id] += 1 / (k + rank)
        sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
        return [unique_docs[i] for i in sorted_ids]

    class CustomEnsembleRetriever(BaseRetriever):
        retrievers: list
        weights:    list = None
        k:          int  = 60
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str) -> List[Document]:
            return reciprocal_rank_fusion([r.invoke(query) for r in self.retrievers], self.k)

        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            raise NotImplementedError

    return CustomEnsembleRetriever(retrievers=[faiss_ret, bm25_ret], weights=[0.6, 0.4])


def build_agent(model_name, llm, tools):
    from typing import Annotated, List
    from typing_extensions import TypedDict
    from langchain_core.messages import SystemMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    SYSTEM_PROMPT = """You are MediBot, an expert Medical AI Assistant with a Hybrid RAG system.

## Tools Available:
1. hybrid_rag_retriever — PRIMARY: FAISS semantic + BM25 keyword. Use FIRST for all medical questions.
2. faiss_rag_retriever — Dense semantic search. Use for conceptual questions.
3. bm25_rag_retriever — Keyword search. Use for exact medical terms/drug names.
4. wikipedia_live_tool — Live Wikipedia. Use as fallback for rare/recent conditions.
5. symptom_checker — ALWAYS call when the user mentions any symptoms.
6. drug_information — ALWAYS call when a specific drug/medication is named.

## Rules:
- Start every medical question with hybrid_rag_retriever.
- Call symptom_checker immediately if symptoms are described.
- Call drug_information when a drug is named.
- Use wikipedia_live_tool for extra depth or unknown topics.
- Structure answers with clear headings and bullet points.
- Always end with a medical disclaimer.

*Medical Disclaimer: For educational purposes only. Always consult a qualified healthcare professional.*"""

    class AgentState(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    llm_with_tools = llm.bind_tools(tools)
    tool_node      = ToolNode(tools)

    def call_model(state: AgentState):
        msgs = list(state['messages'])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {'messages': [response]}

    def run_tools(state: AgentState):
        last = state['messages'][-1]
        used = list(state.get('tools_used', []))
        if hasattr(last, 'tool_calls') and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get('name', 'unknown')
                if name not in used:
                    used.append(name)
        result = tool_node.invoke(state)
        result['tools_used'] = used
        return result

    def should_continue(state: AgentState):
        last = state['messages'][-1]
        return 'tools' if (hasattr(last, 'tool_calls') and last.tool_calls) else END

    graph = StateGraph(AgentState)
    graph.add_node('agent', call_model)
    graph.add_node('tools', run_tools)
    graph.set_entry_point('agent')
    graph.add_conditional_edges('agent', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'agent')
    return graph.compile()


def make_tools(faiss_ret, bm25_ret, ensemble):
    from langchain_core.tools import tool
    import wikipediaapi

    @tool
    def faiss_rag_retriever(query: str) -> str:
        """Semantic/dense vector search in local FAISS medical knowledge base. Best for conceptual medical questions about diseases, treatments, and anatomy."""
        return str(faiss_ret.invoke(query))

    @tool
    def bm25_rag_retriever(query: str) -> str:
        """Keyword BM25 search in local medical knowledge base. Best when exact medical terms, drug names, or specific condition names are mentioned."""
        return str(bm25_ret.invoke(query))

    @tool
    def hybrid_rag_retriever(query: str) -> str:
        """Hybrid RAG: FAISS semantic + BM25 keyword via Reciprocal Rank Fusion. Use this FIRST as the PRIMARY retriever for all medical questions."""
        return str(ensemble.invoke(query))

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base. Use as fallback for rare or recent conditions."""
        wiki_client = wikipediaapi.Wikipedia(language='en', user_agent='MediBot-Streamlit/3.0')
        try:
            page = wiki_client.page(query)
            if page.exists():
                return f"[Source: {page.fullurl}]\n\n{page.text[:3000]}"
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f'Wikipedia error: {e}'

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Analyze comma-separated symptoms and return possible conditions with urgency levels. Always use when user mentions symptoms."""
        symptom_map = {
            'fever':               {'conditions': ['Influenza', 'COVID-19', 'Malaria', 'Typhoid', 'Dengue'], 'urgency': 'Medium'},
            'cough':               {'conditions': ['Asthma', 'COVID-19', 'Tuberculosis', 'Bronchitis'], 'urgency': 'Low-Medium'},
            'chest pain':          {'conditions': ['Heart disease', 'Angina', 'Pneumonia', 'GERD'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'headache':            {'conditions': ['Migraine', 'Hypertension', 'Tension headache', 'Meningitis'], 'urgency': 'Low-Medium'},
            'fatigue':             {'conditions': ['Anemia', 'Diabetes', 'Hypothyroidism', 'Depression'], 'urgency': 'Low'},
            'shortness of breath': {'conditions': ['Asthma', 'Heart failure', 'COVID-19', 'Pulmonary embolism'], 'urgency': '🚨 HIGH'},
            'frequent urination':  {'conditions': ['Diabetes mellitus', 'UTI', 'Prostate issues'], 'urgency': 'Medium'},
            'weight loss':         {'conditions': ['Diabetes', 'Cancer', 'Tuberculosis', 'Hyperthyroidism'], 'urgency': 'Medium-High'},
            'joint pain':          {'conditions': ['Arthritis', 'Gout', 'Lupus', 'Fibromyalgia'], 'urgency': 'Low-Medium'},
            'rash':                {'conditions': ['Eczema', 'Psoriasis', 'Allergic reaction', 'Lupus'], 'urgency': 'Low'},
            'nausea':              {'conditions': ['Gastritis', 'Food poisoning', 'Pregnancy', 'Migraine'], 'urgency': 'Low-Medium'},
            'dizziness':           {'conditions': ['Hypertension', 'Anemia', 'Inner ear disorder', 'Stroke'], 'urgency': 'Medium'},
            'vomiting':            {'conditions': ['Gastritis', 'Food poisoning', 'Appendicitis', 'Migraine'], 'urgency': 'Medium'},
            'abdominal pain':      {'conditions': ['Gastritis', 'Appendicitis', 'IBS', 'Kidney stones'], 'urgency': 'Medium-High'},
            'back pain':           {'conditions': ['Muscle strain', 'Herniated disc', 'Kidney stones', 'Sciatica'], 'urgency': 'Low-Medium'},
            'swelling':            {'conditions': ['Heart failure', 'Kidney disease', 'DVT', 'Lymphedema'], 'urgency': 'Medium'},
            'blurred vision':      {'conditions': ['Diabetes', 'Hypertension', 'Glaucoma', 'Stroke'], 'urgency': '🚨 HIGH'},
            'confusion':           {'conditions': ['Stroke', 'Hypoglycemia', 'Dementia', 'Encephalitis'], 'urgency': '🚨 HIGH'},
        }
        entered = [s.strip().lower() for s in symptoms.split(',')]
        result  = {}
        for sym in entered:
            for key, data in symptom_map.items():
                if key in sym:
                    result[key] = data
        if not result:
            return 'No matches found. Describe symptoms more clearly or consult a healthcare professional.'
        lines = ['Symptom Analysis (Educational Only):\n']
        for sym, data in result.items():
            lines.append(f'  • {sym.capitalize()}')
            lines.append(f'    Possible: {", ".join(data["conditions"])}')
            lines.append(f'    Urgency: {data["urgency"]}')
        lines.append('\nEducational only. Always consult a licensed physician.')
        return '\n'.join(lines)

    @tool
    def drug_information(drug_name: str) -> str:
        """Get drug details: class, uses, dosage, side effects, contraindications, interactions. Use when user asks about a specific medication."""
        drug_db = {
            'paracetamol': {'class': 'Analgesic/Antipyretic', 'uses': 'Fever, mild-moderate pain', 'dosage': '500-1000mg every 4-6h, max 4g/day', 'side_effects': 'Rare at normal doses; overdose causes liver damage', 'contraindications': 'Severe liver disease', 'interactions': 'Warfarin (high doses)'},
            'ibuprofen':   {'class': 'NSAID', 'uses': 'Pain, fever, inflammation', 'dosage': '200-400mg every 4-6h', 'side_effects': 'GI upset, ulcers, increased BP', 'contraindications': 'Peptic ulcer, kidney disease', 'interactions': 'Aspirin, warfarin, ACE inhibitors'},
            'amoxicillin': {'class': 'Penicillin Antibiotic', 'uses': 'Bacterial infections: respiratory, UTI, ear', 'dosage': '250-500mg every 8h', 'side_effects': 'Diarrhea, nausea, rash', 'contraindications': 'Penicillin allergy', 'interactions': 'Warfarin, oral contraceptives'},
            'metformin':   {'class': 'Biguanide Antidiabetic', 'uses': 'Type 2 diabetes (first-line)', 'dosage': 'Start 500mg twice daily, max 2550mg/day', 'side_effects': 'GI upset (temporary); rare: lactic acidosis', 'contraindications': 'Kidney failure, liver disease', 'interactions': 'Alcohol, iodinated contrast'},
            'aspirin':     {'class': 'Salicylate NSAID / Antiplatelet', 'uses': 'Pain, fever; low-dose cardiovascular protection', 'dosage': 'Pain: 325-650mg. Cardio: 75-100mg/day', 'side_effects': 'GI irritation, tinnitus, bleeding', 'contraindications': 'Children <16, peptic ulcer', 'interactions': 'Warfarin, ibuprofen, SSRIs'},
            'insulin':     {'class': 'Hormone / Injectable Antidiabetic', 'uses': 'Type 1 DM (essential), Type 2 DM (refractory)', 'dosage': 'Individualized — medical supervision required', 'side_effects': 'Hypoglycemia, weight gain', 'contraindications': 'Hypoglycemia', 'interactions': 'Beta-blockers, corticosteroids, alcohol'},
        }
        key     = drug_name.lower().strip()
        matched = next((k for k in drug_db if k in key or key in k), None)
        if not matched:
            return f"No local data for '{drug_name}'. Try hybrid_rag_retriever or wikipedia_live_tool."
        i = drug_db[matched]
        return (
            f"{matched.capitalize()}\n"
            f"  Class: {i['class']}\n  Uses: {i['uses']}\n"
            f"  Dosage: {i['dosage']}\n  Side Effects: {i['side_effects']}\n"
            f"  Contraindications: {i['contraindications']}\n"
            f"  Key Interactions: {i['interactions']}\n\n"
            f"Follow your doctor's / pharmacist's instructions."
        )

    return [hybrid_rag_retriever, faiss_rag_retriever, bm25_rag_retriever,
            wikipedia_live_tool, symptom_checker, drug_information]


def get_llm(model_name, keys):
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:    from langchain_openai import ChatOpenAI
    except: ChatOpenAI = None
    try:    from langchain_groq import ChatGroq
    except: ChatGroq = None

    if model_name == 'gemini':
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash', google_api_key=keys['gemini'],
            temperature=0.2, max_output_tokens=1024,
            convert_system_message_to_human=True
        )
    elif model_name == 'openai' and ChatOpenAI:
        return ChatOpenAI(model='gpt-4o-mini', api_key=keys['openai'],
                          temperature=0.2, max_tokens=1024)
    elif model_name == 'groq' and ChatGroq:
        return ChatGroq(model='llama-3.3-70b-versatile', api_key=keys['groq'],
                        temperature=0.2, max_tokens=1024)
    raise ValueError(f"Model {model_name} unavailable")


MODEL_CONFIGS = {
    'gemini': {'label': '✨ Gemini 2.5 Flash', 'key': 'gemini'},
    'openai': {'label': '⚡ GPT-4o-mini',      'key': 'openai'},
    'groq':   {'label': '🦙 Llama-3.3-70b',    'key': 'groq'},
}

# ══════════════════════════════════════════════════════════════════════════════
# Handle init button
# ══════════════════════════════════════════════════════════════════════════════
if init_btn:
    keys = {}
    if gemini_key: keys['gemini'] = gemini_key; os.environ['GOOGLE_API_KEY'] = gemini_key
    if openai_key: keys['openai'] = openai_key; os.environ['OPENAI_API_KEY'] = openai_key
    if groq_key:   keys['groq']   = groq_key;   os.environ['GROQ_API_KEY']   = groq_key

    if not keys:
        st.sidebar.error("⚠️ Enter at least one API key")
    else:
        with st.spinner("⏳ Loading medical knowledge base (first run takes ~2 min)…"):
            try:
                faiss_ret, bm25_ret, _ = load_knowledge_base()
                ensemble = build_ensemble(faiss_ret, bm25_ret)
                tools    = make_tools(faiss_ret, bm25_ret, ensemble)

                available = {k: v for k, v in MODEL_CONFIGS.items() if k in keys}
                first_model = list(available.keys())[0]
                llm   = get_llm(first_model, keys)
                agent = build_agent(first_model, llm, tools)

                st.session_state.faiss_ret       = faiss_ret
                st.session_state.bm25_ret        = bm25_ret
                st.session_state.ensemble        = ensemble
                st.session_state.tools           = tools
                st.session_state.keys            = keys
                st.session_state.available_models= available
                st.session_state.current_model   = first_model
                st.session_state.agent           = agent
                st.session_state.initialized     = True
                st.session_state.init_error      = None
                st.rerun()
            except Exception as e:
                st.session_state.init_error = str(e)
                st.sidebar.error(f"Init failed: {e}")

# Rebuild agent if model switched
if (st.session_state.initialized and st.session_state.agent is None):
    with st.spinner("🔄 Switching model…"):
        try:
            llm   = get_llm(st.session_state.current_model, st.session_state.keys)
            agent = build_agent(st.session_state.current_model, llm, st.session_state.tools)
            st.session_state.agent = agent
        except Exception as e:
            st.error(f"Model switch failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Main chat UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class='medibot-header'>
    <h1>🏥 MediBot</h1>
    <p>Medical RAG Assistant · Hybrid Retrieval · Tool Attribution · MultiModel</p>
</div>""", unsafe_allow_html=True)

# Welcome / not-initialised state
if not st.session_state.initialized:
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;color:#4a7a96;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>🏥</div>
        <h3 style='color:#6b8299;font-family:Lora,serif;'>Enter your API key in the sidebar to begin</h3>
        <p style='font-size:0.85rem;max-width:480px;margin:0.5rem auto;'>
            MediBot uses a Hybrid RAG pipeline (FAISS + BM25) over a Wikipedia medical knowledge base.
            Every response shows which tools were used and why.
        </p>
        <div style='margin-top:2rem;display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;'>
            <div style='background:#0d1520;border:1px solid #1e2d40;border-radius:10px;padding:1rem;width:160px;'>
                <div style='font-size:1.5rem;'>🔀</div>
                <div style='color:#4fc3f7;font-size:0.8rem;margin-top:0.4rem;'>Hybrid RAG</div>
                <div style='color:#3a5a70;font-size:0.7rem;'>FAISS + BM25</div>
            </div>
            <div style='background:#0d1520;border:1px solid #1e2d40;border-radius:10px;padding:1rem;width:160px;'>
                <div style='font-size:1.5rem;'>🩺</div>
                <div style='color:#4fc3f7;font-size:0.8rem;margin-top:0.4rem;'>Symptom Check</div>
                <div style='color:#3a5a70;font-size:0.7rem;'>Urgency levels</div>
            </div>
            <div style='background:#0d1520;border:1px solid #1e2d40;border-radius:10px;padding:1rem;width:160px;'>
                <div style='font-size:1.5rem;'>💊</div>
                <div style='color:#4fc3f7;font-size:0.8rem;margin-top:0.4rem;'>Drug Info</div>
                <div style='color:#3a5a70;font-size:0.7rem;'>Dosage & interactions</div>
            </div>
            <div style='background:#0d1520;border:1px solid #1e2d40;border-radius:10px;padding:1rem;width:160px;'>
                <div style='font-size:1.5rem;'>🌐</div>
                <div style='color:#4fc3f7;font-size:0.8rem;margin-top:0.4rem;'>Live Wikipedia</div>
                <div style='color:#3a5a70;font-size:0.7rem;'>Real-time fallback</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg['role'] == 'user':
        st.markdown(f"<div class='chat-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        # Bot reply
        st.markdown(f"<div class='chat-bot'>{msg['content']}</div>", unsafe_allow_html=True)

        # Tool attribution panel
        tools_used = msg.get('tools_used', [])
        if tools_used:
            badges = "".join(f"<span class='tool-badge'>{TOOL_LABELS.get(t, t)}</span>" for t in tools_used)
            rows   = ""
            for t in tools_used:
                rows += f"""<div class='tool-row'>
                    <span class='check'>✓</span>
                    <div>
                        <strong style='color:#81d4fa;'>{TOOL_LABELS.get(t, t)}</strong><br>
                        <span class='why'>{TOOL_WHY.get(t, '')}</span>
                    </div>
                </div>"""
            st.markdown(f"""
            <div class='tool-used-box'>
                <div class='tool-header'>🔧 Tools invoked</div>
                <div style='margin-bottom:6px;'>{badges}</div>
                {rows}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='tool-used-box'><span style='color:#3a5a70;font-size:0.75rem;'>ℹ️ Answered from model knowledge (no RAG tools invoked)</span></div>", unsafe_allow_html=True)

# ── Input area ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Suggestion chips
if not st.session_state.chat_history:
    st.markdown("<p style='color:#4a7a96;font-size:0.78rem;font-family:DM Mono,monospace;'>Try asking:</p>", unsafe_allow_html=True)
    suggestions = [
        "What is diabetes and how is it treated?",
        "I have fever, cough and fatigue",
        "What are the side effects of ibuprofen?",
        "Tell me about Kawasaki disease",
    ]
    cols = st.columns(len(suggestions))
    for col, sugg in zip(cols, suggestions):
        with col:
            if st.button(sugg, key=f"sugg_{sugg[:20]}", use_container_width=True):
                st.session_state._pending_query = sugg
                st.rerun()

col_input, col_send = st.columns([9, 1])
with col_input:
    user_input = st.text_input(
        "message",
        key="user_input",
        placeholder="Ask a medical question…",
        label_visibility="collapsed",
    )
with col_send:
    send_btn = st.button("Send", use_container_width=True)

# Pick up suggestion click
if hasattr(st.session_state, '_pending_query'):
    user_input = st.session_state._pending_query
    del st.session_state._pending_query
    send_btn = True

# ── Process query ──────────────────────────────────────────────────────────────
if (send_btn or user_input) and user_input and user_input.strip():
    query = user_input.strip()

    # Add user message
    st.session_state.chat_history.append({'role': 'user', 'content': query})

    # Show thinking spinner
    with st.spinner("🔍 MediBot is thinking…"):
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state.conv_history.append(HumanMessage(content=query))
        try:
            result = st.session_state.agent.invoke(
                {'messages': st.session_state.conv_history, 'tools_used': []},
                config={'recursion_limit': 25},
            )
            ai_msgs    = [m for m in result['messages'] if isinstance(m, AIMessage)]
            tools_used = result.get('tools_used', [])
            reply      = ai_msgs[-1].content if ai_msgs else "No response received."
            if isinstance(reply, list):
                reply = " ".join(p.get('text', '') for p in reply if isinstance(p, dict))

            st.session_state.conv_history.append(ai_msgs[-1] if ai_msgs else AIMessage(content=reply))

            # Update stats
            st.session_state.query_count += 1
            for t in tools_used:
                st.session_state.tool_counts[t] += 1

            st.session_state.chat_history.append({
                'role':       'bot',
                'content':    reply,
                'tools_used': tools_used,
            })

        except Exception as e:
            st.session_state.chat_history.append({
                'role':       'bot',
                'content':    f"⚠️ Error: {e}",
                'tools_used': [],
            })

    st.rerun()

# Disclaimer footer
st.markdown("""<div class='disclaimer'>
⚠️ <strong>Medical Disclaimer:</strong> MediBot is for educational purposes only.
It does not constitute medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional for medical decisions.
</div>""", unsafe_allow_html=True)
