import os
import streamlit as st
from collections import defaultdict
from typing import Annotated, List
from typing_extensions import TypedDict
from time import sleep

# ══════════════════════════════════════════════════════════════════
# 🔑 PASTE YOUR API KEYS HERE
# ══════════════════════════════════════════════════════════════════
os.environ["GOOGLE_API_KEY"] = "your_gemini_key_here"
os.environ["GROQ_API_KEY"]   = "your_groq_key_here"
# os.environ["OPENAI_API_KEY"] = "your_openai_key_here"  # optional
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediBot — Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lora:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0f1117;color:#e8e8e8;}
section[data-testid="stSidebar"]{background:#161b27!important;border-right:1px solid #1e2d40;}
section[data-testid="stSidebar"] *{color:#c8d6e5!important;}
.hdr{text-align:center;padding:1.3rem 0 .8rem;border-bottom:1px solid #1e2d40;margin-bottom:1.3rem;}
.hdr h1{font-family:'Lora',serif;font-size:2rem;color:#4fc3f7;margin:0;letter-spacing:-.5px;}
.hdr p{color:#4a7a96;font-size:.75rem;margin:.25rem 0 0;font-family:'DM Mono',monospace;}
.cu{background:#1a2535;border:1px solid #2a3f55;border-radius:14px 14px 4px 14px;
    padding:.85rem 1.1rem;margin:.5rem 0 .3rem 4rem;color:#c8d6e5;font-size:.92rem;line-height:1.6;}
.cb{background:#111827;border:1px solid #1e2d40;border-radius:14px 14px 14px 4px;
    padding:.9rem 1.1rem;margin:.3rem 4rem .3rem 0;color:#dde6f0;font-size:.92rem;line-height:1.65;}
.tb{background:#0b1a26;border:1px solid #14334d;border-left:3px solid #00bcd4;
    border-radius:0 8px 8px 0;padding:.65rem .9rem;margin:.25rem 4rem .8rem 0;
    font-family:'DM Mono',monospace;}
.tb-hdr{color:#00bcd4;font-size:.67rem;font-weight:500;text-transform:uppercase;
        letter-spacing:.1em;margin-bottom:.45rem;}
.badge{display:inline-block;background:#0d1f30;border:1px solid #1a4060;color:#81d4fa;
       border-radius:4px;padding:2px 8px;margin:2px 3px 4px 0;font-size:.7rem;}
.trow{display:flex;gap:8px;align-items:flex-start;margin:5px 0;font-size:.73rem;color:#5a8aaa;}
.trow .ck{color:#26c6da;flex-shrink:0;}
.trow em{color:#3a6a84;font-style:italic;}
.nt{background:#0b1a26;border:1px solid #14334d;border-left:3px solid #2d4a5a;
    border-radius:0 8px 8px 0;padding:.42rem .85rem;margin:.25rem 4rem .8rem 0;
    font-size:.72rem;color:#3a5a70;font-family:'DM Mono',monospace;}
.disc{background:#1a1209;border:1px solid #3d2800;border-radius:8px;
      padding:.55rem .85rem;font-size:.73rem;color:#9a7040;margin-top:.8rem;}
.mbox{background:#0d1520;border:1px solid #1e2d40;border-radius:8px;
      padding:.5rem .7rem;margin-bottom:.4rem;text-align:center;}
.mbox .v{font-size:1.4rem;font-weight:600;color:#4fc3f7;font-family:'DM Mono',monospace;}
.mbox .l{font-size:.67rem;color:#4a7a96;text-transform:uppercase;letter-spacing:.06em;}
.stTextInput input{background:#161b27!important;border:1px solid #2a3f55!important;
    color:#c8d6e5!important;border-radius:10px!important;font-family:'DM Sans',sans-serif!important;}
.stTextInput input:focus{border-color:#4fc3f7!important;
    box-shadow:0 0 0 2px rgba(79,195,247,.1)!important;}
.stButton button{background:#1565c0!important;color:white!important;border:none!important;
    border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important;}
.stButton button:hover{background:#1976d2!important;}
.stSelectbox>div>div{background:#161b27!important;border:1px solid #2a3f55!important;
    color:#c8d6e5!important;border-radius:8px!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Tool metadata (from notebook Step 8) ──────────────────────────────────────
TOOL_LABELS = {
    'hybrid_rag_retriever': 'Hybrid RAG (FAISS + BM25)',
    'faiss_rag_retriever':  'FAISS Semantic RAG',
    'bm25_rag_retriever':   'BM25 Keyword RAG',
    'wikipedia_live_tool':  'Live Wikipedia',
    'symptom_checker':      'Symptom Checker',
    'drug_information':     'Drug Information DB',
}
TOOL_ICONS = {
    'hybrid_rag_retriever': '🔀',
    'faiss_rag_retriever':  '🧠',
    'bm25_rag_retriever':   '🔑',
    'wikipedia_live_tool':  '🌐',
    'symptom_checker':      '🩺',
    'drug_information':     '💊',
}
TOOL_DESCRIPTIONS = {
    'hybrid_rag_retriever': 'FAISS semantic + BM25 keyword via Reciprocal Rank Fusion',
    'faiss_rag_retriever':  'Dense vector similarity — local Wikipedia KB',
    'bm25_rag_retriever':   'TF-IDF keyword matching — local Wikipedia KB',
    'wikipedia_live_tool':  'Real-time Wikipedia API lookup',
    'symptom_checker':      'Rule-based symptom-to-condition mapper with urgency levels',
    'drug_information':     'Local drug DB: dosage, side effects, contraindications, interactions',
}

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in dict(
    initialized=False, chat_history=[], conv_history=[],
    current_model=None, available_models={}, agent=None,
    tools=None, query_count=0, tool_counts=defaultdict(int)
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Base + Tools  (cached — only built once per server session)
# Exact code from notebook Steps 4, 4.5, 5, 9
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_kb_and_tools():
    import wikipediaapi
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import tool
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from pydantic import ConfigDict

    # ── Step 4: Wikipedia KB ──────────────────────────────────────────────────
    MEDICAL_TOPICS = [
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

    wiki = wikipediaapi.Wikipedia(language='en', user_agent='MediBot/2.0')
    all_docs = []
    for topic in MEDICAL_TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                all_docs.append(Document(
                    page_content=page.text[:5000],
                    metadata={'title': page.title, 'source': page.fullurl}))
            sleep(0.3)
        except Exception:
            pass

    # ── Step 4.5: Hybrid RAG ──────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks   = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})
    vectorstore     = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    bm25_retriever  = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # CustomEnsembleRetriever from notebook Step 9 (proper BaseRetriever version)
    def reciprocal_rank_fusion(results, k=60):
        fused_scores, unique_docs = defaultdict(float), {}
        for result_list in results:
            for rank, doc in enumerate(result_list):
                doc_id = (doc.page_content, doc.metadata.get('source'))
                unique_docs.setdefault(doc_id, doc)
                fused_scores[doc_id] += 1 / (k + rank)
        sorted_ids = sorted(fused_scores, key=lambda x: -fused_scores[x])
        return [unique_docs[i] for i in sorted_ids]

    class CustomEnsembleRetriever(BaseRetriever):
        retrievers: list
        weights: list = None
        k: int = 60
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str) -> List[Document]:
            return reciprocal_rank_fusion(
                [r.invoke(query) for r in self.retrievers], self.k)

        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            raise NotImplementedError

    ensemble_retriever = CustomEnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4])

    # ── Step 5: Define 6 Tools ────────────────────────────────────────────────
    @tool
    def faiss_rag_retriever(query: str) -> str:
        """Semantic/dense vector search in local FAISS medical knowledge base. Best for conceptual medical questions about diseases, treatments, and anatomy."""
        return str(faiss_retriever.invoke(query))

    @tool
    def bm25_rag_retriever(query: str) -> str:
        """Keyword BM25 search in local medical knowledge base. Best when exact medical terms, drug names, or specific condition names are mentioned."""
        return str(bm25_retriever.invoke(query))

    @tool
    def hybrid_rag_retriever(query: str) -> str:
        """Hybrid RAG: FAISS semantic + BM25 keyword via Reciprocal Rank Fusion. Use this FIRST as the PRIMARY retriever for all medical questions."""
        return str(ensemble_retriever.invoke(query))

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base. Use as fallback when local RAG is insufficient or for rare/recent conditions."""
        wiki_client = wikipediaapi.Wikipedia(language='en', user_agent='MediBot/2.0')
        try:
            page = wiki_client.page(query)
            if page.exists():
                return f"[Source: {page.fullurl}]\n\n{page.text[:3000]}"
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f'Wikipedia error: {e}'

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Analyze comma-separated symptoms, return possible conditions with urgency levels. Always use when user mentions symptoms."""
        symptom_map = {
            'fever':               {'conditions': ['Influenza','COVID-19','Malaria','Typhoid','Dengue'], 'urgency': 'Medium'},
            'cough':               {'conditions': ['Asthma','COVID-19','Tuberculosis','Bronchitis'], 'urgency': 'Low-Medium'},
            'chest pain':          {'conditions': ['Heart disease','Angina','Pneumonia','GERD'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'headache':            {'conditions': ['Migraine','Hypertension','Tension headache','Meningitis'], 'urgency': 'Low-Medium'},
            'fatigue':             {'conditions': ['Anemia','Diabetes','Hypothyroidism','Depression'], 'urgency': 'Low'},
            'shortness of breath': {'conditions': ['Asthma','Heart failure','COVID-19','Pulmonary embolism'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'frequent urination':  {'conditions': ['Diabetes mellitus','UTI','Prostate issues'], 'urgency': 'Medium'},
            'weight loss':         {'conditions': ['Diabetes','Cancer','Tuberculosis','Hyperthyroidism'], 'urgency': 'Medium-High'},
            'joint pain':          {'conditions': ['Arthritis','Gout','Lupus','Fibromyalgia'], 'urgency': 'Low-Medium'},
            'rash':                {'conditions': ['Eczema','Psoriasis','Allergic reaction','Lupus'], 'urgency': 'Low'},
            'nausea':              {'conditions': ['Gastritis','Food poisoning','Pregnancy','Migraine'], 'urgency': 'Low-Medium'},
            'dizziness':           {'conditions': ['Hypertension','Anemia','Inner ear disorder','Stroke'], 'urgency': 'Medium'},
            'vomiting':            {'conditions': ['Gastritis','Food poisoning','Appendicitis','Migraine'], 'urgency': 'Medium'},
            'abdominal pain':      {'conditions': ['Gastritis','Appendicitis','IBS','Kidney stones'], 'urgency': 'Medium-High'},
            'back pain':           {'conditions': ['Muscle strain','Herniated disc','Kidney stones','Sciatica'], 'urgency': 'Low-Medium'},
            'swelling':            {'conditions': ['Heart failure','Kidney disease','DVT','Lymphedema'], 'urgency': 'Medium'},
            'blurred vision':      {'conditions': ['Diabetes','Hypertension','Glaucoma','Stroke'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'confusion':           {'conditions': ['Stroke','Hypoglycemia','Dementia','Encephalitis'], 'urgency': '🚨 HIGH — Seek immediate care'},
            'toothache':           {'conditions': ['Dental cavity','Gum disease','Dental abscess','Cracked tooth'], 'urgency': 'Low-Medium'},
            'insomnia':            {'conditions': ['Stress/Anxiety','Depression','Sleep apnea','Caffeine'], 'urgency': 'Low'},
            'sleep':               {'conditions': ['Stress/Anxiety','Depression','Sleep apnea','Insomnia'], 'urgency': 'Low'},
            'sore throat':         {'conditions': ['Strep throat','Common cold','Tonsillitis','COVID-19'], 'urgency': 'Low'},
            'runny nose':          {'conditions': ['Common cold','Allergic rhinitis','Influenza','Sinusitis'], 'urgency': 'Low'},
        }
        entered = [s.strip().lower() for s in symptoms.split(',')]
        result = {}
        for sym in entered:
            for key, data in symptom_map.items():
                if key in sym:
                    result[key] = data
        if not result:
            return 'No matches found. Describe symptoms more clearly or consult a healthcare professional.'
        lines = ['✅ Symptom Analysis (Educational Only):\n']
        for sym, data in result.items():
            lines.append(f"  • {sym.capitalize()}")
            lines.append(f"    Possible: {', '.join(data['conditions'])}")
            lines.append(f"    Urgency: {data['urgency']}")
        lines.append('\n⚠️ Educational only. Always consult a licensed physician.')
        return '\n'.join(lines)

    @tool
    def drug_information(drug_name: str) -> str:
        """Get drug details: class, uses, dosage, side effects, contraindications, interactions. Use when user asks about a specific medication."""
        drug_db = {
            'paracetamol': {'class':'Analgesic/Antipyretic','uses':'Fever, mild-moderate pain','dosage':'500-1000mg every 4-6h, max 4g/day','side_effects':'Rare at normal doses; overdose causes liver damage','contraindications':'Severe liver disease','interactions':'Warfarin (high doses)'},
            'ibuprofen':   {'class':'NSAID','uses':'Pain, fever, inflammation','dosage':'200-400mg every 4-6h, max 1200mg/day OTC','side_effects':'GI upset, ulcers, increased BP, kidney issues','contraindications':'Peptic ulcer, kidney disease, pregnancy (3rd trimester)','interactions':'Aspirin, warfarin, ACE inhibitors'},
            'amoxicillin': {'class':'Penicillin Antibiotic','uses':'Bacterial infections: respiratory, UTI, ear, skin','dosage':'250-500mg every 8h or 875mg every 12h','side_effects':'Diarrhea, nausea, rash, yeast overgrowth','contraindications':'Penicillin allergy','interactions':'Warfarin, oral contraceptives'},
            'metformin':   {'class':'Biguanide Antidiabetic','uses':'Type 2 diabetes (first-line)','dosage':'Start 500mg twice daily, max 2550mg/day','side_effects':'GI upset (temporary); rare: lactic acidosis','contraindications':'Kidney failure (eGFR<30), liver disease','interactions':'Alcohol, iodinated contrast, cimetidine'},
            'aspirin':     {'class':'Salicylate NSAID / Antiplatelet','uses':'Pain, fever; low-dose cardiovascular protection','dosage':'Pain: 325-650mg. Cardio: 75-100mg/day','side_effects':'GI irritation, tinnitus (high dose), bleeding','contraindications':'Children <16 (Reye syndrome), peptic ulcer','interactions':'Warfarin, ibuprofen, SSRIs'},
            'insulin':     {'class':'Hormone / Injectable Antidiabetic','uses':'Type 1 DM (essential), Type 2 DM (refractory)','dosage':'Individualized — medical supervision required','side_effects':'Hypoglycemia, weight gain, injection site reactions','contraindications':'Hypoglycemia','interactions':'Beta-blockers, corticosteroids, alcohol'},
        }
        key     = drug_name.lower().strip()
        matched = next((k for k in drug_db if k in key or key in k), None)
        if not matched:
            return f"No local data for '{drug_name}'. Try hybrid_rag_retriever or wikipedia_live_tool."
        i = drug_db[matched]
        return (f"✅ {matched.capitalize()}\n"
                f"  Class: {i['class']}\n  Uses: {i['uses']}\n  Dosage: {i['dosage']}\n"
                f"  Side Effects: {i['side_effects']}\n  Contraindications: {i['contraindications']}\n"
                f"  Key Interactions: {i['interactions']}\n\n"
                f"⚠️ Follow your doctor's/pharmacist's instructions.")

    tools = [hybrid_rag_retriever, faiss_rag_retriever, bm25_rag_retriever,
             wikipedia_live_tool, symptom_checker, drug_information]
    return tools

# ══════════════════════════════════════════════════════════════════════════════
# Model configs (from notebook Step 6) — updated to working model strings
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    'gemini': {
        'label':    '✨ Gemini 2.0 Flash',
        'requires': 'GOOGLE_API_KEY',
        'factory':  lambda: __import__(
            'langchain_google_genai', fromlist=['ChatGoogleGenerativeAI']
        ).ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            google_api_key=os.environ['GOOGLE_API_KEY'],
            temperature=0.2,
            max_output_tokens=1024,
            convert_system_message_to_human=True,
        )
    },
    'openai': {
        'label':    '⚡ GPT-4o-mini',
        'requires': 'OPENAI_API_KEY',
        'factory':  lambda: __import__(
            'langchain_openai', fromlist=['ChatOpenAI']
        ).ChatOpenAI(
            model='gpt-4o-mini',
            api_key=os.environ['OPENAI_API_KEY'],
            temperature=0.2,
            max_tokens=1024,
        )
    },
    'groq': {
        'label':    '🦙 Llama-3.1-8b (Groq)',
        'requires': 'GROQ_API_KEY',
        'factory':  lambda: __import__(
            'langchain_groq', fromlist=['ChatGroq']
        ).ChatGroq(
            model='llama-3.1-8b-instant',
            api_key=os.environ['GROQ_API_KEY'],
            temperature=0.2,
            max_tokens=1024,
        )
    },
}

def get_available_models():
    return {n: c for n, c in MODEL_CONFIGS.items() if os.environ.get(c['requires'])}

# ══════════════════════════════════════════════════════════════════════════════
# Agent builder (from notebook Step 7) with loop-prevention patch
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
You are MediBot, an expert Medical AI Assistant with a Hybrid RAG system.

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
- Structure answers with headers and bullet points.
- Always end with a '🔧 Sources & Tools Used' section listing each tool invoked.
- Finish with the medical disclaimer.

*Medical Disclaimer: For educational purposes only. Always consult a qualified healthcare professional.*
"""

def build_agent(model_name: str, tools):
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import AIMessage as AI
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    class AgentState(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    cfg = MODEL_CONFIGS[model_name]
    llm = cfg['factory']()
    if llm is None:
        raise ValueError(f'Model {model_name} unavailable.')

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
        if not hasattr(last, 'tool_calls') or not last.tool_calls:
            return END
        # Max 3 tool rounds — prevents infinite loops
        rounds = sum(1 for m in state['messages']
                     if hasattr(m, 'tool_calls') and m.tool_calls)
        return 'tools' if rounds <= 3 else END

    graph = StateGraph(AgentState)
    graph.add_node('agent', call_model)
    graph.add_node('tools', run_tools)
    graph.set_entry_point('agent')
    graph.add_conditional_edges('agent', should_continue,
                                {'tools': 'tools', END: END})
    graph.add_edge('tools', 'agent')
    return graph.compile()

# ══════════════════════════════════════════════════════════════════════════════
# Auto-initialise on first load
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.initialized:
    available = get_available_models()
    if not available:
        st.error("⚠️ No API keys found. Edit lines 10-12 at the top of medibot_app.py and add your keys.")
        st.stop()
    with st.spinner("⏳ Building medical knowledge base — first run takes ~2 minutes…"):
        try:
            tools  = load_kb_and_tools()
            first  = list(available.keys())[0]
            agent  = build_agent(first, tools)
            st.session_state.available_models = available
            st.session_state.current_model    = first
            st.session_state.tools            = tools
            st.session_state.agent            = agent
            st.session_state.initialized      = True
            st.rerun()
        except Exception as e:
            st.error(f"Startup error: {e}")
            st.stop()

# Rebuild agent after model switch
if st.session_state.agent is None:
    with st.spinner("🔄 Switching model…"):
        st.session_state.agent = build_agent(
            st.session_state.current_model, st.session_state.tools)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
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
        st.markdown(f"<div class='mbox'><div class='v'>{st.session_state.query_count}</div>"
                    f"<div class='l'>Queries</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='mbox'><div class='v'>{len(st.session_state.chat_history)}</div>"
                    f"<div class='l'>Messages</div></div>", unsafe_allow_html=True)

    if st.session_state.tool_counts:
        st.markdown("<p style='font-size:.67rem;color:#4a7a96;text-transform:uppercase;"
                    "letter-spacing:.06em;margin-bottom:3px;'>Tool usage</p>",
                    unsafe_allow_html=True)
        for t, cnt in sorted(st.session_state.tool_counts.items(), key=lambda x: -x[1]):
            icon = TOOL_ICONS.get(t, '🔧')
            st.markdown(f"<p style='font-size:.72rem;color:#4a7a96;margin:2px 0;"
                        f"font-family:DM Mono,monospace;'>{icon} {TOOL_LABELS.get(t,t)} ×{cnt}</p>",
                        unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conv_history = []
        st.session_state.query_count  = 0
        st.session_state.tool_counts  = defaultdict(int)
        st.rerun()

    st.divider()
    st.markdown("""<div style='font-size:.68rem;color:#2d4a5a;line-height:1.85;'>
    <b style='color:#3a6a84;'>6 Tools active</b><br>
    🔀 Hybrid RAG (FAISS+BM25)<br>
    🧠 FAISS Semantic Search<br>
    🔑 BM25 Keyword Search<br>
    🌐 Live Wikipedia<br>
    🩺 Symptom Checker<br>
    💊 Drug Information DB
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main chat
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class='hdr'>
    <h1>🏥 MediBot</h1>
    <p>Hybrid RAG · 6 Tools · MultiModel · Tool Attribution</p>
</div>""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.chat_history:
    if msg['role'] == 'user':
        st.markdown(f"<div class='cu'>🧑&nbsp; {msg['content']}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='cb'>{msg['content']}</div>",
                    unsafe_allow_html=True)
        tools_used = msg.get('tools_used', [])
        if tools_used:
            badges = "".join(
                f"<span class='badge'>{TOOL_ICONS.get(t,'🔧')} {TOOL_LABELS.get(t,t)}</span>"
                for t in tools_used)
            rows = "".join(
                f"<div class='trow'><span class='ck'>✓</span><div>"
                f"<strong style='color:#81d4fa;'>{TOOL_ICONS.get(t,'🔧')} {TOOL_LABELS.get(t,t)}</strong>"
                f" &nbsp;—&nbsp; <em>{TOOL_DESCRIPTIONS.get(t,'')}</em></div></div>"
                for t in tools_used)
            st.markdown(
                f"<div class='tb'><div class='tb-hdr'>🔧 Tools invoked</div>"
                f"<div style='margin-bottom:5px;'>{badges}</div>{rows}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='nt'>ℹ️ No RAG tools invoked — answered from model knowledge</div>",
                unsafe_allow_html=True)

# Input — wrapped in form so Enter & Send both fire exactly once
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([9, 1])
    with col_in:
        user_input = st.text_input("q", placeholder="Ask a medical question…",
                                   label_visibility="collapsed")
    with col_btn:
        send = st.form_submit_button("Send", use_container_width=True)

# Process — only on explicit submit
if send and str(user_input).strip():
    query = str(user_input).strip()
    st.session_state.chat_history.append({'role': 'user', 'content': query})

    with st.spinner("🔍 MediBot is thinking…"):
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state.conv_history.append(HumanMessage(content=query))
        try:
            result = st.session_state.agent.invoke(
                {'messages': st.session_state.conv_history[-10:], 'tools_used': []},
                config={'recursion_limit': 10})
            ai_msgs    = [m for m in result['messages'] if isinstance(m, AIMessage)]
            tools_used = result.get('tools_used', [])
            reply      = ai_msgs[-1].content if ai_msgs else "No response received."
            if isinstance(reply, list):
                reply = " ".join(p.get('text', '') for p in reply if isinstance(p, dict))
            st.session_state.conv_history.append(
                ai_msgs[-1] if ai_msgs else AIMessage(content=reply))
            st.session_state.query_count += 1
            for t in tools_used:
                st.session_state.tool_counts[t] += 1
            st.session_state.chat_history.append(
                {'role': 'bot', 'content': reply, 'tools_used': tools_used})
        except Exception as e:
            st.session_state.chat_history.append(
                {'role': 'bot', 'content': f"⚠️ Error: {e}", 'tools_used': []})
    st.rerun()

st.markdown("""<div class='disc'>
⚠️ <b>Medical Disclaimer:</b> MediBot is for educational purposes only.
It does not constitute medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional for medical decisions.
</div>""", unsafe_allow_html=True)
