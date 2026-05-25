"""
backend.py — all RAG + LangGraph agent logic for MediBot Streamlit app.
API keys are read from environment variables (set in .env or system env).
"""

import os
from collections import defaultdict
from time import sleep
from typing import Annotated, List

from typing_extensions import TypedDict

# ── LangChain / LangGraph ─────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ConfigDict
from langchain_core.retrievers import BaseRetriever

import wikipediaapi

# ── Optional LLM imports ──────────────────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are MediBot, an expert Medical AI Assistant with a Hybrid RAG system.

## Tools Available:
1. hybrid_rag_retriever — PRIMARY: FAISS semantic + BM25 keyword. Use FIRST for all medical questions.
2. faiss_rag_retriever  — Dense semantic search. Use for conceptual questions.
3. bm25_rag_retriever   — Keyword search. Use for exact medical terms/drug names.
4. wikipedia_live_tool  — Live Wikipedia. Use as fallback for rare/recent conditions.
5. symptom_checker      — ALWAYS call when the user mentions any symptoms.
6. drug_information     — ALWAYS call when a specific drug/medication is named.

## Rules:
- Start every medical question with hybrid_rag_retriever.
- Call symptom_checker immediately if symptoms are described.
- Call drug_information when a drug is named.
- Use wikipedia_live_tool for extra depth or unknown topics.
- Structure answers with headers and bullet points.
- Finish with: *Medical Disclaimer: For educational purposes only. Always consult a qualified healthcare professional.*
"""


# ─────────────────────────────────────────────────────────────────────────────
# RRF + ENSEMBLE RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(results: list, k: int = 60) -> list:
    fused_scores: dict = defaultdict(float)
    unique_docs:  dict = {}
    for result_list in results:
        for rank, doc in enumerate(result_list):
            doc_id = (doc.page_content, doc.metadata.get("source"))
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank)
    sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
    return [unique_docs[d] for d in sorted_ids]


class CustomEnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: list = None
    k: int = 60
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = [r.invoke(query) for r in self.retrievers]
        return reciprocal_rank_fusion(results, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE + RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
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
    "Paracetamol", "Ibuprofen", "Pain management", "Metformin",
    "Aspirin", "Amoxicillin",
    # Health
    "Nutrition", "Vitamin", "Mental health", "Exercise", "Public health",
]


def _fetch_page(wiki, title: str):
    try:
        page = wiki.page(title)
        if page.exists():
            return Document(
                page_content=page.text[:5000],
                metadata={"title": page.title, "source": page.fullurl},
            )
    except Exception:
        pass
    return None


def build_rag_pipeline() -> dict:
    """Fetch Wikipedia docs, build FAISS + BM25, return pipeline dict."""
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/2.0")
    docs = []
    for topic in MEDICAL_TOPICS:
        doc = _fetch_page(wiki, topic)
        if doc:
            docs.append(doc)
        sleep(0.2)

    if not docs:
        raise RuntimeError("No Wikipedia documents loaded. Check network access.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks   = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore     = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever   = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    ensemble_retriever = CustomEnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    return {
        "faiss": faiss_retriever,
        "bm25":  bm25_retriever,
        "ensemble": ensemble_retriever,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS  (closures over pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def make_tools(pipeline: dict):
    faiss_ret    = pipeline["faiss"]
    bm25_ret     = pipeline["bm25"]
    ensemble_ret = pipeline["ensemble"]

    @tool
    def faiss_rag_retriever(query: str) -> str:
        """Semantic/dense vector search in local FAISS medical knowledge base."""
        return str(faiss_ret.invoke(query))

    @tool
    def bm25_rag_retriever(query: str) -> str:
        """Keyword BM25 search for exact medical terms, drug names, or condition names."""
        return str(bm25_ret.invoke(query))

    @tool
    def hybrid_rag_retriever(query: str) -> str:
        """Hybrid RAG: FAISS semantic + BM25 keyword via Reciprocal Rank Fusion. PRIMARY retriever."""
        return str(ensemble_ret.invoke(query))

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base."""
        wiki_client = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/2.0")
        try:
            page = wiki_client.page(query)
            if page.exists():
                return f"[Source: {page.fullurl}]\n\n{page.text[:3000]}"
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f"Wikipedia error: {e}"

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Analyze comma-separated symptoms, return possible conditions with urgency levels."""
        symptom_map = {
            "fever":               {"conditions": ["Influenza", "COVID-19", "Malaria", "Dengue"],             "urgency": "Medium"},
            "cough":               {"conditions": ["Asthma", "COVID-19", "Tuberculosis", "Bronchitis"],       "urgency": "Low-Medium"},
            "chest pain":          {"conditions": ["Heart disease", "Angina", "Pneumonia", "GERD"],           "urgency": "🚨 HIGH — Seek immediate care"},
            "headache":            {"conditions": ["Migraine", "Hypertension", "Tension headache"],           "urgency": "Low-Medium"},
            "fatigue":             {"conditions": ["Anemia", "Diabetes", "Hypothyroidism", "Depression"],     "urgency": "Low"},
            "shortness of breath": {"conditions": ["Asthma", "Heart failure", "COVID-19"],                   "urgency": "🚨 HIGH — Seek immediate care"},
            "frequent urination":  {"conditions": ["Diabetes mellitus", "UTI", "Prostate issues"],           "urgency": "Medium"},
            "weight loss":         {"conditions": ["Diabetes", "Cancer", "Tuberculosis"],                    "urgency": "Medium-High"},
            "joint pain":          {"conditions": ["Arthritis", "Gout", "Lupus"],                            "urgency": "Low-Medium"},
            "rash":                {"conditions": ["Eczema", "Psoriasis", "Allergic reaction"],              "urgency": "Low"},
            "nausea":              {"conditions": ["Gastritis", "Food poisoning", "Migraine"],               "urgency": "Low-Medium"},
            "dizziness":           {"conditions": ["Hypertension", "Anemia", "Stroke"],                      "urgency": "Medium"},
            "vomiting":            {"conditions": ["Gastritis", "Food poisoning", "Appendicitis"],           "urgency": "Medium"},
            "abdominal pain":      {"conditions": ["Gastritis", "Appendicitis", "IBS"],                      "urgency": "Medium-High"},
            "back pain":           {"conditions": ["Muscle strain", "Herniated disc", "Kidney stones"],      "urgency": "Low-Medium"},
            "swelling":            {"conditions": ["Heart failure", "Kidney disease", "DVT"],                "urgency": "Medium"},
            "blurred vision":      {"conditions": ["Diabetes", "Hypertension", "Glaucoma"],                  "urgency": "🚨 HIGH — Seek immediate care"},
            "confusion":           {"conditions": ["Stroke", "Hypoglycemia", "Dementia"],                    "urgency": "🚨 HIGH — Seek immediate care"},
        }
        entered = [s.strip().lower() for s in symptoms.split(",")]
        matched = {}
        for sym in entered:
            for key, data in symptom_map.items():
                if key in sym:
                    matched[key] = data
        if not matched:
            return "No matches found. Describe symptoms more clearly or consult a healthcare professional."
        lines = ["Symptom Analysis (Educational Only):\n"]
        for sym, data in matched.items():
            lines.append(f"  • {sym.capitalize()}")
            lines.append(f"    Possible: {', '.join(data['conditions'])}")
            lines.append(f"    Urgency: {data['urgency']}")
        lines.append("\nEducational only. Always consult a licensed physician.")
        return "\n".join(lines)

    @tool
    def drug_information(drug_name: str) -> str:
        """Get drug details: class, uses, dosage, side effects, contraindications, interactions."""
        drug_db = {
            "paracetamol": {"class": "Analgesic/Antipyretic", "uses": "Fever, mild-moderate pain",
                            "dosage": "500-1000mg every 4-6h, max 4g/day",
                            "side_effects": "Rare at normal doses; overdose causes liver damage",
                            "contraindications": "Severe liver disease", "interactions": "Warfarin (high doses)"},
            "ibuprofen":   {"class": "NSAID", "uses": "Pain, fever, inflammation",
                            "dosage": "200-400mg every 4-6h, max 1200mg/day OTC",
                            "side_effects": "GI upset, ulcers, increased BP, kidney issues",
                            "contraindications": "Peptic ulcer, kidney disease, pregnancy (3rd trimester)",
                            "interactions": "Aspirin, warfarin, ACE inhibitors"},
            "amoxicillin": {"class": "Penicillin Antibiotic", "uses": "Bacterial infections: respiratory, UTI, ear, skin",
                            "dosage": "250-500mg every 8h or 875mg every 12h",
                            "side_effects": "Diarrhea, nausea, rash, yeast overgrowth",
                            "contraindications": "Penicillin allergy", "interactions": "Warfarin, oral contraceptives"},
            "metformin":   {"class": "Biguanide Antidiabetic", "uses": "Type 2 diabetes (first-line)",
                            "dosage": "Start 500mg twice daily, max 2550mg/day",
                            "side_effects": "GI upset (temporary); rare: lactic acidosis",
                            "contraindications": "Kidney failure (eGFR<30), liver disease",
                            "interactions": "Alcohol, iodinated contrast, cimetidine"},
            "aspirin":     {"class": "Salicylate NSAID / Antiplatelet", "uses": "Pain, fever; low-dose cardiovascular protection",
                            "dosage": "Pain: 325-650mg. Cardio: 75-100mg/day",
                            "side_effects": "GI irritation, tinnitus (high dose), bleeding",
                            "contraindications": "Children <16 (Reye syndrome), peptic ulcer",
                            "interactions": "Warfarin, ibuprofen, SSRIs"},
            "insulin":     {"class": "Hormone / Injectable Antidiabetic", "uses": "Type 1 DM (essential), Type 2 DM (refractory)",
                            "dosage": "Individualized — medical supervision required",
                            "side_effects": "Hypoglycemia, weight gain, injection site reactions",
                            "contraindications": "Hypoglycemia", "interactions": "Beta-blockers, corticosteroids, alcohol"},
        }
        key     = drug_name.lower().strip()
        matched = next((k for k in drug_db if k in key or key in k), None)
        if not matched:
            return f"No local data for '{drug_name}'. Try hybrid_rag_retriever or wikipedia_live_tool."
        i = drug_db[matched]
        return (
            f"{matched.capitalize()}\n"
            f"  Class: {i['class']}\n"
            f"  Uses: {i['uses']}\n"
            f"  Dosage: {i['dosage']}\n"
            f"  Side Effects: {i['side_effects']}\n"
            f"  Contraindications: {i['contraindications']}\n"
            f"  Key Interactions: {i['interactions']}\n\n"
            f"Follow your doctor's/pharmacist's instructions."
        )

    return [hybrid_rag_retriever, faiss_rag_retriever, bm25_rag_retriever,
            wikipedia_live_tool, symptom_checker, drug_information]


# ─────────────────────────────────────────────────────────────────────────────
# LLM FACTORY  (reads keys from env, auto-falls-back)
# ─────────────────────────────────────────────────────────────────────────────
def _get_llm():
    """Return first available LLM based on env keys."""
    if ChatGoogleGenerativeAI and os.environ.get("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.2,
            max_output_tokens=1024,
            convert_system_message_to_human=True,
        )
    if ChatOpenAI and os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.2,
            max_tokens=1024,
        )
    if ChatGroq and os.environ.get("GROQ_API_KEY"):
        for model_id in ["llama-3.1-8b-instant", "llama3-8b-8192", "llama-3.3-70b-versatile"]:
            try:
                return ChatGroq(
                    model=model_id,
                    api_key=os.environ["GROQ_API_KEY"],
                    temperature=0.2,
                    max_tokens=1024,
                )
            except Exception:
                continue
    raise RuntimeError(
        "No LLM available. Set GOOGLE_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in your .env file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH AGENT
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:   Annotated[list, add_messages]
    tools_used: List[str]


def build_agent(pipeline: dict):
    tools          = make_tools(pipeline)
    llm            = _get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_node      = ToolNode(tools)

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
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def ask_agent(agent, history: list, question: str):
    """
    history: mutable list of LangChain message objects (HumanMessage / AIMessage).
    Returns (reply_str, tools_used_list).
    """
    history.append(HumanMessage(content=question))
    result = agent.invoke(
        {"messages": history, "tools_used": []},
        config={"recursion_limit": 25},
    )
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    if not ai_msgs:
        return "Sorry, I couldn't generate a response. Please try again.", []

    final_msg  = ai_msgs[-1]
    reply      = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
    tools_used = result.get("tools_used", [])
    history.append(final_msg)
    return reply, tools_used
