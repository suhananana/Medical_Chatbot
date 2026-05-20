"""
agent.py — MediBot core logic
LangChain + LangGraph + FAISS + Wikipedia + Gemini
"""

import os
import time
from typing import Annotated, List, TypedDict

import wikipedia as wiki_lib
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import create_retriever_tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Medical topics loaded into FAISS ──────────────────────────────────────────
MEDICAL_TOPICS = [
    "Diabetes mellitus",
    "Hypertension",
    "COVID-19 pandemic",
    "Cancer",
    "Coronary artery disease",
    "Asthma",
    "Alzheimer's disease",
    "Tuberculosis",
    "Vaccination",
    "Antibiotic",
    "Human body",
    "Immune system",
]

SYSTEM_PROMPT = """\
You are MediBot 🏥, an expert Medical AI Assistant.

Answer questions about diseases, symptoms, treatments,
medications, anatomy, and general health.

Always remind users to consult a doctor.

Tools available:
1. medical_rag_retriever – local FAISS knowledge base (use FIRST)
2. wikipedia – live Wikipedia search (use as fallback or for extra detail)
3. symptom_checker – maps symptoms to possible conditions

Rules:
• Always try medical_rag_retriever first.
• Use wikipedia for additional or up-to-date detail.
• Use symptom_checker when the user describes symptoms.
• At the end of your reply, mention which tools you used.
• End every reply with:

⚠️ For informational purposes only. Always consult a qualified healthcare professional.
"""

# ── Agent state ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tools_used: List[str]


# ── Wikipedia helper ──────────────────────────────────────────────────────────
def _fetch_wiki_page(title: str) -> Document | None:
    """Fetch one Wikipedia page and return a LangChain Document (or None)."""
    try:
        page = wiki_lib.page(title, auto_suggest=False)
        return Document(
            page_content=page.content[:5000],
            metadata={"source": page.url, "title": page.title},
        )
    except wiki_lib.exceptions.DisambiguationError as de:
        try:
            page = wiki_lib.page(de.options[0], auto_suggest=False)
            return Document(
                page_content=page.content[:5000],
                metadata={"source": page.url, "title": page.title},
            )
        except Exception:
            return None
    except Exception:
        return None


# ── Build vector store ────────────────────────────────────────────────────────
def build_vectorstore(progress_callback=None):
    """
    Download Wikipedia articles for MEDICAL_TOPICS and index them in FAISS.

    progress_callback: optional callable(topic_name, loaded_count, total)
                       called after each topic is processed — use it to
                       update a Streamlit progress bar.
    """
    wiki_lib.set_lang("en")
    all_docs: List[Document] = []
    total = len(MEDICAL_TOPICS)

    for i, topic in enumerate(MEDICAL_TOPICS):
        try:
            hits = wiki_lib.search(topic, results=3)
            for title in hits[:2]:
                doc = _fetch_wiki_page(title)
                if doc:
                    all_docs.append(doc)
                time.sleep(0.3)
        except Exception:
            pass

        if progress_callback:
            progress_callback(topic, len(all_docs), i + 1, total)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── Build the full agent ──────────────────────────────────────────────────────
def build_agent(gemini_api_key: str, vectorstore: FAISS):
    """
    Compile the LangGraph ReAct agent and return it alongside the tool list.
    """
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    # --- Tool 1: RAG retriever ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool = create_retriever_tool(
        retriever,
        name="medical_rag_retriever",
        description=(
            "Search the local medical knowledge base built from Wikipedia. "
            "Use this FIRST for any medical question about diseases, symptoms, "
            "treatments, medications, anatomy, or health conditions."
        ),
    )

    # --- Tool 2: Live Wikipedia ---
    wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    wiki_live_tool = WikipediaQueryRun(
        api_wrapper=wiki_api,
        description=(
            "Search Wikipedia live for medical topics not covered locally "
            "or when more up-to-date information is needed."
        ),
    )

    # --- Tool 3: Symptom checker ---
    @tool
    def symptom_checker(symptoms: str) -> str:
        """
        Given a comma-separated list of symptoms, return possible medical
        conditions. Educational purposes only — always consult a doctor.
        """
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
        }
        entered = [s.strip().lower() for s in symptoms.split(",")]
        result = {}
        for sym in entered:
            for key, conds in symptom_map.items():
                if key in sym:
                    result[key] = conds
        if not result:
            return "No specific matches found. Please consult a healthcare professional."
        lines = ["⚠️ Possible conditions (educational only):\n"]
        for sym, conds in result.items():
            lines.append(f"  • {sym.capitalize()}: {', '.join(conds)}")
        lines.append("\n🩺 Always consult a licensed physician for diagnosis.")
        return "\n".join(lines)

    tools = [rag_tool, wiki_live_tool, symptom_checker]

    # --- LLM ---
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.2,
        convert_system_message_to_human=False,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # --- Nodes ---
    def call_model(state: AgentState):
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
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

    # --- Graph ---
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(), tools


# ── Single-turn query helper ──────────────────────────────────────────────────
def ask_medibot(agent, question: str, history: list) -> dict:
    """
    Run one turn of the conversation.

    Parameters
    ----------
    agent    : compiled LangGraph agent
    question : user's new message
    history  : list of LangChain BaseMessage objects (mutated in place)

    Returns
    -------
    dict with:
        reply      – str  (assistant's answer)
        tools_used – list[str]
    """
    history.append(HumanMessage(content=question))

    result = agent.invoke(
        {"messages": history, "tools_used": []},
        config={"recursion_limit": 15},
    )

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    final_msg = ai_msgs[-1]
    reply = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
    tools_used = result.get("tools_used", [])

    history.append(final_msg)
    return {"reply": reply, "tools_used": tools_used}
