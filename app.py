import os
import streamlit as st
import wikipediaapi

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.documents import Document

st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
st.title("🏥 Medical RAG Chatbot")

with st.sidebar:
    st.header("API Keys")
    google_key = st.text_input("Gemini API Key", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    groq_key = st.text_input("Groq API Key", type="password")

if google_key:
    os.environ["GOOGLE_API_KEY"] = google_key
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

@st.cache_resource
def load_medical_db():
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='MediBot')
    topics = [
        "Diabetes mellitus","Hypertension","Asthma",
        "COVID-19","Cancer","Heart disease"
    ]

    docs=[]

    for topic in topics:
        page=wiki.page(topic)
        if page.exists():
            docs.append(
                Document(
                    page_content=page.summary,
                    metadata={"source":topic}
                )
            )

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks=splitter.split_documents(docs)

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore=FAISS.from_documents(chunks,embeddings)
    faiss_retriever=vectorstore.as_retriever()
    bm25=BM25Retriever.from_documents(chunks)

    return faiss_retriever,bm25

faiss_retriever,bm25=load_medical_db()

@tool
def hybrid_rag(query:str):
    faiss_docs=faiss_retriever.invoke(query)
    bm_docs=bm25.invoke(query)

    context=""

    for d in faiss_docs[:2]:
        context += d.page_content + "\n"

    for d in bm_docs[:2]:
        context += d.page_content + "\n"

    return context

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Gemini","OpenAI","Groq"]
)

llm=None

if model_choice=="Gemini" and google_key:
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

elif model_choice=="OpenAI" and openai_key:
    llm=ChatOpenAI(model="gpt-4o-mini")

elif model_choice=="Groq" and groq_key:
    llm=ChatGroq(model="llama-3.3-70b-versatile")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt=st.chat_input("Ask your medical question")

if prompt and llm:

    st.session_state.messages.append(
        {"role":"user","content":prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    context=hybrid_rag.invoke(prompt)

    final_prompt=f"""
    Medical context:
    {context}

    User question:
    {prompt}

    Give a concise answer and include a medical disclaimer.
    """

    response=llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.write(response.content)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response.content
        }
    )
