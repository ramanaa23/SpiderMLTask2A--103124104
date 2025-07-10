
import streamlit as st
import os

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURE API ---
os.environ["OPENAI_API_KEY"] = "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"  
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# --- LOAD CHROMA VECTORSTORE ---
CHROMA_DIR = "chroma_db"
embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_fn
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LLM SETUP (OpenRouter) ---
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.2,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# --- CUSTOM PROMPT TEMPLATE ---
custom_prompt = PromptTemplate.from_template(
    "You are an expert ML assistant. Use only the provided context to answer the question."
    "Context:{context} Question: {question} Answer:"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# --- STREAMLIT INTERFACE ---
st.set_page_config(page_title="RAG ML Paper Q&A", layout="centered") # Sets 
st.title("📘 ML Paper Q&A (RAG + OpenRouter)")

st.markdown("Ask a question based on the research paper(s) you uploaded to the vector database.")

user_question = st.text_input("🧠 Your question:")
ask_btn = st.button("Ask")

if ask_btn and user_question: 
    with st.spinner("Thinking..."):
        result = qa_chain({"query": user_question})

        st.subheader("✅ Answer:")
        st.write(result["result"])

        with st.expander("📄 Retrieved Context Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}**")
                st.markdown(doc.page_content[:500] + "...")


        
