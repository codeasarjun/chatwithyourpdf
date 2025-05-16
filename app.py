import streamlit as st
import tempfile
import os

from core.llm import get_llm, get_embeddings
from core.pdf_loader import load_pdf
from core.rag_chain import build_rag_chain

st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("📄 Upload a PDF and Ask Questions")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    pages = load_pdf(tmp_path)

    llm = get_llm()
    embeddings = get_embeddings()

    chain = build_rag_chain(pages, llm, embeddings)

    question = st.text_input("Ask a question about the uploaded PDF:")

    if question:
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
            st.success("Answer:")
            st.write(response)

    os.remove(tmp_path)
