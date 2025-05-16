import streamlit as st
import tempfile
import os

from core.llm import get_llm, get_embeddings
from core.pdf_loader import load_pdf
from core.rag_chain import build_rag_chain

# Page settings
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("📄 Upload a PDF and Chat with It")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Proceed if PDF is uploaded
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF and setup RAG
    pages = load_pdf(tmp_path)
    llm = get_llm()
    embeddings = get_embeddings()
    chain = build_rag_chain(pages, llm, embeddings)

    # Remove the temp file after loading
    os.remove(tmp_path)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Ask something about the PDF..."):
        # Display user's message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the RAG chain
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save messages to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
