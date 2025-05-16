import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

# Initialize LLM and embeddings
llm = Ollama(model='llama3.2')
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Prompt template
template = """
Answer the question based only on the context provided.

Context: {context}

Question: {question}

** Do not provide any additional details.
"""
prompt = PromptTemplate.from_template(template)

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit UI
st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("📄 Upload a PDF and Ask Questions")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    # Build retriever from uploaded PDF
    store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = store.as_retriever()

    # Build RAG chain
    chain = (
        {
            'context': retriever | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Q&A Interface
    question = st.text_input("Ask a question about the uploaded PDF:")

    if question:
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
            st.success("Answer:")
            st.write(response)

    # Cleanup temp file
    os.remove(tmp_path)
