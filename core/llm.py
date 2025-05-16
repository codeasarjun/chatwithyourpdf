from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

def get_llm():
    return Ollama(model='llama3.2')

def get_embeddings():
    return OllamaEmbeddings(model='nomic-embed-text')
