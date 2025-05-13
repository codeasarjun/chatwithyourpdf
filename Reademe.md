# 💬 Chat with PDF using LLM

Interact with PDF documents using natural language! This project leverages local large language models (LLMs) and embedding-based vector search to answer questions about PDF files efficiently and privately.

## 🚀 Features

- Ask natural language questions about the content of your PDFs.
- Local inference using `llama3:8b` via Ollama — no data leaves your machine.
- Fast and lightweight vector search with `DocArrayInMemorySearch`.
- Embedding powered by `nomic-embed-text` for semantic understanding.

## 🧠 Tech Stack

- **LLM**: `llama3:8b` via [Ollama](https://ollama.com/)
- **PDF Loader**: `PyPDFLoader` from LangChain
- **Embeddings**: `nomic-embed-text`
- **Vector Store**: `DocArrayInMemorySearch`
- **Framework**: Python + LangChain

## 📦 Installation

### Prerequisites

- Python 3.10+
- Ollama installed and running
- `llama3` model pulled via Ollama
- Required Python packages installed (see `requirements.txt` or instructions in the Usage section)

## 🗂️ Usage

1. **Load a PDF document** using PyPDFLoader.
2. **Generate embeddings** with `nomic-embed-text`.
3. **Store and search** using `DocArrayInMemorySearch`.
4. **Query** using `llama3:8b` for context-aware responses.

## ✅ To-Do

- [ ] Add a simple web UI using Streamlit or Gradio
- [ ] Enable support for querying multiple PDFs
- [ ] Add persistent vector store option (e.g., FAISS or Chroma)
- [ ] Improve context retention and memory in conversations

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
