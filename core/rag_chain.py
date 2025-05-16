from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from .utils import format_docs

def build_rag_chain(pages, llm, embeddings):
    store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = store.as_retriever()

    template = """
    Answer the question based only on the context provided.

    Context: {context}

    Question: {question}

    ** Do not provide any additional details.
    """
    prompt = PromptTemplate.from_template(template)

    chain = (
        {
            'context': retriever | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
