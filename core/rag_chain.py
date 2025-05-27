from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from .utils import format_docs


#need to enhance the logic
def build_rag_chain(pages, llm, embeddings):
    store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = store.as_retriever()

    template = """
You are an assistant answering questions based strictly on the provided context.

- Use ONLY the information in the context.
- If the answer is not present or is unclear, say "The answer is not in the document."

Context:
{context}

Question:
{question}
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
