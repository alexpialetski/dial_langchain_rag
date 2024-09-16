from langchain.tools.retriever import create_retriever_tool

from retriever import retrieval_chain

books_rag_tool = create_retriever_tool(
    retrieval_chain,
    "books_retriever",
    "Searches and returns book information from the database.",
)
tools = [books_rag_tool]
