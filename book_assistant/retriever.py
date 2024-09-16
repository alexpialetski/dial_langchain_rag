from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from llm import text_model, embeddings_model
from vector_store import books_collection


class Search(BaseModel):
    """Search over a database of books."""

    query: str = Field(
        ...,
        description="Similarity search query applied to books info.",
    )
    author_name: Optional[str] = Field(None, description="Author of the book")
    publish_year: Optional[int] = Field(None, description="Year book was published")


system = """You are an expert at converting user questions into database queries. \
You have access to a database of information about different kind of books. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
structured_llm = text_model.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


def retrieval(search: Search) -> List[Document]:
    _filter = {}

    # Filter by publish year if provided
    if search.publish_year is not None:
        _filter["publish_year"] = {"$eq": search.publish_year}

    return books_collection.query(
        query_embeddings=embeddings_model.embed_query(search.query), where=_filter
    )


retrieval_chain = query_analyzer | retrieval

# print(retrieval_chain.invoke("book about Apache Portals published in 2005"))
