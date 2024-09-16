from dotenv import load_dotenv

load_dotenv()

import chromadb
from collections.abc import Callable
from chromadb.config import Settings

from books import books_storage
from llm import embeddings_model

# chroma_client = chromadb.PersistentClient(
#     path="./test", settings=Settings(allow_reset=True)
# )
# chroma_client.reset()

chroma_client = chromadb.PersistentClient(path="./test")


# def setup_string_collection(
#     collection_name: str, documents_func: Callable[[], list[str]]
# ) -> chromadb.Collection:
#     try:
#         collection = chroma_client.get_collection(name=collection_name)
#     except ValueError as e:
#         print(f"CREATING VECTOR COLLECTION: {collection_name}")

#         documents = documents_func()
#         collection = chroma_client.create_collection(name=collection_name)

#         collection.add(
#             documents=documents,
#             ids=[f"id{index}" for index, _ in enumerate(documents)],
#         )

#     return collection


def setup_books_collection() -> chromadb.Collection:
    try:
        collection = chroma_client.get_collection(name="books")
    except ValueError as e:
        print(f"CREATING VECTOR COLLECTION: books")

        documents = books_storage.to_documents()
        collection = chroma_client.create_collection(name="books")

        embeddings = embeddings_model.embed_documents(
            [doc.page_content for doc in documents]
        )

        collection.add(
            documents=[doc.page_content for doc in documents],
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id for doc in documents],
        )

    return collection


# authors_collection = setup_string_collection(
#     "authors", lambda: books_storage.get_distinct_authors()
# )

books_collection = setup_books_collection()
