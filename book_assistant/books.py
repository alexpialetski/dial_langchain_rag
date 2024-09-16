from datetime import datetime
from enum import Enum
import json
from typing import Any, List, Optional, Set
from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document


class PublishStatus(str, Enum):
    PUBLISH = "PUBLISH"
    MEAP = "MEAP"


class Book(BaseModel):
    id: int = Field(..., alias="_id")
    title: str
    isbn: str = ""
    pageCount: int
    publishedDate: Optional[datetime] = None
    thumbnailUrl: str = ""
    shortDescription: str = ""
    longDescription: str = ""
    status: PublishStatus
    authors: List[str]
    categories: List[str]

    @field_validator("publishedDate", mode="before")
    def parse_published_date(cls, value):
        if isinstance(value, dict) and "$date" in value:
            return datetime.strptime(value["$date"], "%Y-%m-%dT%H:%M:%S.%f%z")
        raise ValueError("Invalid date format")


class BookDocument(Document):
    id: str

    def __init__(self, id: str, **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)  # type: ignore[call-arg]


class BooksStorage:
    __books: List[Book] = None

    def get_books(self) -> List[Book]:
        if not self.__books:
            self.__books = self.__load_books()

        return self.__books

    def get_distinct_authors(self) -> list[str]:
        authors = set()
        for book in self.get_books():
            authors.update(book.authors)
        return list(authors)

    def get_distinct_categories(self) -> list[str]:
        categories = set()
        for book in self.get_books():
            categories.update(book.categories)
        return list(categories)

    def to_documents(self) -> List[BookDocument]:
        documents = []

        for book in self.get_books():
            categories_content = (
                "<categories>"
                + "".join(
                    f"<category>{category}</category>" for category in book.categories
                )
                + "</categories>"
            )
            authors_content = (
                "<authors>"
                + "".join(f"<author>{author}</author>" for author in book.authors)
                + "</authors>"
            )

            documents.append(
                BookDocument(
                    id=book.id,
                    page_content=f"<book><title>{book.title}</title><description>{book.longDescription}</description>{authors_content}{categories_content}</book>",
                    metadata={
                        "publish_year": (
                            book.publishedDate.year if book.publishedDate else -1
                        ),
                    },
                )
            )

        return documents

    def __load_books(self) -> List[Book]:
        with open("data.json", "r") as file:
            data = json.load(file)

            return [Book(**item) for item in data]


books_storage = BooksStorage()
