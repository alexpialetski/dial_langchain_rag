"""
A simple RAG application based on LangChain.
"""

import os

import uvicorn
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.globals import set_debug

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response


from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from llm import text_model
from tools import tools


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Please provide {name!r} environment variable")
    return value


DIAL_URL = get_env("DIAL_URL")
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

set_debug(LANGCHAIN_DEBUG)

config = {"configurable": {"thread_id": "abc123"}}

memory = MemorySaver()


class CustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self, choice: Choice):
        self._choice = choice

    async def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        self._choice.append_content(token)


system = """
Role:
Act as a professional book store assistant.

Task:
Using the provided data, guide a user through the process of choosing a book to read and help him with possible doubts.
You should be able to compare books or show information in the following format

<format>
Title: title
Description: description
Authors: author1, author2
Categories: category1, category2
</format>

Data:
The provided book database includes title and description of books in xml format, as well as metadata such as the authors and categories.

Constraints:
You should talk professionally, and make an impression of a "salesperson in a shop" and guide a user through the process of choice. 
You should give structured and predictable responses (like comparing two books or showing one item).
"""


class SimpleRAGApplication(ChatCompletion):
    async def chat_completion(self, request: Request, response: Response) -> None:
        with response.create_single_choice() as choice:
            message = request.messages[-1]
            user_query = message.content or ""

            text_model.callback_manager.add_handler(CustomCallbackHandler(choice))

            await response.aflush()

            agent_executor = create_react_agent(
                text_model,
                tools=tools,
                checkpointer=memory,
                state_modifier=system,
            )

            await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=config,
            )


app = DIALApp(DIAL_URL, propagate_auth_headers=True)
app.add_chat_completion("simple-rag", SimpleRAGApplication())


if __name__ == "__main__":
    uvicorn.run(app, port=5000)
