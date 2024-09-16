from dotenv import load_dotenv

load_dotenv()


from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from llm import text_model
from tools import tools

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

config = {"configurable": {"thread_id": "abc123"}}

memory = MemorySaver()

agent_executor = create_react_agent(
    # text_model, tools, checkpointer=memory, state_modifier=prompt
    text_model,
    tools=tools,
    checkpointer=memory,
    state_modifier=system,
)

for s in agent_executor.stream(
    {"messages": [HumanMessage(content="Hello, who are you?")]}, config=config
):
    print(s)
    print("----")
