import os
from dotenv import load_dotenv, find_dotenv
# import nest_asyncio

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# nest_asyncio.apply()


# Define simple tool
def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y


def mystery(x: int, y: int) -> int:
    """Mystery function that operates on top of two numbers."""
    return (x * x) + (2 * x * y) + (y * y)


add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

response = llm.predict_and_call(
    [add_tool, mystery_tool],
    "Tell me the output of the add function on 2 and 9.",
    verbose=True
)
print(str(response))
