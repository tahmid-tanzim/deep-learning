import os
from dotenv import load_dotenv, find_dotenv
# import nest_asyncio

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# nest_asyncio.apply()


# Load Documents
documents = SimpleDirectoryReader(input_files=["./resource/metagpt.pdf"]).load_data()

# Split documents in small chunk/nodes
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Define LLM and Embedding model
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

# Define Summary Index and Vector Index over the Same Data
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Define Query Engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

# Set Metadata
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

# Define Router Query Engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("What is the summary of the document?")
print(str(response))
print(len(response.source_nodes))