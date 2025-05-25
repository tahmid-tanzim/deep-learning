import asyncio
import os
import json
import logging
import nest_asyncio
from typing import List, Dict, Any, TypedDict
from contextlib import AsyncExitStack
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
_ = load_dotenv(find_dotenv())

# Configure logging
current_dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir_path, "chatbot.log")),
        logging.StreamHandler()
    ]
)


class ToolChild(TypedDict):
    name: str
    description: str
    parameters: dict


class ToolType(TypedDict):
    type: str
    function: ToolChild


class PromptChild(TypedDict):
    name: str
    description: str
    arguments: dict


class PromptType(TypedDict):
    type: str
    prompt: PromptChild


class MCP_ChatBot:
    """A chatbot that uses MCP (Model Context Protocol) for tool interactions."""

    # Configuration constants
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    MAX_TOKENS = 1096
    MAX_MESSAGES = 10
    MODEL_NAME = 'gpt-4.1-nano'
    MAX_RETRIES = 3
    TIMEOUT = 30

    def __init__(self):
        """Initialize the chatbot with OpenAI client and empty tool list."""
        # self.session: ClientSession = None
        self.sessions: List[ClientSession] = []
        self.gpt_client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.available_tools: List[ToolType] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_prompts: List[PromptType] = []
        self.prompt_to_session: Dict[str, ClientSession] = {}
        self.resource_to_session: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

    @staticmethod
    def truncate_messages(messages: List[Dict[str, Any]], max_messages: int = MAX_MESSAGES) -> List[Dict[str, Any]]:
        """
        Truncate message history to prevent token limit issues.
        
        Args:
            messages: List of message dictionaries
            max_messages: Maximum number of messages to keep
            
        Returns:
            Truncated list of messages
        """
        if len(messages) > max_messages:
            system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
            recent_messages = messages[-max_messages:]
            return [system_message] + recent_messages if system_message else recent_messages
        return messages

    @staticmethod
    def _debug_messages(messages: List[Dict[str, Any]]):
        """
        Debug the messages by printing them in a readable format.
        
        Args:
            messages: List of message dictionaries
        """
        print("\n--------------------------------------------------\n")
        # Create a serializable version of messages for debugging
        serializable_messages = []
        for msg in messages:
            serializable_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            if "tool_calls" in msg and msg["tool_calls"]:
                serializable_msg["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in msg["tool_calls"]
                ]
            serializable_messages.append(serializable_msg)

        print(json.dumps(serializable_messages, indent=4))
        print("\n--------------------------------------------------\n")

    async def process_query(self, messages: List[Dict[str, Any]]) -> None:
        """
        Process a query with retry logic and proper error handling.
        
        Args:
            messages: List of message dictionaries containing the conversation history
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                messages = self.truncate_messages(messages)

                # Get response from OpenAI
                response = self.gpt_client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=messages,
                    tools=self.available_tools,
                    max_tokens=self.MAX_TOKENS,
                    timeout=self.TIMEOUT
                )

                assistant_message = response.choices[0].message

                # Add assistant's message to the conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None
                })

                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        logging.info(f"Calling tool {tool_name} with args {tool_args}")

                        try:
                            # Call a tool
                            session = self.tool_to_session.get(tool_name)
                            if not session:
                                raise ValueError(f"No session found for tool {tool_name}")

                            result = await session.call_tool(tool_name, arguments=tool_args)

                            # Convert CallToolResult to string before adding to messages
                            result_str = result.content if hasattr(result, 'content') else str(result)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_str
                            })
                        except Exception as e:
                            logging.error(f"Error executing tool {tool_name}: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error executing tool: {str(e)}"
                            })
                    continue
                else:
                    print(assistant_message.content)
                    return

            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    logging.error(f"Failed after {self.MAX_RETRIES} attempts: {e}")
                    print(f"Error: {str(e)}")
                    return
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                continue

    async def get_resource(self, resource_uri: str) -> str:
        """Get a resource from the server."""
        session = self.resource_to_session.get(resource_uri)

        if not session and resource_uri.startswith("papers://"):
            for uri, s in self.resource_to_session.items():
                if uri.startswith("papers://"):
                    session = s
                    break

        if not session:
            error_msg = f"No session found for resource {resource_uri}"
            logging.error(error_msg)
            print(error_msg)
            return

        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                content = result.contents[0].text
                logging.info(f"Resource {resource_uri} contents: {content}")
                print(content)
            else:
                error_msg = f"No contents found for resource {resource_uri}"
                logging.error(error_msg)
                print(error_msg)
        except Exception as e:
            error_msg = f"Error getting resource {resource_uri}: {e}"
            logging.error(error_msg)
            print(error_msg)

    async def list_prompts(self):
        if not self.available_prompts:
            logging.error("No prompts available")
            return

        for prompt in self.available_prompts:
            logging.info(f"Prompt: {prompt['prompt']['name']}")
            logging.info(f"Description: {prompt['prompt']['description']}")
            if prompt['prompt']['arguments']:
                logging.info(f"Arguments: ")
                for arg in prompt['prompt']['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    logging.info(f"  - {arg_name}")

    async def execute_prompt(self, prompt_name: str, arguments: dict):
        session = self.prompt_to_session.get(prompt_name, None)
        if not session:
            logging.error(f"No session found for prompt {prompt_name}")
            return

        try:
            result = await session.get_prompt(prompt_name, arguments=arguments)
            if result and result.messages:
                prompt_content = result.messages[0].content
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) for item in prompt_content)
                logging.info(f"Prompt {prompt_name} \nResult: {text}")
                await self.process_query(text)
        except Exception as e:
            logging.error(f"Error executing prompt {prompt_name}: {e}")

    async def chat_loop(self):
        """Run the main chat loop with proper message management."""
        messages = [{
            "role": "system",
            "content": "You are a helpful research assistant that can search and analyze academic papers using the available tools."
        }]

        print("MCP Chatbot Started")
        print("Type your queries or 'quit' to exit.")
        print("User @folders to list available topics")
        print("User @<topic> to search paper in that topic")
        print("User /prompts to list available prompts")
        print("User /prompt <name> <arg1=value1> <arg2=value2> to execute a prompt")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    break
                if query.startswith('@'):
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri=resource_uri)
                    continue
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()
                    if command == '/prompts':
                        await self.list_prompts()
                        continue
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>...")
                            continue
                        prompt_name = parts[1]
                        args = {}
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        await self.execute_prompt(prompt_name=prompt_name, arguments=args)
                    else:
                        print(f"Unknown command: {command}")
                        continue
                await self.process_query(messages)
                self._debug_messages(messages)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {e}")

    async def cleanup(self):
        """Clean up all MCP sessions."""
        await self.exit_stack.aclose()

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to the MCP server and start the chat loop."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            reader, writer = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(reader, writer)
            )
            await session.initialize()
            self.sessions.append(session)

            # 1. Get available tools from server
            tool_response = await session.list_tools()
            if tool_response.tools:
                print(f"\nConnected to {server_name} with tools:", [t.name for t in tool_response.tools])
                for tool in tool_response.tools:
                    self.tool_to_session[tool.name] = session
                    self.available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })

            # 2. Get available prompts from server
            prompt_response = await session.list_prompts()
            if prompt_response.prompts:
                print(f"\nConnected to {server_name} with prompts:", [p.name for p in prompt_response.prompts])
                for prompt in prompt_response.prompts:
                    self.prompt_to_session[prompt.name] = session
                    self.available_prompts.append({
                        "type": "prompt",
                        "prompt": {
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        }
                    })
            # 3. Get available resources from server
            resource_response = await session.list_resources()
            if resource_response.resources:
                print(f"\nConnected to {server_name} with resources:", [r.name for r in resource_response.resources])
                for resource in resource_response.resources:
                    self.resource_to_session[str(resource.uri)] = session
                    # self.available_resources.append({
                    #     "type": "resource",
                    #     "resource": {
                    #         "name": resource.name,
                    #         "description": resource.description,
                    #         "arguments": resource.inputSchema
                    #     }   
                    # })
        except Exception as e:
            logging.error(f"Error connecting server {server_name}: {e}")
            return

    async def connect_to_servers(self):  # new
        """Connect to all configured MCP servers."""
        server_config_file_path = os.path.join(current_dir_path, "server_config.json")
        try:
            with open(server_config_file_path, "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise


async def main():
    """Main entry point for the chatbot."""
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
