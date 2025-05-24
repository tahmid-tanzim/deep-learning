import asyncio
import os
import json
import logging
import nest_asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
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
        self.session: ClientSession = None
        self.gpt_client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.available_tools: List[dict] = []

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
                    "tool_calls": assistant_message.tool_calls
                })
                
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        logging.info(f"Calling tool {tool_name} with args {tool_args}")
                        
                        try:
                            result = await self.session.call_tool(tool_name, arguments=tool_args)
                            # Convert CallToolResult to string before adding to messages
                            result_str = str(result.content) if hasattr(result, 'content') else str(result)
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

    async def chat_loop(self):
        """Run the main chat loop with proper message management."""
        messages = [{
            "role": "system",
            "content": "You are a helpful research assistant that can search and analyze academic papers using the available tools."
        }]
        
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                
                # Add user message
                messages.append({'role': 'user', 'content': query})
                await self.process_query(messages)
                self._debug_messages(messages)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {e}")
                print(f"\nError: {str(e)}")
                

    async def connect_to_server_and_run(self):
        """Connect to the MCP server and start the chat loop."""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "Model-Context-Protocol/mcp_research_server.py"],
            env=None,
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await self.session.initialize()
                
                # Get available tools from server
                response = await self.session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                # Format tools for OpenAI API
                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in response.tools]
                
                await self.chat_loop()

async def main():
    """Main entry point for the chatbot."""
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())