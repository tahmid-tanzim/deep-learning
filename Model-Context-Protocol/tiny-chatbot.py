import os
import json
import arxiv
import logging
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

# Load environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
gpt_client = OpenAI()

# Configuration
MAX_TOKENS = 1096
MAX_MESSAGES = 10
MODEL_NAME = 'gpt-4.1-nano'
MAX_RETRIES = 3
TIMEOUT = 30

current_dir_path = os.path.dirname(os.path.realpath(__file__))
RESEARCH_PAPER_DIR = os.path.join(current_dir_path, "research_papers")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir_path, "chatbot.log")),
        logging.StreamHandler()
    ]
)

def truncate_messages(messages: List[Dict[str, Any]], max_messages: int = MAX_MESSAGES) -> List[Dict[str, Any]]:
    """Truncate message history to prevent token limit issues."""
    if len(messages) > max_messages:
        # Keep system message if exists and most recent messages
        system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
        recent_messages = messages[-max_messages:]
        if system_message:
            return [system_message] + recent_messages
        return recent_messages
    return messages

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    logging.info(f"Searching papers for topic: {topic} with max_results: {max_results}")
    
    file_path = os.path.join(RESEARCH_PAPER_DIR, topic.lower().replace(" ", "_"), "papers_info.json")
    if os.path.exists(file_path):
        logging.info(f"Found existing research papers for topic '{topic}' in {file_path}")
        try:
            with open(file_path, "r") as json_file:
                papers_info = json.load(json_file)
                paper_ids = list(papers_info.keys())
                logging.info(f"Successfully loaded {len(paper_ids)} existing papers")
                return paper_ids
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading research papers info from {file_path}: {e}")
    
    logging.info("No existing papers found, searching arXiv...")
    client = arxiv.Client()

    try:
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = client.results(search)
        
        # Create directory for this topic
        topic_path = os.path.join(RESEARCH_PAPER_DIR, topic.lower().replace(" ", "_"))
        os.makedirs(topic_path, exist_ok=True)
        file_path = os.path.join(topic_path, "papers_info.json")
        
        # Process each paper and add to papers_info  
        papers_info = {}
        for paper in papers:
            paper_id = paper.get_short_id()
            papers_info[paper_id] = {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'pdf_url': paper.pdf_url,
                'categories': paper.categories,
                'published': str(paper.published.date())
            }
            logging.debug(f"Processed paper: {paper_id} - {paper.title}")
        
        # Save updated papers_info to json file
        with open(file_path, "w") as json_file:
            json.dump(papers_info, json_file, indent=4)
        logging.info(f"Successfully saved {len(papers_info)} papers to {file_path}")
        
        return list(papers_info.keys())
    except Exception as e:
        logging.error(f"Error during arXiv search: {e}")
        raise

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
    logging.info(f"Extracting information for paper ID: {paper_id}")
 
    for topic in os.listdir(RESEARCH_PAPER_DIR):
        logging.debug(f"Searching paper_id {paper_id} in topic {topic}")
        topic_path = os.path.join(RESEARCH_PAPER_DIR, topic)
        if os.path.isdir(topic_path):
            file_path = os.path.join(topic_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            logging.info(f"Found paper {paper_id} in topic {topic}")
                            return json.dumps(papers_info[paper_id], indent=4)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logging.error(f"Error reading {file_path}: {str(e)}")
                    continue
    
    logging.info(f"Paper {paper_id} not found in local storage, searching arXiv...")
    client = arxiv.Client()
    try:
        search = arxiv.Search(
            query=paper_id,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        paper = next(client.results(search))
        new_paper_id = paper.get_short_id()
        new_papers_info = {
            new_paper_id: {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'pdf_url': paper.pdf_url,
                'categories': paper.categories,
                'published': str(paper.published.date())
            }
        }
        
        # Create miscellaneous directory if it doesn't exist
        misc_dir = os.path.join(RESEARCH_PAPER_DIR, "miscellaneous")
        os.makedirs(misc_dir, exist_ok=True)
        
        # Read existing papers info if file exists
        file_path = os.path.join(misc_dir, "papers_info.json")
        existing_papers = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as json_file:
                    existing_papers = json.load(json_file)
                logging.debug(f"Loaded {len(existing_papers)} existing papers from miscellaneous directory")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.error(f"Error reading existing papers from {file_path}: {e}")
                existing_papers = {}
        
        # Update with new paper info
        existing_papers.update(new_papers_info)
        
        # Save the updated papers info to the json file
        with open(file_path, "w") as json_file:
            json.dump(existing_papers, json_file, indent=4)
        logging.info(f"Successfully saved new paper info to {file_path}")
        return json.dumps(new_papers_info[new_paper_id], indent=4)
    except Exception as e:
        error_msg = f"There's no saved information related to paper {paper_id}. Error: {e}"
        logging.error(error_msg)
        return error_msg

mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for"
                    }, 
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topic directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]

def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Execute a tool with proper error handling and logging."""
    try:
        result = mapping_tool_function[tool_name](**tool_args)
        if result is None:
            return "The operation completed but didn't return any results."
        elif isinstance(result, list):
            return ', '.join(result)
        elif isinstance(result, dict):
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as e:
        logging.error(f"Error executing tool {tool_name}: {e}")
        raise

def process_query(messages: List[Dict[str, Any]]) -> None:
    """Process a query with retry logic and proper error handling."""
    for attempt in range(MAX_RETRIES):
        try:
            # Truncate messages if needed
            messages = truncate_messages(messages)
            
            response = gpt_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT
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
                        result = execute_tool(tool_name, tool_args)
                        # Add tool response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    except Exception as e:
                        logging.error(f"Error executing tool {tool_name}: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error executing tool: {str(e)}"
                        })
                
                # Continue the conversation with the tool results
                continue
            else:
                # If no tool calls, print the response and return
                print(assistant_message.content)
                return
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"Failed after {MAX_RETRIES} attempts: {e}")
                print(f"Error: {str(e)}")
                return
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            continue

def chat_loop():
    """Main chat loop with proper message management."""
    messages = [{
        "role": "system",
        "content": "You are a helpful research assistant that can search and analyze academic papers."
    }]
    
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            
            messages.append({'role': 'user', 'content': query})
            process_query(messages)
            print("\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.error(f"Error in chat loop: {e}")
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat_loop()