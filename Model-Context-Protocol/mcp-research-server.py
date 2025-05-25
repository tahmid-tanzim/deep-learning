import os
import json
import arxiv
import logging
from typing import List
from mcp.server.fastmcp import FastMCP

# Configure logging
current_dir_path = os.path.dirname(os.path.realpath(__file__))
RESEARCH_PAPER_DIR = os.path.join(current_dir_path, "research_papers")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir_path, "chatbot.log")),
        logging.StreamHandler()
    ]
)

# Initialize FastMCP server
mcp = FastMCP("research_mcp_server")

# Ensure research papers directory exists
os.makedirs(RESEARCH_PAPER_DIR, exist_ok=True)


@mcp.tool()
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

    # Check for existing papers
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

    # Search arXiv for new papers
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

        # Process and store paper information
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

        # Save papers information
        with open(file_path, "w") as json_file:
            json.dump(papers_info, json_file, indent=4)
        logging.info(f"Successfully saved {len(papers_info)} papers to {file_path}")

        return list(papers_info.keys())
    except Exception as e:
        logging.error(f"Error during arXiv search: {e}")
        raise


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
    logging.info(f"Extracting information for paper ID: {paper_id}")

    # Search in existing topic directories
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

    # Search arXiv if not found locally
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

        # Store in miscellaneous directory
        misc_dir = os.path.join(RESEARCH_PAPER_DIR, "miscellaneous")
        os.makedirs(misc_dir, exist_ok=True)

        # Read existing papers info
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

        # Update and save papers info
        existing_papers.update(new_papers_info)
        with open(file_path, "w") as json_file:
            json.dump(existing_papers, json_file, indent=4)
        logging.info(f"Successfully saved new paper info to {file_path}")
        return json.dumps(new_papers_info[new_paper_id], indent=4)
    except Exception as e:
        error_msg = f"There's no saved information related to paper {paper_id}. Error: {e}"
        logging.error(error_msg)
        return error_msg


@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
    List all available topic folders in the papers directory.
    
    This resource provides a simple list of all available topic folders.
    """
    folders = []

    # Get all topic directories
    if os.path.exists(RESEARCH_PAPER_DIR):
        for topic_dir in os.listdir(RESEARCH_PAPER_DIR):
            topic_path = os.path.join(RESEARCH_PAPER_DIR, topic_dir)
            if os.path.isdir(topic_path):
                papers_file = os.path.join(topic_path, "papers_info.json")
                if os.path.exists(papers_file):
                    folders.append(topic_dir)

    # Create a simple markdown list
    content = "# Available Topics\n\n"
    if folders:
        for folder in folders:
            content += f"- {folder}\n"
        content += f"\nUse @{folder} to access papers in that topic.\n"
    else:
        content += "No topics found.\n"

    return content


@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    Get detailed information about papers on a specific topic.
    
    Args:
        topic: The research topic to retrieve papers for
    """
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = os.path.join(RESEARCH_PAPER_DIR, topic_dir, "papers_info.json")

    if not os.path.exists(papers_file):
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."

    try:
        with open(papers_file, 'r') as f:
            papers_data = json.load(f)

        # Create markdown content with paper details
        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(papers_data)}\n\n"

        for paper_id, paper_info in papers_data.items():
            content += f"## {paper_info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(paper_info['authors'])}\n"
            content += f"- **Published**: {paper_info['published']}\n"
            content += f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n\n"
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"

        return content
    except json.JSONDecodeError:
        return f"# Error reading papers data for {topic}\n\nThe papers data file is corrupted."


@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """
    Generate a prompt for Claude to find and discuss academic papers on a specific topic.
    
    Args:
        topic: The topic to search for papers
        num_papers: Number of papers to search for (default: 5)
        
    Returns:
        A formatted prompt string for searching and analyzing papers
    """
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions:
    1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers})
    2. For each paper found, extract and organize the following information:
       - Paper title
       - Authors
       - Publication date
       - Brief summary of the key findings
       - Main contributions or innovations
       - Methodologies used
       - Relevance to the topic '{topic}'
    
    3. Provide a comprehensive summary that includes:
       - Overview of the current state of research in '{topic}'
       - Common themes and trends across the papers
       - Key research gaps or areas for future investigation
       - Most impactful or influential papers in this area
    
    4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.
    
    Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
