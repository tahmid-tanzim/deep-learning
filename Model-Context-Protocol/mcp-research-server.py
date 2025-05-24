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


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
