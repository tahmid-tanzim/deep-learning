import os
import json
import arxiv

from typing import List
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

dir_path = os.path.dirname(os.path.realpath(__file__))
RESEARCH_PAPER_DIR = os.path.join(dir_path, "research_papers")

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    file_path = os.path.join(RESEARCH_PAPER_DIR, topic.lower().replace(" ", "_"), "papers_info.json")
    if os.path.exists(file_path):
        print(f"Research Papers for `{topic}` already exist in {file_path}")
        try:
            with open(file_path, "r") as json_file:
                papers_info = json.load(json_file)
                return list(papers_info.keys())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading research papers info from {file_path}: {e}")
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
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
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=4)
    print(f"Results are saved in: {file_path}")
    
    return list(papers_info.keys())

print(search_papers("transformer"))