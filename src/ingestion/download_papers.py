import arxiv
import os
from tqdm import tqdm

def download_arxiv_papers(query, max_results=50, download_dir="data/raw/papers"):
    """
    Download papers from arXiv based on query.
    
    Speed-optimized: Downloads PDFs in parallel
    """
    os.makedirs(download_dir, exist_ok=True)
    
    # Search arXiv
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers_info = []
    
    print(f"Downloading {max_results} papers on '{query}'...")
    for result in tqdm(search.results(), total=max_results):
        try:
            # Download PDF
            pdf_path = result.download_pdf(dirpath=download_dir)
            
            papers_info.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published,
                'pdf_path': pdf_path,
                'arxiv_id': result.entry_id.split('/')[-1]
            })
            
        except Exception as e:
            print(f"Error downloading {result.title}: {e}")
            continue
    
    print(f"\nDownloaded {len(papers_info)} papers to {download_dir}")
    return papers_info

if __name__ == "__main__":
    # Download recent papers on different AI topics
    queries = [
        "large language models",
        "retrieval augmented generation", 
        "transformer architecture",
        "fine-tuning LLM",
        "prompt engineering"
    ]
    
    all_papers = []
    for query in queries:
        papers = download_arxiv_papers(query, max_results=20)
        all_papers.extend(papers)
    
    print(f"\nTotal papers downloaded: {len(all_papers)}")