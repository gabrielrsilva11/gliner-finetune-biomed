import json
from Bio import Entrez

Entrez.email = "default@email.com" 

def fetch_pubmed_case_reports(query: str, max_results: int = 50) -> list:
    """
    Fetches article metadata and abstracts from PubMed based on a specific query.
    """
    print(f"Searching PubMed for: {query}")
    
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    pmids = search_results.get("IdList", [])
    if not pmids:
        print("No results found.")
        return[]
    
    print(f"Found {len(pmids)} articles. Fetching abstracts...")
    
    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    articles = Entrez.read(fetch_handle)
    fetch_handle.close()
    
    extracted_data =[]
    
    for article in articles.get("PubmedArticle",[]):
        medline_citation = article["MedlineCitation"]
        article_data = medline_citation["Article"]
        
        pmid = str(medline_citation["PMID"])
        title = article_data.get("ArticleTitle", "No Title Available")
        
        abstract_texts = article_data.get("Abstract", {}).get("AbstractText",[])
        if abstract_texts:
            abstract = " ".join([str(text) for text in abstract_texts])
        else:
            abstract = "No Abstract Available"
            
        if abstract != "No Abstract Available":
            extracted_data.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract
            })
            
    print(f"Successfully extracted {len(extracted_data)} abstracts.\n")
    return extracted_data

def save_to_json(data: list, filename: str):
    """Saves the extracted list of dictionaries to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {filename}")


if __name__ == "__main__":
    # General Cases
    general_query = '"Case Reports"[Publication Type]'
    
    # ADE Focused
    filtered_query = (
        '"Case Reports"[Publication Type] AND '
        '("Drug Interactions"[Mesh] OR "Drug-Related Side Effects and Adverse Reactions"[Mesh])'
    )
    
    NUM_REPORTS_TO_FETCH = 100
    
    # Fetch and save general reports
    general_reports = fetch_pubmed_case_reports(general_query, max_results=NUM_REPORTS_TO_FETCH)
    save_to_json(general_reports, "general_case_reports.json")
    
    # Fetch and save filtered reports
    filtered_reports = fetch_pubmed_case_reports(filtered_query, max_results=NUM_REPORTS_TO_FETCH)
    save_to_json(filtered_reports, "filtered_ade_case_reports.json")