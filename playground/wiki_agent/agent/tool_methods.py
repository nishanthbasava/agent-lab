import requests
from sympy import sympify, SympifyError
import wikipediaapi
from web_agent.retrieval.vector_store import collection 

wiki = wikipediaapi.Wikipedia(user_agent='AgentProject (nishanth.basava123@gmail.com)', language='en')


def calc_tool(expression: str) -> str:
    """
    Tool: evaluate a math expression using SymPy.
    Supports arithmetic, algebra, calculus (diff/integrate),
    equation solving (solve), simplification, etc.
    """
    try:
        expr = sympify(expression)
        result = expr.evalf() if expr.free_symbols == set() else expr
        return str(result)
    except (SympifyError, TypeError) as e:
        return f"Could not parse expression: {e}"
    
def wikipedia_search(query: str) -> str:
    """
    Tool: searches Wikipedia for pages matching a query and
    returns a list of matching page titles. Use this first when
    you're not sure of the exact page title, then pass the best
    match to wikipedia_page_summarizer or wikipedia_page_fetcher.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5
            },
            headers={"User-Agent": "AgentProject (nishanth.basava123@gmail.com)"},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results found for '{query}'."
        titles = [r["title"] for r in results]
        return "Possible matches: " + ", ".join(titles)
    except requests.RequestException as e:
        return f"Error searching Wikipedia for '{query}': {e}"

    
def wikipedia_page_summarizer(page_name: str) -> str:
    """
    Tool: finds the summary of a specific Wikipedia page
    given a valid page name, otherwise returns an exception.
    """
    try: 
        page = wiki.page(page_name)
        if not page.exists():
            return f"No Wikipedia page found for '{page_name}'."
        return page.summary
    except Exception as e:
        return f"Error fetching summary for '{page_name}'."


def wikipedia_page_fetcher(page_name: str) -> str:
    """
    Tool: extracts the entire page contents for a given Wikipedia
    page if that page exists, else returns an exception.
    """ 
    try: 
        page = wiki.page(page_name)
        if not page.exists():
            return f"No Wikipedia page found for '{page_name}'."
        return page.text[:5000] #right now it just fetches the first 5000 characters to not blow up context.
    except Exception as e:
        return f"Error fetching content for '{page_name}'."


def knowledge_base_retriever(query: str) -> str:
    """
    Tool: accesses the internal knowledge base and retrieves the 3 most related
    documents based on cosine similarity with the query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    #finish this method


def knowledge_base_search_filtered(query: str) -> str:
    '''
    Tool: Semantic search over the internal knowledge base. Returns the 3 most
    relevant document chunks for a given query
    '''
    #optional arguments
    print("Placeholder")


def get_full_document(doc_id: str) -> str:
    #returns full document requested
    print("Placeholder")

def list_knowledge_base_sources() -> str:
    #print(knowledge_base_sources)
    print("Placeholder")

def knowledge_base_search_with_score() -> str:
    print("Placeholder") 