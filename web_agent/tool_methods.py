import requests
from sympy import sympify, SympifyError
import wikipediaapi

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

    
def wikipedia_page_summarizer(expression: str) -> str:
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


def wikipedia_page_fetcher(expression: str) -> str:
    """
    Tool: extracts the entire page contents for a given Wikipedia
    page if that page exists, else returns an exception.
    """ 
    try: 
        page = wiki.page(page_name)
        if not page.exists():
            return f"No Wikipedia page found for '{page_name}'."
        return page.text
    except Exception as e:
        return f"Error fetching content for '{page_name}'."

