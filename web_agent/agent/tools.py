import wikipediaapi
from tool_methods import calc_tool, wikipedia_page_fetcher, wikipedia_page_summarizer, wikipedia_search

tools = [

    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for page titles matching a query. Use this when you don't know the exact page title before calling wikipedia_page_summarizer or wikipedia_page_fetcher.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms, e.g. 'quantum entanglement' or 'french revolution causes'"
                }
            },
            "required": ["query"]
        }
    },

    { 
        "name":"wikipedia_page_summarizer",
        "description":"Provides a summary of a valid Wikipedia page and throws an error if not provided with a valid page name",
        "input_schema": {
            "type":"object",
            "properties":{
                "name": {
                    "type":"string",
                    "description":"The name of the webpage (i.e. 'Python_(programming_language)')"
                }
            },
            "required":["name"]
        }
    },

    {
        "name":"wikipedia_page_fetcher",
        "desciption":"",
        "input_schema": {
            "type":"object",
            "properties":{
                "page_name": {
                    "type":"string",
                    "description":"The name of the webpage (i.e. 'Python_(programming_language)')"
                }
            },
            "required":["name"]
        }
    },

    {
        "name":"sympy_calc",
        "description": '''Evaluate math expressions, solve equations, or perform calculus using 
            "SymPy. Input should be valid Python/SymPy syntax, e.g. 'diff(x**2, x)' or 'solve(x**2 - 4, x)'.''',
        "input_schema": {
            "type":"object",
            "properties": {
                "page_name": {
                    "type":"string",
                    "description": "SymPy-parseable expression or command"
                }
            },
            "required":["expression"]
        }
    }

]