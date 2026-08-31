import wikipediaapi
from tool_methods import #write the method names here once I do this


tools = [
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
                "name": {
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
                "expression": {
                    "type":"string",
                    "description": "SymPy-parseable expression or command"
                }
            },
            "required":["expression"]
        }
    }

    #calculator tool
    #wikipedia tool


]