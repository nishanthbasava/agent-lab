wiki_tools = [

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
                "page_name": {
                    "type":"string",
                    "description":"The name of the webpage (i.e. 'Python_(programming_language)')"
                }
            },
            "required":["name"]
        }
    },

    {
        "name":"wikipedia_page_fetcher",
        "description":'''extracts the entire page contents for a given Wikipedia
        page if that page exists, else returns an exception.''',
        "input_schema": {
            "type":"object",
            "properties":{
                "page_name": {
                    "type":"string",
                    "description":"The name of the webpage (i.e. 'Python_(programming_language)')"
                }
            },
            "required":["page_name"]
        }
    }
]


calc_tools = [

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
]


kb_tools = [
    
    {
        "name":"knowledge_base_retriever",
        "description":'''Semantic search over the internal knowledge base. Returns the 3 most
        relevant document chunks for a given query''',
        "input_schema": {
            "type":"object",
            "properties": {
                "query": {
                    "type":"string",
                    "description": '''Natural language question or search phrase to look up in the internal knowledge base. 
                        Be specific — include key terms, names, or topics rather than a full sentence.'''
                }
            },
            "required":["query"]
        }
    },

    {
        "name":"knowledge_base_search_filtered",
        "description":"Semantic search restricted to documents matching filters (e.g. a specific source file or document type)",
        "input_schema": {
            "type":"object",
            "properties": {
                "query": {
                    #fill this in in a bit after outline
                },
                "source": {

                },
                "doc_type": {

                },
                "n_results": {

                }
            },
            "required": ["query"]
        }
    },

    {
        "name":"get_full_document",
        "description":'''Retrieve the full original documebnt given its ID, useful when a search result
        chunk needs more context.''',
        "input_schema": {
            "type":"object",
            "properties":{
                "doc_id": {
                    #fill in property characteristics
                }
            }
        }
    },

    {
        "name":"list_knowledge_base_sources",
        "description":'''lists distinct source documents/categories available in the knowledge base,
        so the agent knows what is can search.''',
        "input_schema": {
            "type":"object",
            "properties":  {},
            "required":[]
        }
    },

    {
        "name":"knowledge_base_search_with_score",
        "description":"yo", #fill this in
        "input_schema": {
            "type":"object",
            "properties":{},
            "required":[]
        }
    }
]