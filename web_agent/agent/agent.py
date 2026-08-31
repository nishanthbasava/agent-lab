from anthropic import Anthropic
from web_agent.agent.tools import tools
from web_agent.retrieval.vector_store import collection
from web_agent.agent.tool_methods import calc_tool, wikipedia_page_fetcher, wikipedia_page_summarizer, wikipedia_search

client = Anthropic()

# maps the schema "name" string -> the actual callable
tool_functions = {
    "sympy_calc": calc_tool,
    "wikipedia_search": wikipedia_search,
    "wikipedia_page_summarizer": wikipedia_page_summarizer,
    "wikipedia_page_fetcher": wikipedia_page_fetcher
}

MAX_STEPS = 5
messages = []

def run_agent(user_message: str):

    steps = 0

    messages.append({"role":"user", "content": input("What is your question?: ")})

    while steps < MAX_STEPS:

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            tools=tools,
            messages = messages
        )

        #Adding Claude's response to context
        messages.append({"role":"assistant", "content":response.content})

        #Checking if model wants to use a tool
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    fn = tool_functions[block.name] # lookup happens here
                    result = fn(**block.input) # actual function from tool_methods.py runs
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role":"user", "content":tool_results})

        else:
            return "".join(b.text for b in response.content if b.type == "text")
        
        step += 1

    return "Agent stopped: reached max steps."

if __name__ == "__main__":
    print(run_agent(input("What is your question?: ")))

    #I still need to fix user_message and MESSAGES, which is still module-level
    #what does "module-level" mean
    #collection is currently unused

    #one forward-looking note, not a bug: tool_functions[block.name] will KeyError if the model ever
    #hallucinates a tool name. Rare, but the robust pattern is to check membership and send back a 
    #tool_result with "is_error": True saying the tool doesn't exist - the model will correct itself. Fine to defer. 

    #afterwards expand the environment and add some reinforcement learning 
    #play around with different agent memory architectures like maybe a vector DB for semantic search?