from anthropic import Anthropic
from tools import tools
from vector_store import collection

client = Anthropic()
steps = 0
MAX_STEPS = 5
messages = []

messages.append({"role":"user", "content": input("What is your question?: ")})

while steps < MAX_STEPS:

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        messages = messages
    )

    #Adding Claude's response to context
    messages.append({"role":"assistant", "content":response.content})

    #Checking if model wants to use a tool
    if response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use"





    for block in response.content:
        print(block.text) 




    

#Write the actual agent loop here
#afterwards expand the environment and add some reinforcement learning 
#play around with different agent memory architectures like maybe a vector DB for semantic search?