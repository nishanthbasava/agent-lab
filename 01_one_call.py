#An agent is a conversation history (a list), a set of tools (functions with descritions), and a loop.
'''
The model never executes anything, it only ever emits requests to use tools. 
Your code is the hands.
Main question when you get a bug is "what exactly is in the message list right now"

'''

'''
A message list is a Python list of dicts. The messages = [...] you pass to the API.
Each entry has a role ("user" or "assistant") and content.
The API is stateless: the model remembers nothing between calls, so every call sends the entire history
again, and "the agent's memory" is just whatever's curently in that list

In the agent loop, four kinds of things get appended to it:
1. Your task -> a user message
2. The model's reply (text and/or tool requests) -> an assistant message
3. Tool results -> sent back inside a user message (that's just the API convention  - the "user" role here really means "
anythign coming from outside the model") 
4. ... and repeat

appending a tool result without the matching request, or the list silently growing until the context
window is full. Debugging an agent is mostly print(messages) and reading. 

Now - back to Lesson 1. Run your script and show me the output.
'''

from anthropic import Anthropic

client = Anthropic() # Reads ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    tools=[{"name":"read_file", "description":"opens a file", 
            "input_schema":{"type":"object", "properties": {"path": {"type":"string", "description":"path to the file to read"}}, "required": ["path"]}}],
    messages=[
        {"role": "user", "content": "What's in notes.txt"}
    ],
)

print(response.stop_reason)

#some of the blocks on response will be: tool-calls, text, or thinking
for block in response.content:
    print(block)

