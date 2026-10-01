# Benefits of using LangChain

# Persistence: LangGraph saves a checkpoint of the workflow's progress as it runs, using a checkpointer (in-memory, SQLite, Postgres, etc). 
# If the process crashes or an API call fails, you can resume from the last checkpoint instead of starting over. Completed @task results are stored,
# so they aren't re-executed. Each run is tied to a thread_id, which is how LangGraph knows which saved state to load.

# Persistence means saving your workflow's progress to storage as it runs, so it isn't lost if something stops.


# Memory: 
# Short-term memory is the state within one thread (conversation). In the Functional API, you add a previous parameter to your entrypoint to get the prior run's 
# return value, so a chatbot can see earlier tunrs. This relies on persistence.
#


# Human-in-the-Loop
# Streaming

from langgraph.func import entrypoint, task

@task
def summarize(doc: str) -> str:
    return llm.invoke(f"Summarize: {doc}").content

@entrypoint()
def workflow(docs: list[str]) -> list[str]:
    futures = [summarize(d) for d in docs]   # all start immediately
    return [f.result() for f in futures]     # then wait for results



#Decorators are just fancy syntax for a function call + reassignment

@entrypoint()
def my_workflow(inputs): ...

# is exactly the same as:
def my_workflow(inputs): ...
my_workflow = entrypoint()(my_workflow)   # reassignment

