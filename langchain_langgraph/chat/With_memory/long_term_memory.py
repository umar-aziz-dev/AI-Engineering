from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, MessagesState, START
import uuid
from langgraph.store.memory import InMemoryStore  

load_dotenv()


@dataclass
class Context:
    user_id: str


llm = ChatGroq(model="llama-3.1-8b-instant")

def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else last_message["content"]

    # Search for relevant memories
    memories = runtime.store.search(
        namespace, query=query, limit=3
    )
    info = "\n".join(d.value["data"] for d in memories)

    memory_context = info if info else "No relevant long-term memory found."
    system_prompt = (
        "You are a helpful assistant. Use the long-term memory context when relevant. "
        "If memory conflicts with the current user message, ask a brief clarifying question.\n\n"
        f"Long-term memory:\n{memory_context}"
    )

    response = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
    
    
    # Store the latest user utterance as a simple long-term memory entry.
    runtime.store.put(
        namespace, str(uuid.uuid4()), {"data": query}
    )
    return {"messages": [response]}
store = InMemoryStore()

builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(store=store)

# Pass context at invocation time
app = graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    {"configurable": {"thread_id": "1"}},
    context=Context(user_id="1"),
)

print(app)