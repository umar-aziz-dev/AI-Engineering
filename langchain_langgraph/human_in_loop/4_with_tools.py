from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()
from langgraph.graph import add_messages,StateGraph,END
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
# Gpt 4o mini   
searchTool = TavilySearch()
llm = ChatGroq(model="llama-3.1-8b-instant")
llmWithTools = llm.bind_tools(tools=[searchTool])
class BasicChat(TypedDict):
    messages: Annotated[list, add_messages]


def chatNode(state: BasicChat) -> str:
    response = llmWithTools.invoke(state["messages"])
    return {"messages": [response]}

def toolsRouter(state: BasicChat):
    last_message = state["messages"][-1]    
    if (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return "toolNode"
    else:        
        return END
    
toolNode = ToolNode(tools=[searchTool])

graph = StateGraph(BasicChat)
graph.add_node("chat", chatNode)
graph.add_node("toolNode", toolNode)
graph.add_conditional_edges("chat", toolsRouter)
graph.add_edge("toolNode", "chat")
graph.set_entry_point("chat")

graph.add_edge("chat", END)



if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    app = graph.compile(checkpointer=memory, interrupt_before=["toolNode"])
    print(app.get_graph().draw_mermaid())
    app.get_graph().print_ascii()
    question = input("Enter your question for the AI (or 'exit' to quit): ")
    events = app.stream({
        "messages": [HumanMessage(content=question)]
    },config=config,stream_mode="values")

    for event in events:
        event["messages"][-1].pretty_print()

    # Resume whenever execution is paused before toolNode.
    while app.get_state(config=config).next:
        state = app.get_state(config=config)
        last_message = state.values["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if tool_calls:
            print("\nInterrupted before tool execution. Pending tool call(s):")
            for call in tool_calls:
                print(f"- {call['name']}: {call.get('args', {})}")

        approval = input("Approve tool execution? (yes/no): ").strip().lower()
        if approval != "yes":
            print("Tool execution denied by user. Exiting.")
            break

        resume_events = app.stream(None, config=config, stream_mode="values")
        for event in resume_events:
            event["messages"][-1].pretty_print()
