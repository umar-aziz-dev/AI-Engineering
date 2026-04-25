from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()
from langgraph.graph import add_messages,StateGraph,END
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
# Gpt 4o mini   
searchTool = TavilySearch()
llm = ChatGroq(model="llama-3.1-8b-instant")
llmWithTools = llm.bind_tools(tools=[searchTool])
class BasicChat(TypedDict):
    messages: Annotated[list, add_messages]


def chatNode(state: BasicChat) -> str:
    last_message = state["messages"][-1]
    print(f"Last message in ChatNode: {last_message}")
    response = llmWithTools.invoke(state["messages"])
    return {"messages": [response]}

def toolsRouter(state: BasicChat):
    last_message = state["messages"][-1]    
    print(f"Last message in toolsRouter: {last_message}")
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
    app = graph.compile()
    
    while(True):
        userMessage = input("User: ")
        if userMessage.lower() == "exit":
            break
        response = app.invoke({"messages": [HumanMessage(content=userMessage)]})
        print(f"AI: {response['messages'][-1].content}")
        