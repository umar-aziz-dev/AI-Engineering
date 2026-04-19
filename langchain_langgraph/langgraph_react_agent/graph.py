

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langchain_langgraph._reflexion_agent_system import tool
from langchain_langgraph.langgraph_react_agent.schema import GraphState

from langgraph.graph import StateGraph, END


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, max_tokens=None)


search_tool = TavilySearch(max_results=5)


@tool
def get_system_time()-> str:
    """GET current system time"""
    import datetime
    return datetime.datetime.now().isoformat()


 
tools_dict = {
    "TavilySearch": search_tool,
    "get_system_time": get_system_time
}
def agent_node(state: GraphState):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else None
    }
def should_continue(state: GraphState):
    
    # Complete control logic
    
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return "end"



def tool_node(state: GraphState):
    tool_calls = state.get("tool_calls", [])
    results = []

    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]

        result = tools_dict[tool_name].invoke(args)
        results.append(result)

    return {
        "messages": [ToolMessage(content=str(results))],
        "tool_results": results
    }
    
    

builder = StateGraph(GraphState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

builder.add_edge("tools", "agent")

graph = builder.compile()

response = graph.invoke({
    "messages": [HumanMessage(content="What is LangGraph?")]
})