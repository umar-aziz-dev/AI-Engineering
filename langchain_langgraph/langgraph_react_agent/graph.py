

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from schema import GraphState
import json
from langgraph.graph import StateGraph, END


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, max_tokens=None)


search_tool = TavilySearch(max_results=5)


@tool
def get_system_time()-> str:
    """GET current system time"""
    import datetime
    return datetime.datetime.now().isoformat()


 
tools = [get_system_time, search_tool]

tools_dict = {
    tool.name: tool
    for tool in tools
}

llm_with_tools = llm.bind_tools([get_system_time, search_tool], tool_choice="auto")

def agent_node(state):
    step = state.get("step_count", 0) + 1
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "tool_calls": response.tool_calls or [],
        "step_count": step
    }

def should_continue(state):
    if state["step_count"] > 5:
        return "end"

    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "tools"

    return "end"


def tool_node(state: GraphState):
    tool_calls = state.get("tool_calls", [])
    results = []
    tool_messages = []

    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]

        result = tools_dict[tool_name].invoke(args)
        results.append(result)
        
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
                name=tool_name,
            )
        )

    return {
        "messages": tool_messages,
        "tool_calls": [],
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
    "messages": [HumanMessage(content="What is the current system time and search for latest news on AI?")]
})

print(response)

# Write response to a file
with open("response.txt", "w") as f:
    f.write(json.dumps(response["messages"][-1].content, indent=4))
    
# Full resonse json in file response.json
import json


def to_json_safe(obj):
    if isinstance(obj, (AIMessage, HumanMessage, ToolMessage)):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {key: to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(item) for item in obj]
    return obj


with open("response.json", "w") as f:
    json.dump(to_json_safe(response), f, indent=4)