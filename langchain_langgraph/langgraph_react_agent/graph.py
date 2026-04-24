
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
from langchain.tools import tool

try:
    from .schema import GraphState
except ImportError:
    from schema import GraphState

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


logger = logging.getLogger(__name__)

MAX_STEPS = 6
SYSTEM_PROMPT = (
    "You are a concise ReAct assistant. Use tools only when needed, "
    "and if tool results are sufficient, provide a direct final answer."
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_tokens=None)


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


def agent_node(state: GraphState) -> GraphState:
    step = state.get("step_count", 0) + 1
    response = llm_with_tools.invoke(state.get("messages", []))

    return {
        "messages": [response],
        "tool_calls": response.tool_calls or [],
        "step_count": step
    }


def should_continue(state: GraphState) -> str:
    if state.get("step_count", 0) >= MAX_STEPS:
        return "end"

    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_msg = messages[-1]

    if getattr(last_msg, "tool_calls", None):
        return "tools"

    return "end"


def tool_node(state: GraphState):
    tool_calls = state.get("tool_calls", [])
    results = []
    tool_messages = []

    for call in tool_calls:
        tool_name = call.get("name", "")
        args = call.get("args", {})

        if tool_name not in tools_dict:
            result = {"error": f"Unknown tool '{tool_name}'", "args": args}
        else:
            try:
                result = tools_dict[tool_name].invoke(args)
            except Exception as exc:
                logger.exception("Tool execution failed: %s", tool_name)
                result = {"error": f"Tool execution failed for '{tool_name}'", "details": str(exc)}

        results.append(result)
        
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call.get("id", "missing_tool_call_id"),
                name=tool_name,
            )
        )

    return {
        "messages": tool_messages,
        "tool_calls": [],
        "tool_results": results
    }
    
    

def build_graph():
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
    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


def to_json_safe(obj: Any):
    if isinstance(obj, (AIMessage, HumanMessage, ToolMessage)):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {key: to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(item) for item in obj]
    return obj


def run_demo() -> dict:
    return graph.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="What is the current system time and search for latest news on AI?")
            ],
            "step_count": 0,
        },
        config={"configurable": {"thread_id": "demo-react-thread"}},
    )


def persist_response(response: dict) -> None:
    with open("response.txt", "w") as f:
        f.write(json.dumps(response["messages"][-1].content, indent=4))

    with open("response.json", "w") as f:
        json.dump(to_json_safe(response), f, indent=4)


if __name__ == "__main__":
    response = run_demo()
    print(response)
    persist_response(response)