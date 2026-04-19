
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_tavily import TavilySearch
import json

tavily_tool = TavilySearch(max_results=5)

def execute_tools(state: dict) -> List[BaseMessage]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    if not tool_calls and getattr(last_message, "additional_kwargs", None):
        tool_calls = last_message.additional_kwargs.get("tool_calls", []) or []

    search_queries = []
    if tool_calls:
        tool_args = tool_calls[0].get("args", {}) or {}
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        search_queries = tool_args.get("search_queries", []) or []

    print(f"Tool calls: {tool_calls}")
    print(f"Search queries: {search_queries}")
    if not search_queries:
        return{"messages": [
            ToolMessage(
                tool_call_id=tool_calls[0]["id"] if tool_calls else "tavily_search",
                content=json.dumps([]),
            )
        ]}

    query_results = []

    for query in search_queries:
        result = tavily_tool.invoke(query)
        query_results.append(result)

    return{"messages": [
        ToolMessage(
            tool_call_id=tool_calls[0]["id"] if tool_calls else "tavily_search",
            content=json.dumps(query_results),
        )
    ]}