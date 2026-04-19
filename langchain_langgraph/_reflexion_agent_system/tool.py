
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_tavily import TavilySearch
import json

tavily_tool = TavilySearch(max_results=5)

def execute_tools(state: dict) -> List[BaseMessage]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
  
  
    search_queries = []
    tool_messages = []
    for tool_call in tool_calls:
        if tool_call.get("name") in ["AnswerQuestion", "RevisedAnswer"] or tool_call.get("tool_name") in ["AnswerQuestion", "RevisedAnswer"]:
            call_id = tool_call.get("id", "unknown_id")
            search_queries = tool_call.get("args", {}).get("search_queries", [])
            
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result
                
            tool_messages.append(ToolMessage(
                tool_call_id=call_id,
                content=json.dumps(query_results),
            ))
    return tool_messages