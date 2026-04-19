
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_tavily import TavilySearch
from schema import AnswerQuestion
import json

tavily_tool = TavilySearch(max_results=5)

def execute_tools(state: dict) -> List[BaseMessage]:
    last_output: AnswerQuestion = state["messages"][-1]

    search_queries = last_output.search_queries

    if not search_queries:
        return []

    query_results = []

    for query in search_queries:
        result = tavily_tool.invoke(query)
        query_results.append(result)

    return [
        ToolMessage(
            content=json.dumps(query_results),
        )
    ]