
from typing import TypedDict, List, Optional , Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_calls: Optional[list]
    tool_results: Optional[list]
    step_count: int