
from typing import Annotated

from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls: NotRequired[list[dict]]
    tool_results: NotRequired[list]
    step_count: NotRequired[int]