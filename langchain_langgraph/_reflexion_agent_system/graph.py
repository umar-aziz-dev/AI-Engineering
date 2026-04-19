from langgraph.graph import END, StateGraph , MessagesState
from chains import first_reponseder_chain, revisor_chain
from tool import execute_tools
from langchain_core.messages import HumanMessage, ToolMessage


graph = StateGraph(MessagesState)

DRAFT = "draft"
EXECUTE_TOOL= "execute_tool"
REVISOR= "revisor"
MAX_ITERATIONS = 2


# Nodes:

graph.add_node(
    "draft",
    lambda state: {"messages": [first_reponseder_chain.invoke({"messages": state["messages"]})]},
)
graph.add_node(EXECUTE_TOOL, execute_tools)
graph.add_node(
    REVISOR,
    lambda state: {"messages": [revisor_chain.invoke({"messages": state["messages"]})]},
)


# Entry point
graph.set_entry_point(DRAFT)

# Edges: 
def should_execute_tool(state: MessagesState):
    count_tool_visits = sum(isinstance(message,ToolMessage) for message in state["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOL


graph.add_edge(DRAFT, EXECUTE_TOOL)
graph.add_edge(EXECUTE_TOOL, REVISOR)
graph.add_conditional_edges(REVISOR, should_execute_tool, {
    EXECUTE_TOOL: EXECUTE_TOOL,
    END: END
})

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke({
    "messages": [
        HumanMessage(content="Write me a blog post on how to write a blog post as a software engineer to seem like good SE having depth knowledge of System Design etc?")
    ]})


print(response["messages"][-1].content)