from typing import TypedDict, List
from dotenv import load_dotenv
from chains import reflection_chain, generation_chain
from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# # ✅ Define state
# class State(TypedDict):
#     messages: List

graph = StateGraph(MessagesState)

REFLECT = "reflect"
GENERATE = "generate"

# ✅ Nodes now receive full state
def generate_node(state: MessagesState):
    return generation_chain.invoke({"messages": state["messages"]})


def reflect_node(state: MessagesState):
    result = reflection_chain.invoke({"messages": state["messages"]})
    return {
       "messages": [HumanMessage(content=result.content)]
    }
        

# ✅ Add nodes
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# (you'll also need edges — see below)

graph.set_entry_point(GENERATE)


def should_continue(state: MessagesState) :
    if(len(state["messages"]) > 2):
        return "end"
    return "reflect"


graph.add_conditional_edges(GENERATE, should_continue,{
        "reflect": REFLECT,
        "end": END
    })
graph.add_edge(REFLECT, GENERATE)


app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()


app.invoke({
    "messages": [
        HumanMessage(content="Write a tweet about the benefits of meditation. no more than 40 words.")
    ]})

