# Command make us able to flow without edges 
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.types import Command , interrupt
from langchain.messages import HumanMessage
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

class CustomState(TypedDict):
    value: str = ""

def nodeA(state: CustomState) -> Command:
    print("Executing Node A")
    return Command(goto="nodeB", update={"value":state["value"] + "A"})

def nodeB(state: CustomState) -> Command:
    print("Executing Node B")
    return Command(goto="nodeC", update={"value":state["value"] + "B"})

def nodeC(state: CustomState) -> Command:
    print("Executing Node C")
    humanResponse = interrupt("Please provide your input for Node C: D/E")
    if humanResponse.strip().upper() == "D":
        return Command(goto="nodeD", update={"value":state["value"] + "C-D"})
    else:
        return Command(goto="nodeE", update={"value":state["value"] + "C-E"})


def nodeD(state: CustomState) -> Command:
    print("Executing Node D")
    return Command(goto=END, update={"value":state["value"] + "D"})

def nodeE(state: CustomState) -> Command:
    print("Executing Node E")
    return Command(goto=END, update={"value":state["value"] + "E"})



graph = StateGraph(CustomState)

graph.add_node("nodeA", nodeA)
graph.add_node("nodeB", nodeB)
graph.add_node("nodeC", nodeC)
graph.add_node("nodeD", nodeD)  
graph.add_node("nodeE", nodeE)

graph.set_entry_point("nodeA")

if __name__ == "__main__":
    config ={
        "configurable": {
            "thread_id": "1"
        }
    }
    app = graph.compile(checkpointer=memory)
    resumed_response = app.invoke({"value": ""}, config=config,stream_mode="updates")
    print(f"Resumed state: {resumed_response}")
    
    print(app.get_state(config=config))
    # Get the value from frontend , and then resume the flow with that value
    user_input = input("Enter value to resume (D/E): ")
    final_response = app.invoke(Command(resume=user_input), config=config)
    
    print(f"Final State after resuming: {final_response}")
    