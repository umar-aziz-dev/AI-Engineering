# Command make us able to flow without edges 
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.types import Command
from langchain.messages import HumanMessage

def nodeA(state: MessagesState) -> Command:
    print("Executing Node A")
    return Command(goto="nodeB", update={"messages": [HumanMessage(content="Message from Node A")]})


