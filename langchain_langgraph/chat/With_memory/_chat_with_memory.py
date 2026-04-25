from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import add_messages,StateGraph,END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver



class BasicChat(TypedDict):
    messages: Annotated[list, add_messages]


memory = InMemorySaver()
llm = ChatGroq(model="llama-3.1-8b-instant")

def chatNode(state: BasicChat) -> str:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(BasicChat)
graph.add_node("chat", chatNode)
graph.set_entry_point("chat")
graph.add_edge("chat", END)



app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    while(True):
        userMessage = input("User: ")
        if userMessage.lower() == "exit":
            break
        response = app.invoke({"messages": [HumanMessage(content=userMessage)]}, {"configurable":{"thread_id":"1"}})
        print(f"AI: {response['messages'][-1].content}")    

