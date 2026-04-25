from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import add_messages,StateGraph,END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Mongodb
DB_URI = "localhost:27017"

from langgraph.checkpoint.mongodb import MongoDBSaver  

    
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





if __name__ == "__main__":
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        while(True):
            userMessage = input("User: ")
            if userMessage.lower() == "exit":
                break
            response = app.invoke({"messages": [HumanMessage(content=userMessage)]}, {"configurable":{"thread_id":"1"}})
            print(f"AI: {response['messages'][-1].content}")
            print(f"Checkpointer: {checkpointer}")
            print(f"Response: {response}")
