from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()
from langgraph.graph import add_messages,StateGraph,END
from langchain_core.messages import HumanMessage
# Gpt 4o mini   
llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChat(TypedDict):
    messages: Annotated[list, add_messages]


def chatNode(state: BasicChat) -> str:
    last_message = state["messages"][-1]
    response = llm.invoke(last_message.content)
    return {"messages": [response]}

graph = StateGraph(BasicChat)
graph.add_node("chat", chatNode)
graph.set_entry_point("chat")

graph.add_edge("chat", END)

if __name__ == "__main__":
    app = graph.compile()
    
    while(True):
        userMessage = input("User: ")
        if userMessage.lower() == "exit":
            break
        response = app.invoke({"messages": [HumanMessage(content=userMessage)]})
        print(f"AI: {response['messages'][-1].content}")
        