from langgraph.graph import END, StateGraph, MessagesState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from  langchain.messages import HumanMessage
load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant")

def generateNode(state: MessagesState) -> MessagesState:
    last_message = state["messages"][-1]
    print(f"Last message in GenerateNode: {last_message}")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def getReviewDecision(state:MessagesState)-> str:
    AIResponse = state["messages"][-1].content
    print(f"AI Response in getReviewDecision: {AIResponse}")
    print("Are you satisfied with the post content? (yes/no)")
    
    user_input = input().strip().lower()
    if user_input == "yes":
        return "postNode"
    else:
        return "collectFeedbackNode"

def collectFeedback(state: MessagesState) -> MessagesState:
    print("Please provide feedback on the post content:")
    feedback = input().strip()
    return {"messages":[HumanMessage(content=feedback)]}

def postNode(state: dict) -> MessagesState:
    print("Post content is approved and published!")
    return {}

graph = StateGraph(MessagesState)
graph.add_node("generateNode", generateNode)
graph.add_node("getReviewDecision", getReviewDecision)
graph.add_node("collectFeedbackNode", collectFeedback)
graph.add_node("postNode", postNode)

graph.add_conditional_edges(
    "generateNode",
    getReviewDecision,
    {
        "postNode": "postNode",
        "collectFeedbackNode": "collectFeedbackNode",
    },
)
graph.add_edge("collectFeedbackNode", "generateNode")
graph.add_edge("postNode", END)

graph.set_entry_point("generateNode")

if __name__ == "__main__":
    
    app = graph.compile()
    app.invoke({"messages": [HumanMessage(content="Write a social media post about the benefits of using LangGraph.")]})