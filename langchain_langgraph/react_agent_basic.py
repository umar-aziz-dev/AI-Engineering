from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, max_tokens=None)
search_tool = TavilySearch(max_results=5)
@tool
def get_system_time()-> str:
    """GET current system time"""
    import datetime
    return datetime.datetime.now().isoformat()



agent = create_agent(
    model=llm,
    tools=[search_tool, get_system_time],   

)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather of Lahore?"}
    ]
})

print(result["messages"][-1].content)
print(result)
# result = llm.invoke("What is the capital of France?")

# print(result)