from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing_extensions import Type,TypedDict,Annotated

# Pydantic models for input and output of tools
class SearchOutput(BaseModel):
    """ Search results from Tavily Search tool exact json format"""
    city:str = Field(description="City name")
    temperature: float = Field(description="Temperature in Celsius")
    
    
# class SearchOutput(TypedDict): # When no validation required, you can also use TypedDict
#     """ Search results from Tavily Search tool"""
#     city:Annotated[str, Field(description="City name")]
#     temperature: Annotated[float, Field(description="Temperature in Celsius")]  

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
    response_format=SearchOutput
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather of Lahore?"}
    ]
})

content = result["messages"][-1].content
structured_output = result["structured_response"]

print(result["messages"][-1].content)
print(structured_output)
# result = llm.invoke("What is the capital of France?")

# print(result)