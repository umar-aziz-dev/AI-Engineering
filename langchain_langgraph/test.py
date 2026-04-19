from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

tavilyTool = TavilySearch(max_results=5)

result = tavilyTool.invoke("What is the weather of Lahore?")

print(result)