from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_langgraph.langgraph_react_agent.schema import GraphState
load_dotenv()
from langchain.agents import create_agent 
from langchain.tools import tool 
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing_extensions import Type,TypedDict,Annotated


# Pydantic models for input and output of tools
class SearchOutput(BaseModel):
    """ Search results from Tavily Search tool"""
    city:str = Field(description="City name")
    temperature: float = Field(description="Temperature in Celsius")
    

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, max_tokens=None)

