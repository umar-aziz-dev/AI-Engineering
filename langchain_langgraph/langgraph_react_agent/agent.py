from pydantic import BaseModel, Field



# Pydantic models for input and output of tools
class SearchOutput(BaseModel):
    """ Search results from Tavily Search tool"""
    city:str = Field(description="City name")
    temperature: float = Field(description="Temperature in Celsius")
    
