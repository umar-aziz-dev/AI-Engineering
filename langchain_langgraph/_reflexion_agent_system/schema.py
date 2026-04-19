from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing:str = Field(description="Critique of the which is missing.")
    superflous:str = Field(description="Critique of the which is superfluous.")
    

class AnswerQuestion(BaseModel):
    answer:str = Field(description="Answer to the question.")
    reflection:Reflection = Field(description="Reflection on the answer.")
    search_queries:List[str] = Field(description="List of search queries 1-3 to improve the answer.")
    
class RevisedAnswer(BaseModel):
        response:str = Field(description="Revised answer to the question based on the reflection and search queries.")
        critique:str = Field(description="Critique of the revised answer.")
        search_queries:List[str] = Field(description="List of search queries 1-3 to further improve the answer.")
        citations:List[str] = Field(description="List of citations used in the revised answer.")    