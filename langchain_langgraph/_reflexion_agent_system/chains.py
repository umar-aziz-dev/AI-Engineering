
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import datetime
load_dotenv()
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from schema import AnswerQuestion, RevisedAnswer
from langchain_core.output_parsers import PydanticOutputParser


actor_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are expert AI researcher.
     Current time: {time}
     
     # Instructions
     1- {first_instruction}
     2- Reflect and critique your answer, Be serve to maximize the quality of your answer.
     3- After the reflection, **list 1-3 search queries separately** that you would use to search for more information to improve your answer.Do not include them inside the reflection.
     """),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question above using the required format.")
]).partial(
    time=datetime.datetime.now()
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, max_tokens=None)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide detailed ~200 words answer."
)

# Parser for the structured output
# parser = PydanticOutputParser(pydantic_object=AnswerQuestion) # in case where directly parse by llm raw text


structured_llm = llm.with_structured_output(AnswerQuestion)
first_reponseder_chain = first_responder_prompt_template | structured_llm

revise_iinstruction = """
Now, based on the reflection and search queries, revise your answer to improve it. Provide a more detailed and accurate response. Also, provide a critique of your revised answer and list 1-3 search queries separately that you would use to further improve your answer. Finally, include a list of citations used in the revised answer.
"""



# revised_structured_llm = llm.with_structured_output(RevisedAnswer)

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_iinstruction
) | llm.with_structured_output(RevisedAnswer)
 
 

# Response with call 
response = first_reponseder_chain.invoke({"messages": [("user", "Write me a blog post on how to write a blog post as a software engineer to seem like good SE having depth knowledge of System Design etc?")]})

print(response)