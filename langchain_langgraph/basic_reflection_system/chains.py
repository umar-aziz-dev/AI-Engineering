from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI  
from dotenv import load_dotenv
load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a twitter influencer assistant tasked with writing excellent tweets."),
    MessagesPlaceholder("messages")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You improve tweets and rewrite them better."),
    MessagesPlaceholder("messages")
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, max_tokens=None)

# result = llm.invoke(generation_prompt.format_messages(user_input="What is the capital of France?"))

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm