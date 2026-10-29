from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

data = [
    {"positive":"Your dress is very beautiful"},
    {"negative":"Your dress is not very beautiful"},
    {"positive":"Your are a very good person"},
    {"negative":"Your are a very bad person"},
    {"neutral":"Your are a very neutral person"},
]

new_data =[
    "You are good",
    "I don't like you",
    "I am neutral",
    "You are bad"
]

categorize_data = PromptTemplate(
    input_variables=["data", "new_data"],
    template="""
    You are a helpful assistant that can categorize data.
    Reference data: {data}
    New data to categorize: {new_data}
    """
)

summarize_count = PromptTemplate(
    input_variables=["categorize"],
    template="""
    You are a helpful assistant that can summarize the categorized data and give a response accordingly.
    {categorize}
    """
)

chain1 = categorize_data | llm
response1 = chain1.invoke({"data": data, "new_data": new_data})

print("--- Categorization Response ---")
print(response1.content)

chain2 = summarize_count | llm
response2 = chain2.invoke({"categorize": response1.content})

print("\n--- Summary Response ---")
print(response2.content)