import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

template = PromptTemplate(
    input_variables = ["Tone", "content", "language", "Audience"],
    template = "You are a {Tone} writer. Write a {content} in {language} for {Audience}"
)

chain = template | llm
response = chain.invoke({"Tone": "friendly", "content": "Philosophy", "language": "English", "Audience": "students"})
print(response.content)