from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


template = PromptTemplate(
    input_variables=["question"],
    template=""" Give me a step by step solution to the question: {question}"""
)

chain = template | llm
response = chain.invoke({"question": "Approach to learning Docker and Kubernetes"})
print(response.content)