import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
)

template = PromptTemplate (
    imput_variables = ["keyword"],
    template = "Think like a astrologer and give me the horoscope for {keyword} and marriage age and type of marriage"
)

chain = template | llm
response = chain.invoke({"keyword": "8 feb 2003, female"})
print(response.content)