from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

message = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thank you for using the chatbot!")
        break
    message.append(HumanMessage(content=user_input))
    response = llm.invoke(message)
    message.append(AIMessage(content=response.content))
    print("Bot: ", response.content)
    print(message)