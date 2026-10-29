from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()

llm = ChatGroq(
model="openai/gpt-oss-20b",
api_key=os.getenv("GROQ_API_KEY")
   )


system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with tasks."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="chat_history")
   ])

history =[]



def chat(user_prompt):
    message = HumanMessage(content=user_prompt)
    history.append(message)
    response = llm.invoke(system_prompt.invoke({"input": user_prompt, "chat_history": history}))
    history.append(AIMessage(content=response.content))
    return response.content


print(history)
print(chat("I am Ranjitha"))
print(history)
print(chat("who am I?"))
print(history)