from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.5);


#! without message history
'''while True:
    user_input = str(input("User: "))
    if user_input == "exit":
        break
    output = model.invoke(user_input)
    print(f"AI: {output.content}")'''


#? Using message history with the help of message class from langchain

message = [
    SystemMessage(content="You are a helpful assistant."),
]
while True:
    user_input = str(input("User: "))
    if(user_input == "exit"):
        break
    message.append(HumanMessage(content=user_input))
    output = model.invoke(message)
    message.append(AIMessage(content=output.content))
    print(f"AI: {output.content}")