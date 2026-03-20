from langchain_openai import ChatOpenAI;

from dotenv import load_dotenv;

load_dotenv();

chat_model = ChatOpenAI(model='gpt-4o-mini');

result = chat_model.invoke("what is the capital of Norwat country.")
print(f"Result: {result.content}");