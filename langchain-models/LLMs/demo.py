import langchain 
from dotenv import load_dotenv;
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(model='gpt-4o-mini');

result = llm.invoke("What is LangChain?.")
print(f"Result: {result}");