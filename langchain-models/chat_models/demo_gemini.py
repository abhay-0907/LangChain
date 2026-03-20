from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv;

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.5);

result = model.invoke("Tell me 5 creative ideas. Give me the answer in point wise");

print(f"Result: {result.content}");