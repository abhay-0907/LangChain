from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatAnthropic(model="claude-2-100k", temperature=0.2)
result = chat_model.invoke("Hello, how are you?")