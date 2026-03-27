from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

model = OpenAI(model="gpt-4o-mini")

def dynamic_prompt():
    print("What do I for you...")
    input = str(input("Enter the prompt "))
    prompt = f"Read the user query very carefully, query is {input} and then give the ouput."
    output = model.invoke(prompt)
    print(f"Output: {output}")
    return

dynamic_prompt()
