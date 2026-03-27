from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model = OpenAI(model="gpt-4o-mini")
def static_prompt(prompt):
    output = model.invoke(prompt)
    print(f"Output: {output}")
    return

input = str(input("Enter a prompt: "))
static_prompt(input);