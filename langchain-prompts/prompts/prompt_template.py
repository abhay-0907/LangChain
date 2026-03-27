from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.5);




topic = str(input("Enter the topic: "))
language = str(input("choose the langauge. eg. english/hindi/french/german: "))
level = str(input("Enter the level. eg. beginner/advanced: "))

prompt_template = load_prompt("prompt.json")

prompt = prompt_template.invoke({ 
    "topic":topic,
    "language": language ,
    "level":level
})

output = model.invoke(prompt)
print(f"Output: {output.content}")