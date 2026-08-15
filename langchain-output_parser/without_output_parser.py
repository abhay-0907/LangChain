from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

t1 = PromptTemplate(
    template="Generate the detailed report on the given {topic}.\n The report should be contain minimum 500 words.",
    input_variables=["topic"]
)
t2 = PromptTemplate(
    template="summarize the given {text} into 5 points",
    input_variables=["text"]
)

prompt1 = t1.invoke({"topic":"Big Bang Theory"})

result = llm_model.invoke(prompt1)

prompt2 = t2.invoke({"text":result.text})

new_result = llm_model.invoke(prompt2)

print("Detailed result\n",result.text)
print("Summarize result\n",new_result.text)

