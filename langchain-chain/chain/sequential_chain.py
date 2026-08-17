from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Give me the detailed information of the given {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the given {text} in 5 point ", input_variables=["text"]
)

model = OpenAI()

output_parser = StrOutputParser()

chain = prompt1 | model | output_parser | model | output_parser
result = chain.invoke({"topic":"Unemployment in India"})

print(result)
