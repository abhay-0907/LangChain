from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
prompt_template = PromptTemplate(
    template="Give me 5 interesting facts about {topic}", input_variables=["topic"]
)
output_parser = StrOutputParser()

chain = prompt_template | model | output_parser

result = chain.invoke("Space")
print(result)

chain.get_graph().print_ascii()