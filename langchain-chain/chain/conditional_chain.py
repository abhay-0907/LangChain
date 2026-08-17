from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found — check your .env file")

llm = ChatOpenAI()

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the text")

output_parser = PydanticOutputParser(pydantic_object=Sentiment)
str_parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="Find the sentiment of the given product feedback. Feedback: {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":output_parser.get_format_instructions()}
)

classifier_chain = prompt1 | llm | output_parser


prompt2 = PromptTemplate(
    template="Write the appropiate response of this positive product feedback \n Feedback {feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
    template="Write the appropiate response of the negative product feedback \n Feedback:{feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda X: X.sentiment == "positive", prompt2 | llm | str_parser),
    (lambda X: X.sentiment == "negative", prompt3 | llm | str_parser),
    RunnableLambda(lambda X : "Couldn't find the sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback":"Best $1,600 I've ever spent on a pocket heater. The battery life is truly unmatched—I've never seen anything drain this fast while doing absolutely nothing. Love the premium design; it looks absolutely stunning sitting on the charger for the third time today"})

print(result)


