from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class Car(BaseModel):
    model : str = Field(description="Model of the cars")
    seats : int = Field(description="Seats in the each car")
    car_type : str = Field(description="Type of the car electric, petrol or Diesal")
    top_speed : int = Field(ge=270,description="Top speed of the car should be at least 270 KM/hr")
    price : float = Field(description="Price of the car in indian rupees")


output_parser = PydanticOutputParser(pydantic_object=Car)



format_instructions = output_parser.get_format_instructions()

template = PromptTemplate(
    template="find the car of the given {company}.\n{format_instructions}",
    input_variables=["company"],
    partial_variables={"format_instructions": format_instructions}
)

prompt = template.invoke({"company":"tesla"})

result = llm.invoke(prompt)

final_result = output_parser.parse(result.text)

print(final_result)
