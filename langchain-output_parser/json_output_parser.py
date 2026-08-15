from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
output_parser = JsonOutputParser()

template = PromptTemplate(
    template="write the all character name of the movie {movie} \n {format_instruction}",
    input_variables=["movie"],
    partial_variables={"format_instruction":output_parser.get_format_instructions()}
)

prompt = template.invoke({"movie":"Game of thrones"})

result = model.invoke(prompt)
parsed = output_parser.parse(result.text)
print(type(parsed))

