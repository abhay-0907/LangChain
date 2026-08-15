from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

output_parser = StrOutputParser()

t1 = PromptTemplate(
    template="Write the pseudo code of the algorithm {algo}",
    input_variables=["algo"]
)

t2 = PromptTemplate(
    template="""Convert the following pseudo code into {lang}:

{pseudo_code}
""",
    input_variables=["lang", "pseudo_code"]
)

# Step 1: Create first prompt
prompt1 = t1.invoke({
    "algo": "Radix Sort"
})

# Step 2: Send prompt to LLM
op1 = llm.invoke(prompt1)

# Step 3: Parse AIMessage → string
parsed_result1 = output_parser.invoke(op1)

# Step 4: Create second prompt using parsed string
prompt2 = t2.invoke({
    "lang": "Rust and Ruby",
    "pseudo_code": parsed_result1
})

# Step 5: Send second prompt to LLM
op2 = llm.invoke(prompt2)

# Step 6: Parse AIMessage → string
parsed_result2 = output_parser.invoke(op2)

print(parsed_result1)
print(parsed_result2)