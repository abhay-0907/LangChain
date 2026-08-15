# this method has been removed from langchain v1
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser
# from dotenv import load_dotenv
# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# schemas = [
#     ResponseSchema(name="title of the research paper",description="write the abstract of the paper"),
#     ResponseSchema(name="Dataset Used",description="which dataset used in the research paper (if exist or mention on the paper)"),
#     ResponseSchema(name="Methodology",description="Write the methodologies used in the paper"),
#     ResponseSchema(name="citation",description="Write the citation of the paper")
# ]

# output_parser = StructuredOutputParser.from_response_schemas(schemas)


# format_instructions = output_parser.get_format_instructions()

# template = PromptTemplate(
#     template="find the best research paper on the {topic}.\n{format_instructions",
#     input_variables=["topic"],
#     partial_variables={"format_instructions": format_instructions}
# )

# prompt = template.invoke({"topic":"Brain Tumor Detection using Deep learning"})

# result = llm.invoke(prompt)


# final_result = output_parser.parse(result.text)

# print(final_result)
