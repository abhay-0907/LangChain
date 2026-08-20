from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

output_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Tell me the summary of the book and in the end you will give the real life example based on that book essence.\n The book is: {book_name}",
    input_variables=["book_name"]
)

prompt2 = PromptTemplate(
    template="You are a great literaturist and novelist. please check the summary and example is right according to the book name.\n the book name is: {summary}",
    input_variables=['summary']
)


chain = RunnableSequence(prompt1,model,output_parser,prompt2,model,output_parser)

result = chain.invoke({"book_name":" IKIGAI"})
print(result)