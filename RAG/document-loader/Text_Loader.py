from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
loader = TextLoader(
    file_path='../data/transcript.txt',
    encoding='utf-8'
)

docs = loader.load()
transcript = docs[0].page_content
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
prompt = PromptTemplate(
    template="What is the meaning of the given transcript of the youtube video. please explain in point wise. \n Transcript = {transcript}",
    input_variables=["transcript"],
)

chain = prompt | model | StrOutputParser()
result = chain.invoke({"transcript": transcript})
print(result)

