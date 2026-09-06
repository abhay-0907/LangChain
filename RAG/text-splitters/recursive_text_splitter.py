from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

file_path = '../data/ThinkStraight.pdf'

docs = PyPDFLoader(file_path).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)


result = (text_splitter.split_documents(docs))    
print(result)