from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

file_path = "../data/ThinkStraight.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)

docs = text_splitter.split_documents(docs)

for i in range(len(docs)):
    print(docs[i].page_content)