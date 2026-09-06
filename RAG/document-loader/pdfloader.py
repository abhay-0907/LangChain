from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

file_path = '../data/ThinkStraight.pdf'

loader = PyPDFLoader(file_path)
mu_loader = PyMuPDFLoader(file_path)

docs = loader.load()
docs1 = mu_loader.load()

print(docs)
print(docs1)
