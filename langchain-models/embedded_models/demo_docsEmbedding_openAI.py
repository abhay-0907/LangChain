from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv;

load_dotenv();

docs_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=128
)
documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning models learn patterns from data.",
    "LangChain helps build applications with LLMs.",
    "Vector embeddings convert text into numerical representations.",
    "OpenAI provides powerful embedding models like text-embedding-3-small."
]
resulted_embeddings = docs_embeddings.embed_documents(documents);

print(resulted_embeddings)
