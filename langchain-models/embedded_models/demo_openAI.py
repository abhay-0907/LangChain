from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

vector_embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=128);

embedding_result = vector_embeddings.embed_query("The speed of light is 299,792,458 meters per second")

print(embedding_result)