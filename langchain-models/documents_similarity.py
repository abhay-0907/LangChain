from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
vectors = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "The Taj Mahal in Agra is a UNESCO World Heritage Site famous for its white marble beauty and Mughal architecture.",  # travel
    "Python is widely used for data science, machine learning, and backend web development.",  # tech
    "Eating a balanced diet with fruits, vegetables, and proteins helps maintain good health.",  # health
    "Space missions by ISRO and NASA explore planets, satellites, and the origins of the universe.",  # space
    "Electric vehicles reduce carbon emissions and are a sustainable alternative to petrol cars."  # environment
]

documents_embeddings = vectors.embed_documents(documents)


query = "How do electric cars help reduce pollution and protect the environment?"

query_embeddings = vectors.embed_query(query);

similarity_score = cosine_similarity([query_embeddings], documents_embeddings)

print(f'similarity score: ${similarity_score}');