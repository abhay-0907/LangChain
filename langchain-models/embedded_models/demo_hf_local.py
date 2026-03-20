from langchain_huggingface import HuggingFaceEmbeddings

result = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = result.embed_query("Machine learning models learn patterns from data.")

print(str(embeddings))