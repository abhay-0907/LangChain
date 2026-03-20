from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    """The Taj Mahal in Agra is a UNESCO World Heritage Site built by Shah Jahan. 
    It attracts millions of tourists every year due to its white marble beauty. 
    Interestingly, modern tourism apps use Python-based backend systems to manage bookings and visitor analytics. 
    Weather conditions and pollution levels can affect the preservation of the monument over time.""",

    """Python is widely used in data science, AI, and backend systems. 
    It is also used in space research by organizations like NASA for simulations and data processing. 
    Some developers even build health tracking apps using Python frameworks. 
    However, learning syntax and debugging can sometimes be challenging for beginners.""",

    """A healthy lifestyle includes proper nutrition, exercise, and mental well-being. 
    Many wearable devices track health metrics and send data to cloud systems built using modern technologies. 
    Pollution and environmental conditions can negatively impact respiratory health. 
    Drinking enough water daily is one of the simplest habits for maintaining wellness.""",

    """Space agencies like ISRO and NASA conduct missions to explore planets and study the universe. 
    These missions rely heavily on software, simulations, and data analysis tools. 
    Satellite technology also helps monitor climate change and environmental damage on Earth. 
    Launching rockets requires precise engineering and significant funding.""",

    """Electric vehicles reduce carbon emissions and help fight climate change. 
    They rely on battery technology and software systems for efficiency and performance. 
    Some EV dashboards are built using modern programming languages for better user experience. 
    However, charging infrastructure is still developing in many regions."""
]

resulted_embeddings = embeddings.embed_documents(documents);


query = """How do electric vehicles help the environment, and what role does technology play in improving their efficiency and performance?"""

query_embeddings = embeddings.embed_query(query);

similarity_score = cosine_similarity([query_embeddings], resulted_embeddings)[0]

index, score = sorted(list(enumerate(similarity_score)),key=lambda x: x[1])[-1]

print(query)
print(f"Document {index} is most similar with a similarity score of {score}")
