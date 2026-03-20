# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     temperature=0.7,
#     max_new_tokens=200,
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What are the 3 rarest things on Earth?")
# print(result.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    do_sample=False,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What are the 3 rarest things on Earth?")
print(result.content)