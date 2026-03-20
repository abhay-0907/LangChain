from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="",
    task="text-generation",
    pipeline_kwargs=dict(
        temperatur= 0.5,
    )
)

result = llm.invoke("What is pytorch.");

print(f'result: {result}');