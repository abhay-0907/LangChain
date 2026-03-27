from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    template="""
    You are a good topic explainer using storytelling method.
Explain the given {topic} in a simple way.

Important Instructions:
1. Use storytelling.
2. Use {language}.
3. Level: {level}.
""",
    input_variables=["topic", "language", "level"]
)

prompt_template.save("prompt.json")