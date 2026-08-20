from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

output_parser = StrOutputParser()

def count_words(text):
    return len(text.split())


prompt1 = PromptTemplate(
    template="Write the a very funny joke in the topic in hindi language.\n The topic is {topic}",
    input_variables=["topic"]  
)

joke_chain = RunnableSequence(
    prompt1,
    model,
    output_parser
)

parallel_chain = RunnableParallel(
    {
        "joke":joke_chain,
        "words":RunnableLambda(count_words)
    }
)

final_chain = RunnableSequence(
    joke_chain,
    parallel_chain
)

result = final_chain.invoke({"topic":"Human vs Animal"})

print(result)