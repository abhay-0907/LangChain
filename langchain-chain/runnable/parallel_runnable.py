from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model="gpt-5.6-luna", temperature=1.5)

output_parser = StrOutputParser()

tweet_prompt = PromptTemplate(
    template="""You are a professional social media manager with expertise in Twitter/X. Your job is to write engaging tweets based on the given topic, with the goal of increasing user engagement, including likes, shares, reposts, and comments.

    Use attention-grabbing language, a conversational tone, and relevant hashtags (#) to maximize reach and engagement.

    Topic: {topic}

    Create a compelling Twitter/X post based on the topic above. Ensure that the post is grammatically correct, concise, engaging, and optimized for Twitter/X.""",
    input_variables=["topic"]
)
linkedin_prompt = PromptTemplate(
    template="""You are a professional social media manager with expertise in LinkedIn. Your job is to write engaging LinkedIn posts based on the given topic, with the goal of increasing user engagement, including likes, shares, comments, and meaningful interactions.

    Use a professional yet engaging tone, compelling language, and relevant hashtags (#) to maximize reach and visibility.

    **Topic:** {topic}

    Create a high-quality LinkedIn post based on the topic above. Ensure that the post is grammatically correct, informative, engaging, and optimized for LinkedIn. Encourage readers to share their opinions or experiences in the comments.
    """,
    input_variables=["topic"]
)

parallel_chain = RunnableParallel(
    {"X":RunnableSequence(tweet_prompt,model,output_parser),
    "linkedin":RunnableSequence(linkedin_prompt,model,output_parser)},
)

result = parallel_chain.invoke({"topic":"How the AI change the tech industry"})

print("X tweet = ",result["X"])
print("linkedin Post = ",result["linkedin"])
