# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv;

# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# prompt1 = PromptTemplate(
#     template="Generate the notes from on the given text \n{topic}", input_variables=["topic"]
# )

# prompt2 = PromptTemplate(
#     template="Generate the 5 question and answer from the following text \n {topic}",
#     input_variables=['topic']
# )

# prompt3 = PromptTemplate(
#     template="Merge the notes and quiz into a single document \n notes -> {notes} and quizes -> {quiz}",
#     input_variables=['notes'
#     ,'quiz']
# )
# output_parser = StrOutputParser()
# paralle_chain = RunnableParallel(
#     {
#         "notes":prompt1|llm|output_parser,
#         "quiz":prompt2|llm|output_parser
#     }
# )

# merged_chain = prompt3 | llm | output_parser

# chain = paralle_chain | merged_chain

# result = chain.invoke({"topic":"How does self-attention work?"})

# print(result)


#! code enhanced by the claude 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY not found — check your .env file")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)

prompt1 = PromptTemplate.from_template("""
You are an expert educator creating study notes for a student preparing for exams or interviews.

Given the topic below, generate clear, well-structured notes that:
- Start with a one-line definition or summary
- Break the concept into logical sub-sections with headers
- Use bullet points for key facts, not dense paragraphs
- Include a simple analogy or example if it aids understanding
- Bold key terms the student should remember
- End with a "Key Takeaways" section (2-3 bullet points max)

Keep it concise and skimmable — this is for quick revision, not a textbook chapter.

Topic:
{topic}
""")

prompt2 = PromptTemplate.from_template("""
You are creating a short quiz to test understanding of the topic below.

Generate exactly 5 question-answer pairs that:
- Cover different aspects of the topic (not 5 variations of the same question)
- Range from conceptual ("what is X") to applied ("why does X matter" or "when would you use X")
- Have concise, accurate answers (2-4 sentences each, not one-word answers)
- Are formatted as:
  Q1: ...
  A1: ...

Avoid trick questions or ambiguous phrasing — the goal is reinforcement, not confusion.

Topic:
{topic}
""")

prompt3 = PromptTemplate.from_template("""
Combine the notes and quiz below into a single, well-formatted study document.

Structure the output as:
1. A title: "{topic}"
2. The notes section, unchanged in content but cleanly formatted
3. A "Practice Questions" section with the quiz
4. Do not add commentary, disclaimers, or explain what you're doing — output only the final document

Notes:
{notes}

Quiz:
{quiz}
""")

output_parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | llm | output_parser,
        "quiz": prompt2 | llm | output_parser,
        "topic": RunnablePassthrough(),
    }
)

merged_chain = prompt3 | llm | output_parser

chain = parallel_chain | merged_chain

if __name__ == "__main__":
    result = chain.invoke({"topic": "How does self-attention work?, with Mathematical Intution"})
    print(result)