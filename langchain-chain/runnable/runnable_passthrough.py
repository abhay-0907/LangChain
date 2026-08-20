from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

output_parser = StrOutputParser()

# First prompt: Generate summary + real-life example
prompt1 = PromptTemplate(
    template="""Tell me the summary of the book and at the end give a
real-life example based on the essence of the book.

The book is: {book_name}""",
    input_variables=["book_name"]
)

# Second prompt: Verify and explain
prompt2 = PromptTemplate(
    template="""You are a great literaturist and novelist.

The first response contains a summary and a real-life example of the book.
Check whether they are accurate according to the actual book.

Book name: {book_name}

Summary and example generated earlier:
{summary}

Now provide your final response in this format:

1. Original Summary
   - Give the summary clearly.

2. Real-Life Example
   - Give the real-life example clearly.

3. Verification and Explanation
   - Explain whether the summary and example accurately represent
     the book.
   - If anything is incorrect or misleading, correct it.
""",
    input_variables=["book_name", "summary"]
)

# First chain
chain1 = RunnableSequence(
    prompt1,
    model,
    output_parser
)

# Final chain
chain = (
    {
        "book_name": RunnablePassthrough(),
        "summary": chain1
    }
    | prompt2
    | model
    | output_parser
)

result = chain.invoke(
    {"book_name": "The Brothers Karamazov by Fyodor Dostoevsky"}
)

print(result)