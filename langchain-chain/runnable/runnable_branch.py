from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(temperature=0.7)


class FindCategory(BaseModel):
    category:Literal["complain","refund","general"]  = Field(description="choose the category of the email")



output_parser = PydanticOutputParser(pydantic_object=FindCategory)
string_parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="Which category does the following email belong to? {email}\n{format_instruction}", 
    input_variables=["email"],
    partial_variables={"format_instruction":output_parser.get_format_instructions()}
)

category_chain = RunnableSequence(prompt1,model,output_parser)


prompt2 = PromptTemplate(
    template="Write the appropiate response of this complain email to the user\n email: {email}",
    input_variables=["email"]
)
prompt3 = PromptTemplate(
    template="Write the appropiate response of this refund email to the user \n email: {email}",
    input_variables=["email"]
)
prompt4 = PromptTemplate(
    template="Write the appropiate response of this general email to the user \n email: {email}",
    input_variables=["email"]
)

branch_chain = RunnableBranch(
    (lambda X: X.category == "complain", prompt2 | model | string_parser),
    (lambda X: X.category == "refund", prompt3 | model | string_parser),
    (lambda X: X.category == "general", prompt4 | model | string_parser),
    RunnableLambda(lambda X : "Couldn't find the category")
)

final_chain = RunnableSequence(category_chain,branch_chain)

result = final_chain.invoke({"email":"""Dear Philips Customer Support,

I am writing to lodge a complaint regarding my Philips air purifier, which has not been functioning properly.

I purchased this product expecting reliable performance and good air purification. However, I have been experiencing several issues. The purifier is making an unusual noise, the air-quality indicator is not working correctly, and the device does not appear to be effectively purifying the air.

**Product Details:**

* **Product Model:** Philips AC1711/61
* **Purchase Date:** 15 June 2026
* **Purchase From:** Amazon India
* **Invoice/Order Number:** 402-7856342-1234567
* **Warranty Status:** Under Warranty
* **Purchase Price:** ₹18,499

I request you to look into this matter and arrange an inspection and repair or replacement of the product at the earliest, as applicable under the warranty terms.

Please let me know the next steps and the expected resolution timeline. I can provide the purchase invoice, photographs, videos, or any other information required.

I look forward to a prompt resolution of my complaint.

Sincerely,
John Doe
"""})

print(result)