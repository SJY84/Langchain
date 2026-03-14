import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
groq_key = os.getenv("GROQ_API")

model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, temperature=0.7)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="List the key technical specifications of {phone}.",
    input_variables=["phone"]
)

prompt2 = PromptTemplate(
    template="List the pros and cons of buying {phone}.",
    input_variables=["phone"]
)

prompt3 = PromptTemplate(
    template="""Based on the phone information below:
    SPECIFICATIONS: {specs}
    PROS AND CONS:  {pros_cons}
    Give a final buying recommendation in simple words.""",
    input_variables=["specs", "pros_cons"]
)

specs_chain= prompt1 | model | parser
pros_cons_chain= prompt2 | model | parser

parallel_chain = RunnableParallel({
    "specs"    : specs_chain,
    "pros_cons": pros_cons_chain
})

final_chain    = prompt3 | model | parser
complete_chain = parallel_chain | final_chain

phone  = input("Enter phone name: ")
print("="*50)

result = complete_chain.invoke({"phone": phone})

print(result)

print("="*50)
print(" "*50)

print(pros_cons_chain.invoke({"phone": phone}))

print("="*50)
print(" "*50)

complete_chain.get_graph().print_ascii()