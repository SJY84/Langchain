import os
from dotenv import load_dotenv

load_dotenv()

api_key= os.getenv("GROQ_API")


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate    

model=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.7,
    max_retries=2,
)

promt= PromptTemplate(
    input_variables="muscle",
    template="Give me a {muscle} workout plan."
)

text=promt.format(muscle="chest and triceps")


messages = [
    ('system', "You are a personal trainer with good knowledge of the exercises. Give the best workout aslo explaing which muscle it targets along with the form and the number of sets and reps."),
    ('user', text),
    ('ai', "Sure! Here's your workout plan.")

    

    ]

response = model.invoke(messages)
print(response.content)





