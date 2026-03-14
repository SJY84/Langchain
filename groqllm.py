import os
from dotenv import load_dotenv

load_dotenv()

api_key= os.getenv("GROQ_API")


from langchain_groq import ChatGroq

model=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.7,
    max_retries=2,
)

messages = [
    ('system', "You are a cricekt expert with good knowledge of the game who can explain complex topics in simple terms."),
    ('user', 'What are the different field postions?')

]

response = model.invoke(messages)
print(response.content)








