import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key= os.getenv("GROQ_API")


system_message="You are a travel expert... give budget in Indian Rupees (INR)."

places_prompt = ChatPromptTemplate.from_messages([
    ('system', system_message ),
    ('human', "Give me top 10 places to visit in {destination}..."),
    
])

itinerary_prompt = ChatPromptTemplate.from_messages([
    ('system', system_message),
    ('human', "For the given {places} give an itinerary for 5 days with best time to visit and best food to try in each place.")
])

budget_prompt = ChatPromptTemplate.from_messages([  
    ('system', system_message),
    ('human', "For the given {itinerary} give a budget estimate for 5 days trip in Indian Rupees (INR) only. Never use USD.")
])




model=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.7,
    max_retries=2,
)

parser = StrOutputParser()

messages=[
    ('system',"You are a travel expert with good knowledge of the places to visit, the best time to visit and the best food to try in each place. You also have good knowledge of the budget estimate for a trip in Indian currency. Give the best recommendations for a trip based on the destination provided by the user.")
]

places_chain= places_prompt | model |  parser

itinerary_chain= itinerary_prompt | model | parser 

budget_chain= budget_prompt | model | parser

final_chain= places_chain | itinerary_chain | budget_chain

response=final_chain.invoke("Vietnam")
print(response)
final_chain.get_graph().print_ascii()