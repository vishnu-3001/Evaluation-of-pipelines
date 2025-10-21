from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")
open_ai_org = os.getenv("OPENAI_ORG_ID")   
open_ai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  

def get_model():
    return ChatOpenAI(
        model="gpt-4o",
        openai_api_key=open_ai_key,
        organization=open_ai_org,     
        temperature=0,
    )

# def get_embedding_model():
#     return OpenAIEmbeddings(
#         model="text-embedding-3-large",
#         openai_api_key=open_ai_key      
#     )
