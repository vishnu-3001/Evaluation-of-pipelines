from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

open_ai_key=os.getenv("OPENAI_API_KEY")

def get_model():
    model=ChatOpenAI(model="gpt-4o",api_key=open_ai_key,temperature=0)
    return model

def get_embedding_model():
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=open_ai_key)
    return embedding_model
