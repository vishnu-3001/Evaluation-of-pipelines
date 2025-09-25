from pinecone import Pinecone
import os
from dotenv import load_dotenv
from utils import *;
load_dotenv()
pinecone_api_key=os.getenv("PINECONE_API_KEY")
pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
index_name=os.getenv("PINECONE_INDEX_NAME")
namespace=os.getenv("PINECONE_NAMESPACE")

pc=Pinecone(api_key=pinecone_api_key,environment=pinecone_environment)
index=pc.Index(index_name)
embedding_dimension=1536
embeddings_model=get_embedding_model()

def get_stats():
    stats=index.describe_index_stats(namespace=namespace)
    return stats

def get_data():
    stats=get_stats()
    if(stats.total_vector_count==0):
        print("hello")
