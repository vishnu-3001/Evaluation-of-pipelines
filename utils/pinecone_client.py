from pinecone import Pinecone
import os
from dotenv import load_dotenv
from utils import *;
import json
from pathlib import Path
import uuid
from sentence_transformers import SentenceTransformer


load_dotenv()
pinecone_api_key=os.getenv("PINECONE_API_KEY")
pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
index_name=os.getenv("PINECONE_INDEX_NAME")
namespace=os.getenv("PINECONE_NAMESPACE")
open_ai_key=os.getenv("OPEN_AI_API_KEY")

pc=Pinecone(api_key=pinecone_api_key,environment=pinecone_environment)
index=pc.Index(index_name)
embedding_dimension=1024
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")



BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "train-v2.0.json"

def get_stats():
    stats=index.describe_index_stats(namespace=namespace)
    return stats

vectors=[]
def get_data():
    stats = get_stats()
    if stats.get("total_vector_count", 0) == 0:
        with open('train-v2.0.json', 'r') as f:
            data = json.load(f)
        for article in data['data']:
            title=article['title']
            for p_idx,para in enumerate(article['paragraphs']):
                print(p_idx)
                context=para['context']
                context_embedding = embedding_model.encode(context, normalize_embeddings=True).tolist()
                metadata = {
                    "title": title,
                    "context": context,
                    "questions": [q["question"] for q in para["qas"]],
                    "answers": [a["text"] for q in para["qas"] for a in q["answers"]],
                    "is_impossible": [str(q["is_impossible"]) for q in para["qas"]]  # cast to string if multiple
                }
                vector_id=str(uuid.uuid4())
                vectors.append({
                    "id":vector_id,
                    "values":context_embedding,
                    "metadata":metadata,
                })
    print("Data inserted successfully")
    return vectors

vectors=get_data()
size=100
for i in range(0,len(vectors),size):
    print("upserting",i)
    index.upsert(vectors[i:i+size],namespace=namespace)