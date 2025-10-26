from pinecone import Pinecone
import os
from dotenv import load_dotenv
from utils import *;
import json
from pathlib import Path
import uuid
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv()
pinecone_api_key=os.getenv("PINECONE_API_KEY")
pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
index_name=os.getenv("PINECONE_INDEX_NAME")
namespace=os.getenv("PINECONE_NAMESPACE")
open_ai_key=os.getenv("OPEN_AI_API_KEY")

pc=Pinecone(api_key=pinecone_api_key,environment=pinecone_environment)
index=pc.Index(index_name)
embedding_dimension=1024
embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "train-v2.0.json"
namespace = os.getenv("PINECONE_NAMESPACE")  # e.g. "default"

def get_stats():
    stats=index.describe_index_stats(namespace=namespace)
    return stats


vectors=[]
def get_data():
    stats = get_stats()
    vectors = []

    if stats.get("total_vector_count", 0) == 0:
        with open('train-v2.0.json', 'r') as f:
            data = json.load(f)

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

        for article in data['data']:
            title = article['title']
            for p_idx, para in enumerate(article['paragraphs']):
                print("creating vector ",p_idx)
                context = para['context']
                chunks = splitter.split_text(context)

                for chunk_idx, chunk in enumerate(chunks):
                    # ✅ Correct: use embed_documents for docs
                    context_embedding = embedding_model.embed_documents([chunk])[0]

                    metadata = {
                        "title": title,
                        "chunk_id": f"{p_idx}_{chunk_idx}",
                        "text": chunk,
                        "questions": [q["question"] for q in para["qas"]],
                        "answers": [a["text"] for q in para["qas"] for a in q["answers"]],
                        "is_impossible": [str(q["is_impossible"]) for q in para["qas"]],
                    }

                    vector_id = str(uuid.uuid4())
                    vectors.append({
                        "id": vector_id,
                        "values": context_embedding,
                        "metadata": metadata,
                    })

        print(f"✅ Inserted {len(vectors)} document vectors successfully")
    else:
        print("Vectors already exist in DB")

    return vectors


vectors=get_data()
size=100
for i in range(0,len(vectors),size):
    print("upserting",i)
    index.upsert(vectors[i:i+size],namespace=namespace)