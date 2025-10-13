import os
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain.prompts import PromptTemplate
from utils import *;
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from sentence_transformers import SentenceTransformer

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload


load_dotenv()
model=get_model()

pinecone_api_key=os.getenv("PINECONE_API_KEY")
pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
index_name=os.getenv("PINECONE_INDEX_NAME")
namespace=os.getenv("PINECONE_NAMESPACE")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_model,
    pinecone_api_key=pinecone_api_key  
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

async def call_llm(question):
    try:
        prompt_template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Question: {{question}}
        Helpful Answer:"""
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template,
        )
        chain=prompt|model
        response=await chain.invoke({"question":question})
        output=response.content().strip().lower() if hasattr(response, "content") else "No response from model"
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
async def call_rag(question):
    try:
        prompt_template = """
    You are a helpful AI assistant. Use the following retrieved context to answer the user's question.

    Context:
    {context}

    Question:
    {input}

    Answer clearly and concisely:
    """
        prompt = PromptTemplate(
            input_variables=["input", "context"],
            template=prompt_template,
        )
        document_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        result = await rag_chain.invoke({"input": question})
        output=result.content().strip().lower() if hasattr(result, "content") else "No response from model"
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def call_agent(question):
    return "hello"