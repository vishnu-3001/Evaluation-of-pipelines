import os
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain.prompts import PromptTemplate
from utils import *;
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langgraph.graph import StateGraph,END
from typing import TypedDict
from langchain_community.embeddings import SentenceTransformerEmbeddings



# uvicorn main:app --host 0.0.0.0 --port 8000 --reload


load_dotenv()
model=get_model()

pinecone_api_key=os.getenv("PINECONE_API_KEY")
pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
index_name=os.getenv("PINECONE_INDEX_NAME")
namespace=os.getenv("PINECONE_NAMESPACE")
embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")

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
        response=await chain.ainvoke({"question":question})
        # print(response)
        output=response.content.strip().lower() if hasattr(response, "content") else "No response from model"
        print(output)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
async def call_rag(question):
    try:
        prompt_template = """
    You are a helpful AI assistant. Use the following retrieved context to answer the user's question.Dont make the answer if you donet know, dont hallucinate, just say that you dont now or you are unable to get the answer.

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
        result = await rag_chain.ainvoke({"input": question})
        output=result.content.strip().lower() if hasattr(result, "content") else "No response from model"
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class AgentState(TypedDict):
    question: str
    context: str
    answer: str


agent_prompt_template = """
You are a helpful AI assistant. Use the following retrieved context to answer the user's question. 
Don't make the answer if you don't know; don't hallucinate. Just say that you don't know or are unable to get the answer.

Context:
{context}

Question:
{input}

Answer clearly and concisely:
"""
agent_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template=agent_prompt_template,
)

def retrieve_context(state: AgentState) -> AgentState:
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    state["context"] = "\n\n".join([doc.page_content for doc in docs])
    return state

def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]
    chain = agent_prompt | model 
    result = chain.invoke({"question": question, "context": context})
    state["answer"] = result.content
    return state

# ❌ remove the old `state_schema` dict entirely

# ✅ Proper LangGraph construction
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_answer)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_agent = graph.compile()
def call_agent(question: str):
    inputs = {"question": question}
    result = rag_agent.invoke(inputs)
    return {
        "question": question,
        "answer": result["answer"],
        "context": result["context"]
    }
