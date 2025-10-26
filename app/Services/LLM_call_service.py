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
from .metrics import evaluate_hallucination



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
    pinecone_api_key=pinecone_api_key,
    namespace=namespace
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
        output=response.content.strip().lower() if hasattr(response, "content") else "No response from model"
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
async def call_rag(question):
        original_answer="John F. Kennedy"
        original_content="The Apollo program was the third United States human spaceflight program carried out by NASA, which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft to follow the one-person Project Mercury, which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in an address to Congress on May 25, 1961."
        try:
            docs = []
            try:
                docs = await retriever.aget_relevant_documents(question)
            except Exception as e_async:
                try:
                    docs = retriever.get_relevant_documents(question)
                except Exception as e_sync:
                    raise e_sync
            prompt_template = """
            You are a helpful AI assistant. Use the following retrieved context to answer the user's question.
            Context:
            {context}
            Question:
            {input}
            Answer:
            """
            prompt = PromptTemplate(input_variables=["input", "context"], template=prompt_template)
            document_chain = create_stuff_documents_chain(model, prompt)
            result = await document_chain.ainvoke({"input": question, "context": docs})
            output = result.content.strip() if hasattr(result, "content") else str(result)
            extracted_content=[]
            for doc in docs:
                extracted_content.append(doc.page_content)
            # metrics=evaluate_hallucination(output,original_answer,extracted_content,original_content)
            return output
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")
    



class AgentState(TypedDict):
    question: str
    context: str
    answer: str

agent_prompt_template = """
You are a helpful and careful AI assistant.

Use the following context to answer the user's question. If the answer can be determined or strongly inferred from the context, answer directly and concisely.

If the context does not contain enough information, respond with: "I'm sorry, I don't have enough information to answer that."

Context:
{context}

Question:
{input}

Answer:
"""

verification_prompt_template = """
You are an expert answer verifier. Your task is to check if the 'Generated Answer' is fully supported by the 'Context' for the 'Question'.

1. If the 'Generated Answer' is fully supported by the context, output ONLY the 'Generated Answer' text.
2. If the 'Generated Answer' is:
    a) The refusal phrase ("I'm sorry, I don't have enough information to answer that."), OR
    b) Contains information not in the context (a hallucination), OR
    c) Incorrect based on the context,
   ...then your final output must be ONLY the refusal phrase: "I'm sorry, I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Generated Answer:
{answer}

Final Verified Answer:
"""

def retrieve_context(state: AgentState) -> AgentState:
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    state["context"] = "\n\n".join([doc.page_content for doc in docs])
    return state

def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]
    prompt = PromptTemplate.from_template(agent_prompt_template)
    chain = prompt | model
    result = chain.invoke({"context": context, "input": question})
    state["answer"] = result.content
    return state

def verify_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]
    initial_answer = state["answer"]
    prompt = PromptTemplate.from_template(verification_prompt_template)
    chain = prompt | model
    verified_content = chain.invoke({
        "context": context,
        "question": question,
        "answer": initial_answer
    })
    state["answer"] = verified_content.content.strip()
    return state

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_answer)
graph.add_node("verify", verify_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "verify")
graph.add_edge("verify", END)

rag_agent = graph.compile()

def call_agent(question: str):
    inputs = {"question": question, "context": "", "answer": ""}
    result = rag_agent.invoke(inputs)
    return {
        "question": question,
        "answer": result["answer"],
        "context": result["context"]
    }

