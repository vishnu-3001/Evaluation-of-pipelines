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
from langchain.callbacks import get_openai_callback
from typing import TypedDict, Optional, Dict, Any




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


verification_prompt_template = """
You are an expert verifier. Your task is to assess whether the 'Generated Answer' is supported by the 'Context' for the given 'Question'.

Please:
1. Analyze if the Generated Answer is correct and supported by the context.
2. Provide your reasoning concisely in 1–3 sentences.
3. If it is unsupported, state *why* (missing evidence, contradicts context, etc.).

Output format:
Reason:
<your reasoning>

Context:
{context}

Question:
{question}

Generated Answer:
{answer}

Final Output:
Reason:
"""


# ----------------------------- LLM Call -----------------------------
async def call_llm(context: str, question: str):
    try:
        # === GENERATION PHASE ===
        gen_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful assistant.
You are given a passage and a question from a reading comprehension dataset.
Answer the question *only* using the passage below.
If you don’t know the answer from the passage, say “I don’t know.”

Passage:
{context}

Question:
{question}

Answer:
"""
        )
        gen_chain = gen_prompt | model

        with get_openai_callback() as cb_gen:
            gen_response = await gen_chain.ainvoke({"context": context, "question": question})
            answer = gen_response.content.strip() if hasattr(gen_response, "content") else "No response from model"

        generate_tokens = {
            "prompt_tokens": cb_gen.prompt_tokens,
            "completion_tokens": cb_gen.completion_tokens,
            "total_tokens": cb_gen.total_tokens,
            "total_cost": cb_gen.total_cost
        }

        # === VERIFICATION PHASE ===
        verify_prompt = PromptTemplate.from_template(verification_prompt_template)
        verify_chain = verify_prompt | model

        with get_openai_callback() as cb_ver:
            verify_result = await verify_chain.ainvoke({
                "context": context,
                "question": question,
                "answer": answer
            })
            reason = verify_result.content.strip() if hasattr(verify_result, "content") else "No verification response"

        verify_tokens = {
            "prompt_tokens": cb_ver.prompt_tokens,
            "completion_tokens": cb_ver.completion_tokens,
            "total_tokens": cb_ver.total_tokens,
            "total_cost": cb_ver.total_cost
        }

        return {
            "response": answer,
            "reason": reason,
            "context": context,
            "token_data": {
                "generate": generate_tokens,
                "verify": verify_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ----------------------------- RAG Call -----------------------------
async def call_rag(question):
    try:
        # === RETRIEVE DOCUMENTS ===
        docs = []
        try:
            docs = await retriever.aget_relevant_documents(question)
        except Exception:
            docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        # === GENERATION PHASE ===
        gen_prompt = PromptTemplate(
            input_variables=["input", "context"],
            template=agent_prompt_template
        )
        document_chain = create_stuff_documents_chain(model, gen_prompt)

        with get_openai_callback() as cb_gen:
            gen_result = await document_chain.ainvoke({"input": question, "context": docs})
            answer = gen_result.content.strip() if hasattr(gen_result, "content") else str(gen_result)

        generate_tokens = {
            "prompt_tokens": cb_gen.prompt_tokens,
            "completion_tokens": cb_gen.completion_tokens,
            "total_tokens": cb_gen.total_tokens,
            "total_cost": cb_gen.total_cost
        }

        # === VERIFICATION PHASE ===
        verify_prompt = PromptTemplate.from_template(verification_prompt_template)
        verify_chain = verify_prompt | model

        with get_openai_callback() as cb_ver:
            verify_result = await verify_chain.ainvoke({
                "context": context,
                "question": question,
                "answer": answer
            })
            reason = verify_result.content.strip() if hasattr(verify_result, "content") else "No verification response"

        verify_tokens = {
            "prompt_tokens": cb_ver.prompt_tokens,
            "completion_tokens": cb_ver.completion_tokens,
            "total_tokens": cb_ver.total_tokens,
            "total_cost": cb_ver.total_cost
        }

        return {
            "response": answer,
            "reason": reason,
            "context": [doc.page_content for doc in docs],
            "token_data": {
                "generate": generate_tokens,
                "verify": verify_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")

# async def call_llm(context: str, question: str):
#     try:
#         prompt_template = """
# You are a helpful assistant.
# You are given a passage and a question from a reading comprehension dataset.
# Answer the question *only* using the passage below.
# If you don’t know the answer from the passage, say “I don’t know.”

# Passage:
# {context}

# Question:
# {question}

# Answer:
# """
#         prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template=prompt_template,
#         )

#         chain = prompt | model

#         with get_openai_callback() as cb:
#             response = await chain.ainvoke({"context": context, "question": question})
#             output = response.content.strip() if hasattr(response, "content") else "No response from model"

#         token_data = {
#             "prompt_tokens": cb.prompt_tokens,
#             "completion_tokens": cb.completion_tokens,
#             "total_tokens": cb.total_tokens,
#             "total_cost": cb.total_cost
#         }
#         # print({"response":output,"token_usage":token_data})

#         return {
#             "response": output,
#             "token_usage": token_data
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# async def call_rag(question):
#         try:
#             docs = []
#             try:
#                 docs = await retriever.aget_relevant_documents(question)
#             except Exception as e_async:
#                 try:
#                     docs = retriever.get_relevant_documents(question)
#                 except Exception as e_sync:
#                     raise e_sync
#             prompt_template = """
#             You are a helpful AI assistant. Use the following retrieved context to answer the user's question.
#             Context:
#             {context}
#             Question:
#             {input}
#             Answer:
#             """
#             prompt = PromptTemplate(input_variables=["input", "context"], template=prompt_template)
#             document_chain = create_stuff_documents_chain(model, prompt)
#             with get_openai_callback() as cb:
#                 result = await document_chain.ainvoke({"input": question, "context": docs})
#                 output = result.content.strip() if hasattr(result, "content") else str(result)
#             extracted_content=[]
#             for doc in docs:
#                 extracted_content.append(doc.page_content)
#             token_data={
#                 "prompt_tokens": cb.prompt_tokens,
#                 "completion_tokens": cb.completion_tokens,
#                 "total_tokens": cb.total_tokens,
#                 "total_cost": cb.total_cost
#             }
#             return {
#                 "response": output,
#                 "context": extracted_content,
#                 "token_usage": token_data
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")
    


class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    reason: str
    docs: Optional[list]
    generate_tokens: Optional[Dict[str, Any]]
    verify_tokens: Optional[Dict[str, Any]]



agent_prompt_template = """
You are a helpful AI assistant. Use the following retrieved context to answer the user's question.

Context:
{context}

Question:
{input}

Answer:
"""

verification_prompt_template = """
You are an expert verifier. Your task is to assess whether the 'Generated Answer' is supported by the 'Context' for the given 'Question'.

Please:
1. Analyze if the Generated Answer is correct and supported by the context.
2. Provide your reasoning concisely in 1–3 sentences.
3. If it is unsupported, state *why* (missing evidence, contradicts context, etc.).

Output format:
Reason:
<your reasoning>

Context:
{context}

Question:
{question}

Generated Answer:
{answer}

Final Output:
Reason:
"""


def retrieve_context(state: AgentState) -> AgentState:
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    state["docs"]=docs
    state["context"] = "\n\n".join([doc.page_content for doc in docs])
    return state


def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]

    prompt = PromptTemplate.from_template(agent_prompt_template)
    chain = prompt | model

    with get_openai_callback() as cb:
        result = chain.invoke({"context": context, "input": question})
        state["answer"] = result.content
        state["generate_tokens"] = {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
            "total_cost": cb.total_cost
        }

    return state


def verify_answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state["context"]
    initial_answer = state["answer"]

    prompt = PromptTemplate.from_template(verification_prompt_template)
    chain = prompt | model

    with get_openai_callback() as cb:
        result = chain.invoke({
            "context": context,
            "question": question,
            "answer": initial_answer
        })
        state["reason"] = result.content.strip()
        state["verify_tokens"] = {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
            "total_cost": cb.total_cost
        }

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
    inputs = {"question": question, "context": "", "answer": "", "reason": "", "docs": []}
    result = rag_agent.invoke(inputs)

    token_data = {
        "generate_node": result.get("generate_tokens", {}),
        "verify_node": result.get("verify_tokens", {})
    }

    extracted_content = [doc.page_content for doc in result.get("docs", []) or []]

    return {
        "response": result.get("answer", ""),
        "context": extracted_content,
        "reason": result.get("reason", ""),
        "token_data": token_data
    }

