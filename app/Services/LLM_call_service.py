from fastapi import HTTPException
from langchain.prompts import PromptTemplate
from utils import *;
model=get_model()

async def call_llm(question):
    try:
        prompt_template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Question: {question}
        Helpful Answer:"""
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template,
        )
        response = model.chat.prompt(
            prompt.format(context="", question=question)
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def call_rag(question):
    return "hello"
def call_agent(question):
    return "hello"