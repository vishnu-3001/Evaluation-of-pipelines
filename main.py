from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.Routes.llm_routes import llm_router
from app.Services import *
from utils import *;
import json
import asyncio
import time

app=FastAPI(
    title="Evaluation of pipelines",
    description="A project to evaluate different AI Pipelines",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(llm_router)

async def llm_api():
    json_path = "app/Services/test-cases.json"  
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for case in data:
            question=case["question"]
            context=case["context"]
            start=time.time()
            response=await call_llm(context, question)
            end=time.time()
            latency=end-start
            hallucination=evaluate_hallucination_llm(response["response"],case["answer"],context)
            result={"response":response["response"],"latency":latency,"tokens_data":response["token_data"],"hallucination":hallucination}
            print(result)

async def llm_rag():
    json_path = "app/Services/test-cases.json"  
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for case in data:
            question=case["question"]
            start=time.time()
            response=await call_rag(question)
            end=time.time()
            latency=end-start
            hallucination=evaluate_hallucination(response["response"],case["answer"],response["context"],case["context"])
            result={"response":response["response"],"latency":latency,"tokens_data":response["token_data"],"hallucination":hallucination}
            print(result)
def llm_agent():
    json_path = "app/Services/test-cases.json"  
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for case in data:
            question=case["question"]
            start=time.time()
            response=call_agent(question)
            end=time.time()
            latency=end-start
            hallucination=evaluate_hallucination(response["response"],case["answer"],response["context"],case["context"])
            result={"response":response["response"],"latency":latency,"tokens_used":response["token_data"],"hallucination":hallucination}
            print(result)

if __name__=="__main__":
    # uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)
    # asyncio.run(llm_api())
    # asyncio.run(llm_rag())
    llm_agent()    
