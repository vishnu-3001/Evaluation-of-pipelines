from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.Routes.llm_routes import llm_router
from utils import *;

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

if __name__=="__main__":
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)
    
