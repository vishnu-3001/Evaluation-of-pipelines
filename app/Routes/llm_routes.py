from fastapi import APIRouter,HTTPException,Request
from fastapi.responses import JSONResponse
from app.Services import *
llm_router=APIRouter()

@llm_router.post("/api")
async def api_response(req:Request):
    try:
        body=await req.json()
        query=body.get("query")
        if not query:
            raise HTTPException(status_code=400,detail="Query parameter is required")
        result=await call_llm(query)
        return JSONResponse(content={"response":result})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@llm_router.post("/rag")
async def rag_response(req:Request):
    try:
        body=await req.json()
        query=body.get("query")
        if not query:
            raise HTTPException(status_code=400,detail="Query parameter is required")
        result=await call_rag(query)
        return JSONResponse(content={"response":result})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@llm_router.post("/agent")
async def agent_response(req:Request):
    try:
        body=await req.json()
        query=body.get("query")
        if not query:
            raise HTTPException(status_code=400,detail="Query parameter is required")
        result=call_agent(query)
        return JSONResponse(content={"response":result})
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))