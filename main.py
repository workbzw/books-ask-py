from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from urllib.parse import unquote
from src.pipeline.service import PipelineService
import asyncio
import json
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 获取环境变量
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
API_KEY = os.getenv("API_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

app = FastAPI(
    title="FastAPI App",
    description="Your API Description",
    version="1.0.0",
    debug=DEBUG
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 定义请求体模型
class UrlRequest(BaseModel):
    url: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/book-vec")
async def book2vec(request: UrlRequest):
    try:
        # 从请求体中获取URL
        decoded_url = unquote(request.url)
        print(decoded_url)
        # 运行pipeline
        pipeline_service = PipelineService()
        pipeline_service.pipeline_book2vec_run(decoded_url)
        result = decoded_url
        return {
            "code": 200,
            "msg": "success", 
            "data": {"data":result}
        }
    except Exception as e:
        return {
            "code": 500,
            "msg": "error", 
            "data": {"data":str(e)}
        }
    

    
@app.get("/api/book-ask")
async def ask_book(
    question: str = Query(..., description="The question to ask"),
    source: str = Query(..., description="The source of the book")
):
    async def generate_response():
        try:
            pipeline_service = PipelineService()
            async for chunk in pipeline_service.ask_book_stream(question, source):
                # 将每个文本块包装成SSE格式
                yield f"data: {json.dumps({'code': 200, 'msg': 'success', 'data': {'data':chunk}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'code': 500, 'msg': 'error', 'data': {'data':str(e)}})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
