from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from urllib.parse import unquote
from src.pipeline.service import PipelineService
import asyncio
import json

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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/book-vec/{url:path}")
async def book2vec(url: str):
    try:
        # URL解码
        decoded_url = unquote(url)
        print(decoded_url)
        # 运行pipeline
        pipeline_service = PipelineService()
        pipeline_service.pipeline_book2vec_run(decoded_url)
        result = decoded_url
        return {
            "code": 200,
            "msg": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/book-ask/{question}/{source}")
async def ask_book(question: str, source: str):
    async def generate_response():
        try:
            pipeline_service = PipelineService()
            async for chunk in pipeline_service.ask_book_stream(question,source):
                # 将每个文本块包装成SSE格式
                yield f"data: {json.dumps({'code': 200, 'msg': 'success', 'data': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'code': 500, 'msg': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
