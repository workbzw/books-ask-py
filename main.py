import os
import json
import traceback

from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from urllib.parse import unquote
from src.services.service import AskService
from pydantic import BaseModel
from src.pipeline_item import Processor
from src.pipeline_item.data_scraping_step import DataScrapingStep
from src.pipeline_item.document_reader import DocumentReaderStep
from src.pipeline_item.pinecone_save_step import PineconeSaveStep
from src.pipeline_item.temp_file_saver import TempFileSaverStep
from src.pipeline_item.text_cleaner import TextCleanerStep
from src.pipeline_item.text_embedding import TextEmbeddingStep
from src.pipeline_item.text_splitter import TextSplitterStep

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

origins = [
    "http://localhost:3000",  # React 开发服务器
    "http://localhost:5173",  # Vite 开发服务器
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"  # 在开发环境中允许所有源（生产环境不建议）
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


# 定义请求体模型
class UrlRequest(BaseModel):
    url: str


@app.post("/api/book-vec-url")
async def book_vec_url(request: UrlRequest):
    """
    上传url，爬取data，并保存到向量库
    """
    try:
        # 从请求体中获取URL
        decoded_url = unquote(request.url)
        print(decoded_url)
        # 运行pipeline
        processor = Processor()
        processor.add_step(DataScrapingStep())
        processor.add_step(TextCleanerStep())
        processor.add_step(TextSplitterStep(chunk_size=600, chunk_overlap=120))
        processor.add_step(TextEmbeddingStep(model_name='all-MiniLM-L6-v2'))
        processor.add_step(PineconeSaveStep(
            api_key=os.getenv("PINECONE_API_KEY"),
            source=decoded_url,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        ))
        result = await processor.process([decoded_url])

        return {
            "code": 200,
            "msg": "success",
            "data": {"data": result}
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "code": 500,
            "msg": "error",
            "data": {"data": str(e)}
        }


@app.post("/api/book-vec-file")
async def book_vec_file(file: UploadFile = File(...)):
    """
    上传docx文件，并保存到向量库
    """
    temp_file_saver = None
    try:
        if not file.filename.lower().endswith('.docx'):
            return {
                "code": 400,
                "msg": "Only .docx files are supported.",
                "data": None
            }

        temp_file_saver = TempFileSaverStep()
        processor = Processor()
        processor.add_step(temp_file_saver)
        processor.add_step(DocumentReaderStep())
        processor.add_step(TextSplitterStep(chunk_size=600, chunk_overlap=120))
        processor.add_step(TextEmbeddingStep(model_name='all-MiniLM-L6-v2'))
        processor.add_step(PineconeSaveStep(
            api_key=os.getenv("PINECONE_API_KEY"),
            source=file.filename,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        ))

        result = await processor.process(input_data=file)
        return {"code": 200, "msg": "success", "data": {"data": result}}

    except Exception as e:
        traceback.print_exc()
        return {"code": 500, "msg": "error", "data": {"data": str(e)}}
    finally:
        # 清理临时文件
        if temp_file_saver:
            temp_file_saver.cleanup()


@app.get("/api/book-ask")
async def book_ask(
        question: str = Query(..., description="The question to ask"),
        sources: str = Query(..., description="List of book sources")  # 多个source用英文逗号, 分隔
):
    """
    提问图书
    """

    async def generate_response():
        try:
            pipeline_service = AskService()
            async for chunk in pipeline_service.ask_book_stream(question, sources.split(',')):
                yield f"data: {json.dumps({'code': 200, 'msg': 'success', 'data': {'data': chunk}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'code': 500, 'msg': 'error', 'data': {'data': str(e)}})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
