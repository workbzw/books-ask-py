from abc import ABC, abstractmethod
import time
from typing import Any, List, Dict
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    """RAG Pipeline类"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        # 初始化Pinecone客户端
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        # 初始化Embedding模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            model_name="deepseek-chat",
            temperature=0.7
        )
        
        # 设置提示词模板
        self.prompt_template = """
        你是一位“严厉的教练型人物”，以强硬的爱来激励别人前进。你能看见别人的潜力，绝不允许他们停留在平庸之中，经常用严酷的真相和夸张的警告来驱使他们采取行动。你的特色是几乎夸张地使用恐吓策略，描绘那些犹豫不决者的黯淡未来。你隐藏的一面是非常关心他人的成功，但用一种高度竞争和苛求的态度掩盖自己的感性，避免显得过于温情。
        请基于以下上下文信息回答问题。
        如果无法从上下文中找到答案，请说"抱歉，我在提供的内容中找不到相关信息"。
        
        上下文信息:
        {context}
        
        问题: {question}
        
        请用中文回答，要求：
        1. 回答要准确、完整、温情
        2. 如果可能，请以朋友的身份问出一个有深度的、启发性的问题，以引导用户继续提问，直接问问题，不需要说出类似：我向你提问等话术
        3. 你非常热爱这份任务，但不需要表述出来
        
        回答:
        """
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def get_relevant_chunks(self, question: str,source:str, top_k: int = 3) -> List[Dict]:
        """检索相关文本块"""
        # 生成问题的向量表示
        print("-----------------time embeding------------------")
        start_time = time.time()
        question_embedding = self.embedding_model.encode([question])[0]
        end_time = time.time()
        print(f"Embedding generation took {end_time - start_time:.2f} seconds")
        print("-----------------time embeding------------------")
        
        # 在Pinecone中检索相似文本
        query_response = self.index.query(
            vector=question_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={'source':source}
        )
        
        return query_response.matches
    
    def generate_prompt(self, question: str, chunks: List[Dict]) -> str:
        """生成提示词"""
        # 从检索结果中提取文本
        context = "\n".join([chunk.metadata["text"] for chunk in chunks])
        
        # 使用模板生成提示词
        return self.prompt.format(context=context, question=question)
    
    def get_llm_response(self, prompt: str) -> str:
        """获取LLM回答"""
        response = self.llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        return response
    
    def query(self, question: str,source:str) -> Dict:
        """执行RAG查询"""
        try:
            # 1. 检索相关文本块
            relevant_chunks = self.get_relevant_chunks(question,source=source)
            
            # 2. 生成提示词
            prompt = self.generate_prompt(question, relevant_chunks)
            
            # 3. 获取LLM回答
            answer = self.get_llm_response(prompt)
            
            return {
                "status": "success",
                "question": question,
                "context": [chunk.metadata["text"] for chunk in relevant_chunks],
                "answer": answer
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 

    async def query_stream(self, question: str,source:str) -> List[Dict]:
        """执行RAG流式查询"""
        try:
            # 1. 检索相关文本块
            relevant_chunks = self.get_relevant_chunks(question,source=source)
            
            # 2. 生成提示词
            prompt = self.generate_prompt(question, relevant_chunks)
            
            # 3. 获取LLM流式回答
            async for chunk in self.llm.astream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield chunk
            
        except Exception as e:
            yield f"Error: {str(e)}" 