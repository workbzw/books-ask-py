# 启动项目: 
sh start.sh
# 停止项目: 
sh stop.sh

# api:

## 提问并得到steam方式的回答
http://localhost:8000/api/book-ask?question=你叫什么名字&sources=book_name_1,book_name_2

## 上传文件并向量化存储:
method: post
http://localhost:8000/api/book-vec-file

## 给到url，利用爬虫获取data，并向量化存储:
method: post
params: url	
http://localhost:8000/api/book-vec-url