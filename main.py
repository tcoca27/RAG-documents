# import uvicorn
# import lancedb
# from fastapi import Body, FastAPI, HTTPException, Query, status, middleware, Request, Response
# from fastapi.responses import StreamingResponse 
# from contextlib import asynccontextmanager
# import time
# from datetime import datetime
# from uuid import uuid4
# from typing import Callable, Literal, Annotated
# from pydantic import (BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt,
#                       computed_field, field_validator, model_validator)
# from datetime import datetime
# from uuid import uuid4
# import ollama
# from db_init import Product


# class ModelRequest(BaseModel):
#     prompt: Annotated[str, Field(min_length=1, max_length=10000)]  

# class TextModelRequest(ModelRequest):
#     temperature: Annotated[float, Field(gte=0.0, lte=1.0, default=0.0)]  

# class ModelResponse(BaseModel):
#     request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]  
#     content: Annotated[str, Field(min_length=0, max_length=10000)]  
#     created_at: datetime = datetime.now()

# db: lancedb.DBConnection = lancedb.connect("db")
# table = db.open_table("adidas")
# llm = None

# # @asynccontextmanager  
# # def lifespan(app: FastAPI):
# #     db = lancedb.connect("db")

# #     yield  
    

# app = FastAPI()

# @app.middleware("http")  
# async def monitor_service(req: Request, call_next: Callable) -> Response:  
#     start_time = time.time()
#     response: Response = await call_next(req)
#     response_time = round(time.time() - start_time, 4)  
#     request_id = uuid4().hex  
#     response.headers["X-Response-Time"] = str(response_time)
#     response.headers["X-API-Request-ID"] = request_id  
#     with open("usage.log", "a") as file:  
#         file.write(
#             f"Request ID: {request_id}"
#             f"\nDatetime: {datetime.utcnow().isoformat()}"
#             f"\nEndpoint triggered: {req.url}"
#             f"\nClient IP Address: {req.client.host}"
#             f"\nResponse time: {response_time} seconds"
#             f"\nStatus Code: {response.status_code}"
#             f"\nSuccessful: {response.status_code < 400}\n\n"
#         )
#     return response

# @app.get("/")
# def root_controller():
#   return {"status": "healthy"}

# @app.post("/generate/text")  
# def serve_text_to_text_controller(request: Request, body: TextModelRequest = Body(...)) -> ModelResponse: 
#     use_kb = ollama.generate(model='llama3', prompt=f'You will receive a user prompt. You have access to a knowledge base of clothing products. Based on the received prompt you have to decide if you should use the database of products to recommend something to the user. Here is the prompt, output only "True" if you should use the knowledge base and only "False" otherwise: {body.prompt}' )
#     print(use_kb['response'])
#     response = {}
#     if bool(use_kb['response']):
#         results = table.search(body.prompt).limit(5).to_pydantic(Product)
#         print(results)
#         ollama_response = ollama.generate(model='llama3', prompt=f'You will receive a user prompt asking something about clothing products or requesting a clothing product and also a list of the 5 most relevant products. Recommend the user the products. User query: {body.prompt}; Products (split by | character): {"|".join(str(r) for r in results)}. Your answer should have the form Query Answer: [your answer for the user] and then Used Products: [list of indexes of products used, split by , character]' )
#         print(ollama_response['response'])
#         response = ollama_response['response']
#         query_ans = response.split("Used Products: ")[0]
#         used_products = response.split("Used Products: ")[1]
#         return ModelResponse(content={"query_answer": query_ans, "used_products": used_products})
#     else:
#         ollama_response = ollama.generate(model='llama3', prompt=f'You will receive a user prompt. You are specialized in being a sales agent for a clothing store. If the prompt is not about clothes, please do your best to stir the user in a direction in which you can answer. User query: {body.prompt}' )    
#         response = ollama_response['response']
#     return ModelResponse(content=response)

# if __name__ == "__main__":
#   uvicorn.run("main:app", port=8000, reload=True)


# documents = SimpleDirectoryReader("./data/carry1st/").load_data()
# if len(documents):
#     vector_store = LanceDBVectorStore(uri="./lancedb", mode="overwrite", query_type="hybrid")
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     index = VectorStoreIndex.from_documents(
#         documents, storage_context=storage_context
#     )


import mimetypes
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import lancedb.rerankers
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import lancedb
import asyncio
import aiofiles
import os
from lancedb.pydantic import LanceModel, Vector
import constants
import pyarrow as pa

app = FastAPI()

main_llm = Ollama(model="llama3", request_timeout=120.0)
summarizer_llm = Ollama(model="llama3",  request_timeout=120.0)
rewriter_llm = Ollama(model="llama3",  request_timeout=120.0)
generalist_llm = Ollama(model="llama3",  request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

class Document(LanceModel):
    vector: Vector(768)
    text: str
    id: str

reranker = lancedb.rerankers.ColbertReranker()
db = lancedb.connect("./lancedb")
table = db.create_table("carry1st", schema=Document, mode="overwrite")
vector_store = LanceDBVectorStore(uri="./lancedb", table=table)
# vector_store._add_reranker(reranker)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
index = None
service_context = ServiceContext.from_defaults(llm=main_llm, embed_model=embed_model)
try:
    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, vector_store=vector_store)
except:
    index = VectorStoreIndex.from_documents([], service_context=service_context, vector_store=vector_store)

chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 3

class DocumentsResponse(BaseModel):    
    text: str
    source_name: Optional[str] = None
    source_path: Optional[str] = None
    page: Optional[int] = None
    
class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[DocumentsResponse]] = None

async def save_file(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return file_path

def is_valid_file(file: UploadFile) -> bool:
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in constants.ALLOWED_EXTENSIONS:
        return False
    
    if file.size > constants.MAX_FILE_SIZE:
        return False
    
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type not in ['text/plain', 'application/pdf', 'application/msword', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                         'text/csv']:
        return False
    
    return True

@app.get("/upload", response_model=List[str])
async def uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    print(files)
    return files

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    valid_files = [file for file in files if is_valid_file(file)]
    if not valid_files:
        raise HTTPException(status_code=400, detail="No valid files were uploaded")

    try:
        file_paths = await asyncio.gather(*[save_file(file) for file in valid_files])
        new_documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        for doc in new_documents:
            index.insert(doc)

        return JSONResponse(
            content={
                "message": f"{len(valid_files)} file(s) uploaded and indexed successfully",
                "files": [file.filename for file in valid_files],
                "skipped_files": [file.filename for file in files if file not in valid_files]
            },
            status_code=200
        )
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return JSONResponse(
            content={"message": "An error occurred while processing the files"},
            status_code=500
        )
    finally:
        for file in files:
            await file.close()

def summarize(text: str) -> str:
    summary = summarizer_llm.complete(constants.SUMMARIZE_PROMPT.format(chat_history=chat_memory.get_all(), query=text))
    return summary.text

def rewrite(text: str) -> str:
    rewritten = rewriter_llm.complete(constants.REWRITE_PROMPT.format(chat_history=chat_memory.get_all(), query=text))
    return rewritten.text

def generalist(query: str) -> str:
    response = generalist_llm.complete(constants.GENERALIST_PROMPT.format(chat_history=chat_memory.get_all(), query=query))
    return response.text

def retrieve_and_synthesize(query: str, top_k: int = 3) -> tuple:
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    results = query_engine.retrieve(query)
    res_rerank = reranker.rerank_fts(query=query, fts_results=pa.table([pa.array([r.text for r in results]), pa.array([r.score for r in results]), pa.array([i for i in range(len(results))])], names=['text', 'score', 'index'])).to_pandas()
    filtered_results = []
    for _, row in res_rerank.iterrows():
        original_index = row['index']
        if row['relevance_score'] < constants.SOURCE_THRESHOLD:
            filtered_results.append(results[original_index].text)
    retrieved_info = "\n".join(filtered_results)
    synthesized = main_llm.complete(constants.RETRIEVE_SYNTHESIZE_PROMPT.format(query=query, retrieved_info=retrieved_info, company_name=constants.COMPANY_NAME))
    sources = [DocumentsResponse(text=node.text, source_name=node.metadata['file_name'], source_path=node.metadata['file_path'], page=node.metadata.get('page_label')) for node in results if node.score > constants.SOURCE_THRESHOLD]
    return synthesized.text, sources

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    query = request.message
    chat_history = chat_memory.get_all()
    
    tool_decision = main_llm.complete(constants.MAIN_PROMPT.format(chat_history=chat_history, query=query)).text.strip().lower()
    print(tool_decision)
    
    if "summarize" in tool_decision:
        response = summarize(query)
        if "generalist" in response:
            print(response)
            response = generalist(query)
    elif "rewrite" in tool_decision:
        response = rewrite(query)
        if "generalist" in response:
            print(response)
            response = generalist(query)
    elif "retrieve" in tool_decision:
        response, sources = retrieve_and_synthesize(query, request.top_k)
    elif "generalist" in tool_decision:
        response = generalist(query)
    else:
        response = "I'm not sure how to process that request. Could you please rephrase or ask something else?"
        sources = None
        
    chat_memory.put(ChatMessage(role="user", content=query))
    chat_memory.put(ChatMessage(role="assistant", content=response))
    
    return ChatResponse(response=response, sources=sources if ('sources' in locals() and len(sources) > 0) else None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)