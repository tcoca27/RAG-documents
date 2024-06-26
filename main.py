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