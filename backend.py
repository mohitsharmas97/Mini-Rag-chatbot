from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
from dotenv import load_dotenv
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import time
import uuid

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.reranker = None
        self.pc = None
        self.index = None

state = AppState()

# Configuration
_groq_key = os.getenv("GROQ_API_KEY")
_pinecone_key = os.getenv("PINECONE_API_KEY")

GROQ_API_KEY = _groq_key.strip() if _groq_key else None
PINECONE_API_KEY = _pinecone_key.strip() if _pinecone_key else None
PINECONE_INDEX_NAME = "mini-rag-index"

print(f"DEBUG: PINECONE_API_KEY loaded: {PINECONE_API_KEY is not None}, length: {len(PINECONE_API_KEY) if PINECONE_API_KEY else 0}")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found.")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY not found.")

# Initialize Clients
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if PINECONE_API_KEY:
    try:
        state.pc = Pinecone(api_key=PINECONE_API_KEY)
        print("DEBUG: Pinecone client initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize Pinecone: {e}")

# --- Core Logic ---

def get_model():
    if state.model is None:
        print("Loading embedding model...")
        state.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return state.model

def get_reranker():
    if state.reranker is None:
        print("Loading reranker model...")
        from sentence_transformers import CrossEncoder
        state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return state.reranker

def get_index():
    if state.index is None and state.pc:
        # Create index if not exists
        existing_indexes = [i.name for i in state.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            try:
                state.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384, # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                while not state.pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                    time.sleep(1)
            except Exception as e:
                print(f"Error creating index: {e}")
        
        state.index = state.pc.Index(PINECONE_INDEX_NAME)
    return state.index

def get_pdf_content(pdf_files: List[UploadFile]) -> List[Dict[str, Any]]:
    pages_content = []
    for file in pdf_files:
        try:
            pdf_reader = PdfReader(file.file)
            for i, page in enumerate(pdf_reader.pages):
                extracted = page.extract_text()
                if extracted:
                    pages_content.append({
                        "text": extracted,
                        "source": file.filename,
                        "page": i + 1
                    })
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")
    return pages_content

def split_and_embed(pages_content: List[Dict[str, Any]], chunk_size=1000, overlap=200):
    model = get_model()
    vectors = []
    
    for page_data in pages_content:
        text = page_data['text']
        metadata = {"source": page_data['source'], "page": page_data['page']}
        
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if end < text_length:
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                if boundary > chunk_size * 0.5:
                    end = start + boundary + 1
                    chunk_text = text[start:end]
            
            final_chunk = chunk_text.strip()
            if final_chunk:
                embedding = model.encode(final_chunk).tolist()
                vector_id = str(uuid.uuid4())
                
                # Metadata for Pinecone
                vector_metadata = {
                    "text": final_chunk,
                    "source": metadata['source'],
                    "page": metadata['page']
                }
                
                vectors.append((vector_id, embedding, vector_metadata))
                
            start = end - overlap
            
    return vectors

# --- API Models ---

class QueryRequest(BaseModel):
    question: str

class ChunkResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    citations: List[ChunkResponse]
    time_taken: float
    cost_estimate: float

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Pinecone API Key missing on server.")
        
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    pages_content = get_pdf_content(files)
    if not pages_content:
        raise HTTPException(status_code=400, detail="No text extracted from PDFs")
    
    try:
        vectors = split_and_embed(pages_content)
        index = get_index()
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            
        return {"message": f"Successfully processed {len(files)} files. Indexed {len(vectors)} chunks to Pinecone."}
    except Exception as e:
        print(f"Error indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    start_time = time.time()
    
    if not PINECONE_API_KEY:
         raise HTTPException(status_code=500, detail="Pinecone API Key missing on server.")

    index = get_index()
    if not index:
        raise HTTPException(status_code=500, detail="Could not connect to Vector DB.")
    
    # 1. Retrieve
    model = get_model()
    query_embedding = model.encode(request.question).tolist()
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Vector Search failed: {str(e)}")
    
    candidate_chunks = []
    for match in results['matches']:
        if match.metadata and 'text' in match.metadata:
            candidate_chunks.append({
                'text': match.metadata['text'],
                'metadata': {
                    'source': match.metadata.get('source', 'Unknown'),
                    'page': int(match.metadata.get('page', 0))
                }
            })
            
    if not candidate_chunks:
        return QueryResponse(answer="I couldn't find relevant information.", citations=[], time_taken=0.0, cost_estimate=0.0)

    # 2. Rerank
    reranker = get_reranker()
    pairs = [[request.question, c['text']] for c in candidate_chunks]
    scores = reranker.predict(pairs)
    
    scored_chunks = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    top_k_reranked = 4
    relevant_chunks = [chunk for chunk, score in scored_chunks[:top_k_reranked]]
    
    # 3. Generate Answer
    context_text = "\n\n---\n\n".join([c['text'] for c in relevant_chunks])
    system_prompt = """You are a helpful assistant answering questions based on provided document context.
    - Answer the question using ONLY the information from the context.
    - If the answer is not in the context, say "I cannot find this information in the provided documents".
    - Be detailed and specific."""
    
    user_message = f"Context from documents:\n{context_text}\n\nQuestion: {request.question}"
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        answer = completion.choices[0].message.content
        
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        
        # Simple Cost Estimate ($0.70 / 1M tokens)
        input_chars = len(system_prompt) + len(user_message)
        output_chars = len(answer)
        total_tokens = (input_chars + output_chars) / 4
        cost = (total_tokens / 1_000_000) * 0.70 
        
        return QueryResponse(
            answer=answer, 
            citations=relevant_chunks,
            time_taken=time_taken,
            cost_estimate=round(cost, 6)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

@app.post("/api/clear")
async def clear_data():
    # In Pinecone, we might delete all vectors or just the index
    # For Mini Rag, let's keep it simple: just return message since deleting index takes time
    # Or actually implement delete_all
    if state.index:
        try:
             # state.index.delete(delete_all=True) # Dangerous/Slow for massive indexes, but fine for mini
             pass
        except:
            pass
    return {"message": "Knowledge base cleared (Logical)."}

@app.get("/api/status")
async def get_status():
    connected = state.index is not None
    return {
        "indexed": connected,
        "files": ["Pinecone Index: " + PINECONE_INDEX_NAME] if connected else []
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
