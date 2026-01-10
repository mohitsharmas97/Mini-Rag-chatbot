from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
from groq import Groq
from pinecone import Pinecone
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "mini-rag-index"

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Global variables for lazy loading
index = None
embedding_model = None
reranker_model = None

def get_index():
    """Lazy load Pinecone index"""
    global index
    if index is None:
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            print(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        except:
            print(f"Index not found, will be created on first upload")
    return index

def get_embedding_model():
    """Lazy load embedding model"""
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
    return embedding_model

def get_reranker():
    """Lazy load reranker model"""
    global reranker_model
    if reranker_model is None:
        print("Loading reranker model...")
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✓ Reranker model loaded")
    return reranker_model

# Helper functions
def extract_pdf_text(file):
    """Extract text from PDF file"""
    pages_content = []
    try:
        pdf_reader = PdfReader(file)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                pages_content.append({
                    "text": text,
                    "source": secure_filename(file.filename),
                    "page": i + 1
                })
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pages_content

def chunk_and_embed(pages_content, chunk_size=1000, overlap=200):
    """Split text into chunks and create embeddings"""
    model = get_embedding_model()
    vectors = []
    
    for page_data in pages_content:
        text = page_data['text']
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            
            if chunk:
                embedding = model.encode(chunk).tolist()
                vector_id = str(uuid.uuid4())
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "source": page_data['source'],
                        "page": page_data['page']
                    }
                })
            
            start = end - overlap
    
    return vectors

# Routes
@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process PDF files"""
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    all_pages = []
    
    # Extract text from all PDFs
    for file in files:
        if file.filename.endswith('.pdf'):
            pages = extract_pdf_text(file)
            all_pages.extend(pages)
    
    if not all_pages:
        return jsonify({"error": "No text extracted from PDFs"}), 400
    
    # Create embeddings and upload to Pinecone
    try:
        vectors = chunk_and_embed(all_pages)
        idx = get_index()
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            idx.upsert(vectors=batch)
        
        return jsonify({
            "message": f"Successfully processed {len(files)} files. Indexed {len(vectors)} chunks."
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer questions using RAG"""
    start_time = time.time()
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # 1. Get query embedding
        model = get_embedding_model()
        query_embedding = model.encode(question).tolist()
        
        # 2. Search Pinecone
        idx = get_index()
        results = idx.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        # 3. Extract candidates
        candidates = []
        for match in results['matches']:
            if match.metadata and 'text' in match.metadata:
                candidates.append({
                    'text': match.metadata['text'],
                    'metadata': {
                        'source': match.metadata.get('source', 'Unknown'),
                        'page': int(match.metadata.get('page', 0))
                    }
                })
        
        if not candidates:
            return jsonify({
                "answer": "I couldn't find relevant information.",
                "citations": [],
                "time_taken": 0.0,
                "cost_estimate": 0.0
            })
        
        # 4. Rerank
        reranker = get_reranker()
        pairs = [[question, c['text']] for c in candidates]
        scores = reranker.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in scored[:4]]
        
        # 5. Generate answer with Groq
        context = "\n\n---\n\n".join([c['text'] for c in top_chunks])
        
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions using ONLY the provided context. If the answer is not in the context, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        
        answer = completion.choices[0].message.content
        time_taken = round(time.time() - start_time, 2)
        
        # Simple cost estimate
        total_chars = len(context) + len(question) + len(answer)
        cost = (total_chars / 4 / 1_000_000) * 0.70
        
        return jsonify({
            "answer": answer,
            "citations": top_chunks,
            "time_taken": time_taken,
            "cost_estimate": round(cost, 6)
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        idx = get_index()
        stats = idx.describe_index_stats()
        return jsonify({
            "indexed": True,
            "files": [f"Pinecone Index: {PINECONE_INDEX_NAME}"],
            "vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0
        })
    except:
        return jsonify({"indexed": False, "files": []})

@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all data from index"""
    try:
        idx = get_index()
        idx.delete(delete_all=True)
        return jsonify({"message": "Knowledge base cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Mini RAG Chatbot Server Starting...")
    print("="*50)
    print(f"📍 Server: http://127.0.0.1:5000")
    print(f"📊 Pinecone Index: {PINECONE_INDEX_NAME}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
