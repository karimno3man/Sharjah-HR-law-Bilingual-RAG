from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from rag_engine import RAGEngine

app = FastAPI(
    title='Sharjah Rag',
    version='0.1.0',
)

# Allow CORS (useful if you ever call from a different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS / JS / images if you add them later)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global RAG instance
rag = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    # Initialize RAG engine
    print("Loading RAG engine...")
    rag = RAGEngine(
        document_path="arabic_text_and_tables.txt",
        openai_api_key=openai_api_key
    )
    print(f"RAG engine loaded successfully! Total chunks: {len(rag.chunks)}")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def root():
    """Serve the chat UI"""
    return FileResponse("static/index.html")


@app.get("/api/info")
def api_info():
    """API info endpoint"""
    return {
        "message": "Sharjah RAG API",
        "version": "0.1.0",
        "endpoints": {
            "POST /ask": "Ask a question",
            "GET /ping": "Health check",
            "GET /stats": "Get system stats"
        }
    }


@app.get("/ping")
def ping():
    """Health check endpoint"""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "status": "healthy",
        "chunks_loaded": len(rag.chunks)
    }


@app.get("/stats")
def stats():
    """Get RAG statistics"""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "total_chunks": len(rag.chunks),
        "embedding_model": "intfloat/multilingual-e5-large",
        "llm_model": "gpt-4o-mini",
        "retrieval_methods": ["FAISS", "BM25"]
    }


@app.post('/ask')
def ask(payload: AskRequest):
    """Main RAG endpoint - answer questions"""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Validate input
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get answer from RAG
        answer = rag.answer_question(payload.question)
        
        return {
            "answer": answer
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
