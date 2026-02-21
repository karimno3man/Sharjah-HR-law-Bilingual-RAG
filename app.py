<<<<<<< Updated upstream
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
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

# In-memory chat history per session
chat_histories: dict[str, list] = {}

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
        openai_api_key=openai_api_key,
        
    )
    print(f"RAG engine loaded successfully! Total chunks: {len(rag.chunks)}")


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


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
        # Resolve or create session
        session_id = payload.session_id or str(uuid.uuid4())
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        history = chat_histories[session_id]

        # Get answer from RAG (history is updated in-place by answer_question)
        answer = rag.answer_question(payload.question, history=history)

        return {
            "answer": answer,
            "session_id": session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
=======
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import sqlite3
from openai import OpenAI
from rag_engine import RAGEngine
import speech_recognition as sr
import wave

class TTSRequest(BaseModel):
    text: str
    voice: str = "verse"
    model: str = "tts-1"

class LoginRequest(BaseModel):
    username: str
    password: str

DB_PATH = "users.db"

recognizer = sr.Recognizer()


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

def init_user_db():
    """Initialize SQLite DB and seed dummy users."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        dummy_users = ["omar", "hadeer", "adham", "karim", "emad"]
        for username in dummy_users:
            cur.execute(
                """
                INSERT OR IGNORE INTO users (username, password)
                VALUES (?, ?)
                """,
                (username, "P@$$w0rd"),
            )

        conn.commit()
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Return True if username/password match a user in DB."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM users WHERE username = ? AND password = ?",
            (username, password),
        )
        row = cur.fetchone()
        return row is not None
    finally:
        conn.close()

# Serve static assets (CSS / JS / images if you add them later)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global RAG instance
rag = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag
    
    # Initialize the user database
    init_user_db()
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    # Initialize RAG engine
    print("Loading RAG engine...")
    rag = RAGEngine(
        document_path="arabic_text_and_tables.txt",
        openai_api_key=openai_api_key,
        
    )
    print(f"RAG engine loaded successfully! Total chunks: {len(rag.chunks)}")


from typing import Optional

class AskRequest(BaseModel):
    question: str
    username: str
    conversation_id: Optional[int] = None


@app.get("/")
def root():
    """Serve the root UI (login page)"""
    return FileResponse("static/login.html")

@app.get("/chat")
def chat_ui():
    """Serve the chat UI"""
    return FileResponse("static/chat.html")

@app.post("/api/login")
def login(payload: LoginRequest):
    """API endpoint to validate user credentials"""
    if verify_user(payload.username, payload.password):
        return {"status": "success", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")


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
    """Main RAG endpoint - answer questions and persist history"""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Validate input
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        
        # Get user ID
        cur.execute("SELECT id FROM users WHERE username = ?", (payload.username,))
        user_row = cur.fetchone()
        if not user_row:
            raise HTTPException(status_code=401, detail="User not found")
        user_id = user_row[0]
        
        # Handle conversation creation
        from datetime import datetime
        now = datetime.now().isoformat()
        conv_id = payload.conversation_id
        
        if not conv_id:
            # Create a new conversation title from a snippet of the question
            title = payload.question[:30] + "..." if len(payload.question) > 30 else payload.question
            cur.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, title, now, now)
            )
            conv_id = cur.lastrowid
        else:
            # Update the existing conversation's updated_at timestamp
            cur.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            
        # Insert user message
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, "user", payload.question, now)
        )
            
        # Get answer from RAG
        answer = rag.answer_question(payload.question)
        
        # Insert assistant message
        answer_now = datetime.now().isoformat()
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, "assistant", answer, answer_now)
        )
        
        conn.commit()
        
        return {
            "answer": answer,
            "conversation_id": conv_id
        }
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    finally:
        conn.close()

@app.get("/api/conversations")
def get_conversations(username: str):
    """Get all conversations for a specific user"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_row = cur.fetchone()
        if not user_row:
            raise HTTPException(status_code=401, detail="User not found")
        
        cur.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_row[0],)
        )
        rows = cur.fetchall()
        return [{"id": row[0], "title": row[1], "updated_at": row[2]} for row in rows]
    finally:
        conn.close()

@app.get("/api/conversations/{conv_id}/messages")
def get_messages(conv_id: int):
    """Get all messages for a specific conversation"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,)
        )
        rows = cur.fetchall()
        return [{"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]} for row in rows]
    finally:
        conn.close()


# New endpoint sketch
@app.post("/tts")
async def text_to_speech(payload: TTSRequest):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=payload.text
    )
    audio_bytes = response.content
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")

# STT endpoint
@app.get("/listen")
def speech_to_text():
    try:
        mic = sr.Microphone(sample_rate=16000)
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.listen(source, phrase_time_limit=10)

        pcm_bytes = audio.get_raw_data()
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
            response_format="verbose_json"
        )

        print(f"Detected Language: {result.language}")
        print(f"Text: {result.text}")

        return {"language": result.language, "text": result.text}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

>>>>>>> Stashed changes
