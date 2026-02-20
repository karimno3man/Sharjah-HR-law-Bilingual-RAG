from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag_engine import RAGEngine


DB_PATH = os.path.join(os.path.dirname(__file__), "operational_db")

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


def get_user_id(username: str) -> int:
    """Get user ID for a given username, or raise HTTPException if not found."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Unknown user")
        return row[0]
    finally:
        conn.close()

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag

    # Initialize / seed users database
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


class AskRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = []
    username: Optional[str] = None
    conversation_id: Optional[int] = None


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def root():
    """Serve the login page"""
    return FileResponse("static/login.html")


@app.get("/chat")
def chat_page():
    """Serve the chat UI"""
    return FileResponse("static/index.html")


@app.post("/api/login")
def login(payload: LoginRequest):
    """Simple username/password login using SQLite users table."""
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    if not verify_user(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"success": True}


@app.get("/api/conversations")
def list_conversations(username: str):
    """List conversations for the given user, newest first."""
    user_id = get_user_id(username)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, username: str):
    """Get a single conversation with its messages for the given user."""
    user_id = get_user_id(username)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )
        conv = cur.fetchone()
        if not conv or conv[1] != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        cur.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        )
        msgs = cur.fetchall()

        return {
            "id": conv[0],
            "title": conv[2],
            "created_at": conv[3],
            "updated_at": conv[4],
            "messages": [
                {
                    "role": m[0],
                    "content": m[1],
                    "created_at": m[2],
                }
                for m in msgs
            ],
        }
    finally:
        conn.close()


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
        "embedding_model": "BAAI/bge-m3",
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

    if not payload.username:
        raise HTTPException(status_code=400, detail="Username is required for conversations")

    try:
        # Get answer from RAG
        answer = rag.answer_question(payload.question, history=payload.history, debug=True)

        # Persist conversation and messages
        user_id = get_user_id(payload.username)
        conversation_id = payload.conversation_id
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()

            # Create conversation if needed
            if conversation_id is None:
                title = payload.question.strip()
                if len(title) > 80:
                    title = title[:77] + "..."
                cur.execute(
                    """
                    INSERT INTO conversations (user_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, title or "New conversation", now, now),
                )
                conversation_id = cur.lastrowid

            # Insert user and assistant messages
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, "user", payload.question, now),
            )
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, "assistant", answer, now),
            )

            # Update conversation timestamp
            cur.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

            conn.commit()
        finally:
            conn.close()

        return {
            "answer": answer,
            "conversation_id": conversation_id,
        }

    except HTTPException:
        # Re-raise HTTPException directly
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
