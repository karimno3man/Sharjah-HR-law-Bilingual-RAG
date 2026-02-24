# Sharjah HR Law - Bilingual RAG

A bilingual (Arabic/English) legal assistant for Sharjah's HR law. Ask questions in either language, get answers grounded in the actual legal text.

## How It Works

1. The Arabic legal document is chunked by articles and tables
2. Chunks are indexed using **FAISS** (semantic) + **BM25** (keyword) search
3. Both results are merged via **Reciprocal Rank Fusion (RRF)**
4. **GPT-4o-mini** generates the final answer from the retrieved chunks
5. English queries are auto-translated to Arabic for better retrieval

## Setup

```bash
git clone https://github.com/karimno3man/Sharjah-HR-law-Bilingual-RAG.git
cd Sharjah-HR-law-Bilingual-RAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
$env:OPENAI_API_KEY="sk-your-key-here"
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Run with Docker

If you prefer to run the project using Docker, follow these steps:

1. Create a `.env` file in the project root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   ```

2. Build the Docker containers:
   ```bash
   docker-compose build
   ```

3. Start the application:
   ```bash
   docker-compose up
   ```

4. After starting the containers, open [http://localhost:8000](http://localhost:8000) in your browser to access the chat interface.

5. Stop the application when needed:
   ```bash
   docker-compose down
   ```

## Project Structure

```
app.py                      # FastAPI server
rag_engine.py               # RAG engine (chunking, retrieval, generation)
arabic_text_and_tables.txt  # Source legal document
static/index.html           # Chat UI
requirements.txt            # Python dependencies
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Chat UI |
| POST   | `/ask`   | Ask a question (`{"question": "..."}`) |


## Tech Stack

- **FastAPI** - web server
- **Sentence Transformers** (`multilingual-e5-large`) - embeddings
- **FAISS** - vector search
- **BM25** - keyword search
- **OpenAI GPT-4o-mini** - translation + answer generation
