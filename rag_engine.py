import re
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI


@dataclass
class Chunk:
    content: str
    metadata: Dict


class RAGEngine:
    def __init__(self, document_path: str, openai_api_key: str):
        """Initialize RAG engine with document and API key"""
        # Load and process document
        with open(document_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        self.chunks = self._split_by_madda_or_table(raw_text)
        
        for c in self.chunks:
            c.content = self._normalize_arabic(c.content)
            c.metadata["number"] = self._extract_number(c.metadata["header"])
            c.metadata["source"] = "Legal_Document"
        
        # Initialize embedder and FAISS
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        
        texts = [c.content for c in self.chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(np.array(embeddings))
        
        self.id2chunk = {i: c for i, c in enumerate(self.chunks)}
        
        # Initialize BM25
        tokenized_corpus = [c.content.split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
    
    def _split_by_madda_or_table(self, text: str) -> List[Chunk]:
        pattern = re.compile(
            r'(?:^|\n)\s*(?:'
            r'(?:المادة\s*[\(\[]?\s*\d+\s*[\)\]]?)|'
            r'(?:[\(\[]?\s*\d+\s*[\)\]]?\s*المادة)|'
            r'(?:الجدول\s*(?:رقم)?\s*[\(\[]?\s*\d+\s*[\)\]]?)|'
            r'(?:جدول\s+\S+)'
            r')',
            re.MULTILINE
        )

        matches = list(pattern.finditer(text))
        chunks = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            header = match.group().strip()

            chunk_type = "article" if "مادة" in header else "table"

            chunks.append(
                Chunk(
                    content=chunk_text,
                    metadata={
                        "type": chunk_type,
                        "header": header
                    }
                )
            )
        return chunks
    
    def _normalize_arabic(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('ـ', '')
        return text.strip()
    
    def _extract_number(self, header: str):
        m = re.search(r'\d+', header)
        return m.group() if m else None
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on Arabic characters"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return "en"
        
        # If more than 30% Arabic characters, consider it Arabic
        return "ar" if (arabic_chars / total_chars) > 0.3 else "en"
    
    # ========== NEW: TRANSLATION METHOD ==========
    def translate_to_arabic(self, text: str) -> str:
        """Translate English text to Arabic using GPT-4o-mini"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a translator. Translate the following text to Arabic. Output ONLY the Arabic translation, no explanations or additional text."
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    def retrieve_chunks(self, query: str, top_k=3):
        # FAISS
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        _, faiss_ids = self.faiss_index.search(q_emb, top_k)

        # BM25
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ids = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]

        # Merge
        candidate_ids = list(set(faiss_ids[0]) | set(bm25_ids))
        return [self.id2chunk[i] for i in candidate_ids]
    
    # ========== MODIFIED: ANSWER QUESTION WITH TRANSLATION ==========
    def answer_question(self, query: str):
        # Detect original language
        original_lang = self.detect_language(query)
        
        # NEW: If query is in English, translate to Arabic for better retrieval
        if original_lang == "en":
            print(f"[INFO] English query detected: '{query}'")
            arabic_query = self.translate_to_arabic(query)
            print(f"[INFO] Translated to Arabic: '{arabic_query}'")
            retrieval_query = arabic_query
        else:
            retrieval_query = query
        
        # Retrieve using Arabic query (whether original or translated)
        retrieved = self.retrieve_chunks(retrieval_query, top_k=1)
        
        context = "\n\n".join(
            f"[{c.metadata['header']}]\n{c.content}"
            for c in retrieved
        )
        
        # Generate answer in the ORIGINAL language
        if original_lang == "ar":
            system_msg = "أنت مساعد قانوني متخصص. أجب بدقة استناداً إلى النصوص المقدمة فقط. كن مختصراً ومباشراً."
            user_prompt = f"""النصوص القانونية:
{context}

السؤال: {query}

أجب بناءً على النصوص أعلاه فقط. إذا لم تجد إجابة واضحة، قل "غير مذكور في الوثيقة"."""
        else:
            system_msg = "You are a legal assistant. Answer accurately based only on the provided texts. Be concise and direct."
            user_prompt = f"""Legal texts:
{context}

Question: {query}

Answer based only on the texts above. If you cannot find a clear answer, say "Not mentioned in the document"."""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content
