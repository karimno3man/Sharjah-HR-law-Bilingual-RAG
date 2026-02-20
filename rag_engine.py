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


# ========== NEW CHUNKING FUNCTIONS (INTEGRATED) ==========

def normalize_arabic_indic(text: str) -> str:
    """Convert Arabic-Indic numerals to Western numerals"""
    arabic_indic = '٠١٢٣٤٥٦٧٨٩'
    western = '0123456789'
    translation = str.maketrans(arabic_indic, western)
    return text.translate(translation)


def is_article_header(line: str) -> bool:
    """Determine if a line is an article header (not a reference)"""
    stripped = line.strip()
    
    # Too long to be a header
    if len(stripped) > 200:
        return False
    
    # Skip obvious references - handle both "المادة" and "الما دة" (with space)
    reference_patterns = [
        'بالمادة', 'من المادة', 'من هذه المادة',
        'في المادة', 'على المادة', 'وفق المادة',
        'المادة رقم', 'نص المادة',
        'بالما دة', 'من الما دة', 'من هذه الما دة',
        'في الما دة', 'على الما دة', 'وفق الما دة',
        'الما دة رقم', 'نص الما دة'
    ]
    
    for pattern in reference_patterns:
        if pattern in stripped:
            return False
    
    # Check if it matches article header patterns
    # Handle both "المادة" and "الما دة" (with space between ال and مادة)
    if re.match(r'^الما?\s*دة\s*[\(\[]?\s*\d+', stripped):
        return True
    
    if re.match(r'^[\(\[]?[^\(\)\[\]]*\d+\s*[\)\]]\s*الما?\s*دة', stripped):
        return True
    
    if stripped.endswith(('المادة', 'الما دة')) and re.search(r'\d+', stripped):
        return True
    
    return False


def is_table_header(line: str) -> bool:
    """Determine if a line is a table header"""
    stripped = line.strip()
    
    # Check for various table header patterns:
    # - ## جدول رقم (X)
    # - ### جدول رقم (X)
    # - ### **الجدول رقم (X)**
    # - **جدول رقم (X):**
    # - **الجدول رقم (X):**
    # - جدول رقم (X):
    # - استكمال جدول رقم (X) (continuations)
    
    # Pattern 1: Markdown headers (## or ###) followed by table reference
    if re.match(r'^#{2,3}\s*(\*\*)?.*?(ال)?جدول رقم\s*\(', stripped):
        return True
    
    # Pattern 2: Bold markdown table headers
    if re.match(r'^\*\*.*?(ال)?جدول رقم\s*\(', stripped):
        return True
    
    # Pattern 3: Plain table headers starting with جدول رقم
    if re.match(r'^(ال)?جدول رقم\s*\(', stripped):
        # Skip references to tables (not actual table headers)
        if any(pattern in stripped for pattern in ['بالجدول', 'من الجدول', 'في الجدول', 'على الجدول', 'للجدول', 'وفق الجدول', 'وفقا للجدول']):
            return False
        return True
    
    # Pattern 4: Table continuations (استكمال)
    if 'استكمال جدول رقم' in stripped:
        return True
    
    return False


def chunk_by_article_and_table(text: str, normalize_numerals: bool = True) -> List[Chunk]:
    """
    Chunk Arabic legal document by articles (المادة) and tables (جدول).
    
    Handles formats:
    - المادة (5), (5) المادة, )text 5( المادة
    - جدول المخالفات, جدول وظائف المهندسين
    """
    
    # Normalize Arabic-Indic numerals
    if normalize_numerals:
        text = normalize_arabic_indic(text)
    
    lines = text.split('\n')
    chunk_starts = []
    
    for i, line in enumerate(lines):
        if is_article_header(line):
            chunk_starts.append((i, line.strip(), 'article'))
        elif is_table_header(line):
            chunk_starts.append((i, line.strip(), 'table'))
    
    # Create chunks
    chunks = []
    
    for idx, (line_num, header, chunk_type) in enumerate(chunk_starts):
        start_line = line_num
        end_line = chunk_starts[idx + 1][0] if idx + 1 < len(chunk_starts) else len(lines)
        
        chunk_lines = lines[start_line:end_line]
        chunk_text = '\n'.join(chunk_lines).strip()
        
        # Extract number
        number_match = re.search(r'\d+', header)
        number = number_match.group() if number_match else None
        
        # Calculate positions
        start_pos = sum(len(l) + 1 for l in lines[:start_line])
        end_pos = start_pos + len(chunk_text)
        
        chunks.append(
            Chunk(
                content=chunk_text,
                metadata={
                    'type': chunk_type,
                    'header': header,
                    'number': number,
                    'chunk_size': len(chunk_text),
                    'start_line': start_line,
                    'end_line': end_line,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'source': 'Legal_Document'
                }
            )
        )
    
    return chunks


def sub_chunk_large_articles(chunks: List[Chunk], max_size: int) -> List[Chunk]:
    """Split large chunks into smaller sub-chunks"""
    new_chunks = []
    
    for chunk in chunks:
        if chunk.metadata['chunk_size'] <= max_size:
            new_chunks.append(chunk)
        else:
            # Split by paragraphs
            paragraphs = chunk.content.split('\n')
            
            sub_chunk_content = ""
            sub_chunk_index = 1
            
            for para in paragraphs:
                if len(sub_chunk_content) + len(para) > max_size and sub_chunk_content:
                    new_chunks.append(
                        Chunk(
                            content=sub_chunk_content.strip(),
                            metadata={
                                **chunk.metadata,
                                'subsection': f'part_{sub_chunk_index}',
                                'chunk_size': len(sub_chunk_content),
                                'is_subchunk': True,
                                'original_size': chunk.metadata['chunk_size']
                            }
                        )
                    )
                    sub_chunk_content = para + "\n"
                    sub_chunk_index += 1
                else:
                    sub_chunk_content += para + "\n"
            
            # Add the last sub-chunk
            if sub_chunk_content.strip():
                new_chunks.append(
                    Chunk(
                        content=sub_chunk_content.strip(),
                        metadata={
                            **chunk.metadata,
                            'subsection': f'part_{sub_chunk_index}',
                            'chunk_size': len(sub_chunk_content),
                            'is_subchunk': True,
                            'original_size': chunk.metadata['chunk_size']
                        }
                    )
                )
    
    return new_chunks


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text"""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('ـ', '')
    return text.strip()


# ========== RAG ENGINE (UPDATED WITH NEW CHUNKING) ==========

class RAGEngine:
    def __init__(self, document_path: str, openai_api_key: str, max_chunk_size: int = 5000):
        """Initialize RAG engine with document and API key"""
        print("Loading and chunking document...")
        
        # Load document
        with open(document_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # NEW: Use the improved chunking function
        self.chunks = chunk_by_article_and_table(raw_text, normalize_numerals=True)
        
        # Normalize text in chunks
        for c in self.chunks:
            c.content = normalize_arabic_text(c.content)
        
        # NEW: Sub-chunk large articles
        if max_chunk_size:
            print(f"Sub-chunking articles larger than {max_chunk_size} characters...")
            original_count = len(self.chunks)
            self.chunks = sub_chunk_large_articles(self.chunks, max_chunk_size)
            print(f"Chunks: {original_count} → {len(self.chunks)} (after sub-chunking)")
        
        # Print chunking summary
        articles = sum(1 for c in self.chunks if c.metadata['type'] == 'article')
        tables = sum(1 for c in self.chunks if c.metadata['type'] == 'table')
        print(f"Total chunks: {len(self.chunks)} ({articles} articles, {tables} tables)")
        
        # Initialize embedder and FAISS
        print("Creating embeddings...")
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        
        texts = [c.content for c in self.chunks]
        passage_texts = [f"passage: {text}" for text in texts]
        embeddings = self.embedder.encode(passage_texts, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(np.array(embeddings))
        
        self.id2chunk = {i: c for i, c in enumerate(self.chunks)}
        
        # Initialize BM25
        print("Initializing BM25...")
        tokenized_corpus = [c.content.split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Conversation memory (current session only)
        self.conversation_history: List[Dict] = []
        self.max_history_turns = 6  # Keep last 6 turns (3 Q&A pairs)
        
        print("✅ RAG engine ready!")
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on Arabic characters"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return "en"
        
        return "ar" if (arabic_chars / total_chars) > 0.3 else "en"
    
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
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    
    def rewrite_query_with_history(self, query: str) -> str:
        """
        If there's conversation history, rewrite the query as a fully standalone question.
        This ensures follow-up questions like 'what about exceptions?' retrieve correctly.
        """
        if not self.conversation_history:
            return query  # No history yet, use query as-is
        
        # Build a compact history string
        history_text = ""
        for turn in self.conversation_history[-self.max_history_turns:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query rewriter. Given a conversation history and a follow-up question, "
                        "rewrite the follow-up question as a single, fully self-contained question "
                        "that includes all necessary context from the history. "
                        "Output ONLY the rewritten question, nothing else. "
                        "Keep the same language as the follow-up question."
                    )
                },
                {
                    "role": "user",
                    "content": f"Conversation history:\n{history_text}\nFollow-up question: {query}\n\nRewritten standalone question:"
                }
            ],
            temperature=0,
            max_tokens=200
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten

    def clear_history(self):
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared.")

    def retrieve_chunks(self, query: str, top_k=5, k=60):
        """
        Retrieve relevant chunks using Reciprocal Rank Fusion (RRF)
        
        RRF combines rankings from FAISS and BM25 using the formula:
        RRF_score(d) = sum over all rankings( 1 / (k + rank(d)) )
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            k: RRF constant (default 60, typical range 30-100)
        """
        # FAISS semantic search
        query_text = f"query: {query}"
        q_emb = self.embedder.encode([query_text], normalize_embeddings=True)
        
        # Get more candidates for better fusion
        search_k = min(top_k * 2, len(self.chunks))
        faiss_scores, faiss_ids = self.faiss_index.search(q_emb, search_k)

        # Convert FAISS results to Python native types and filter out invalid indices (-1)
        faiss_ids_list = [int(idx) for idx in faiss_ids[0] if idx >= 0]

        # BM25 keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranking = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:search_k]

        # RRF: Reciprocal Rank Fusion
        rrf_scores = {}
        
        # Add FAISS rankings to RRF
        for rank, idx in enumerate(faiss_ids_list):
            if idx in self.id2chunk:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Add BM25 rankings to RRF
        for rank, idx in enumerate(bm25_ranking):
            if idx in self.id2chunk:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Handle case where no chunks were retrieved
        if not rrf_scores:
            return []
        
        # Sort by RRF score (higher is better)
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k chunks
        return [
            self.id2chunk[idx] 
            for idx, score in sorted_chunks[:top_k]
            if idx in self.id2chunk
        ]
    
    def answer_question(self, query: str, debug=False):
        """Answer a question using RAG with conversation memory."""
        # Detect original language
        original_lang = self.detect_language(query)

        # Step 1: Rewrite query using conversation history for better retrieval
        standalone_query = self.rewrite_query_with_history(query)
        if debug and standalone_query != query:
            print(f"[DEBUG] Original query: '{query}'")
            print(f"[DEBUG] Rewritten query: '{standalone_query}'")

        # Translate to Arabic if needed for better retrieval
        if original_lang == "en":
            if debug:
                print(f"[DEBUG] English query detected: '{standalone_query}'")
            arabic_query = self.translate_to_arabic(standalone_query)
            if debug:
                print(f"[DEBUG] Translated to Arabic: '{arabic_query}'")
            retrieval_query = arabic_query
        else:
            retrieval_query = standalone_query

        # Retrieve chunks
        retrieved = self.retrieve_chunks(retrieval_query, top_k=10)
        
        # Handle case where no chunks were retrieved
        if not retrieved:
            if original_lang == "ar":
                return "عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك في الوثيقة."
            else:
                return "Sorry, I couldn't find relevant information for your question in the document."
        
        if debug:
            print(f"\n[DEBUG] Retrieved {len(retrieved)} chunks:")
            for i, c in enumerate(retrieved[:5], 1):
                subsection = f" ({c.metadata.get('subsection')})" if c.metadata.get('subsection') else ""
                print(f"  {i}. {c.metadata['header']}{subsection}")
                print(f"     Size: {c.metadata['chunk_size']} chars")
        
        # Build context
        context = "\n\n".join(
            f"[{c.metadata['header']}]\n{c.content}"
            for c in retrieved
        )
        
        # Generate answer in original language
        if original_lang == "ar":
            system_msg = """أنت مساعد قانوني متخصص.
            مهمتك هي استخراج جميع المعلومات ذات الصلة من النصوص المقدمة فقط.
            لا تفترض ولا تضف أي معلومة من خارج النص.

            يجب عليك:
            - الإجابة بنفس لغة السؤال تماماً (إذا كان السؤال بالعربية يجب أن تكون الإجابة بالعربية فقط).
            - عدم استخدام أي لغة أخرى في الإجابة.

            إذا كان السؤال يطلب عناصر متعددة (نقاط، شروط، فئات، رواتب، إلخ) فيجب عليك:
            - استخراج جميع العناصر المذكورة كاملة بدون حذف أي نقطة.
            - عدم تكرار أي نقطة.
            - عرض الإجابة في نقاط واضحة إذا كانت موجودة كنقاط في النص.

           """

            user_prompt = f"""النصوص القانونية:

            {context}

             السؤال:
            {query}

            تعليمات مهمة:
            - اقرأ جميع النصوص بعناية.
            - استخرج كل العناصر المرتبطة بالسؤال بالكامل.
            - لا تحذف أي بند مذكور في النص.
            - لا تكرر أي بند.
            - لا تعتمد على الفهم العام، فقط على النص الحرفي أعلاه.
            - في حال اختلاف صياغة السؤال عن النص (مثل اختلاف بسيط في الكلمات)، اعتبر المعنى المقصود إذا كان واضحاً من السياق.
            - يجب أن تكون الإجابة بنفس لغة السؤال فقط.

            أجب الآن."""

        else:
            system_msg = """You are a professional legal assistant.
            Your task is to extract ALL relevant information strictly from the provided texts.
            Do NOT add information from outside the texts.

            CRITICAL LANGUAGE RULE: The source texts are in Arabic, but the user's question is in English.
            You MUST translate all relevant information from Arabic to English and answer ENTIRELY in English.
            Do NOT include any Arabic text in your response. Every word of your answer must be in English.

            If the question requires multiple items (bullets, conditions, categories, salaries, etc.), you MUST:
            - Retrieve all listed items completely.
            - Do not omit any item mentioned in the text.
            - Do not repeat any item.
            - Preserve bullet structure if present in the text.

            """

            user_prompt = f"""The following legal texts are in Arabic. Read them carefully and answer the question in English only.

            Arabic legal texts:
            {context}

            Question (in English — your answer MUST also be in English):
            {query}

            Important instructions:
            - Carefully review all provided Arabic texts.
            - Extract every relevant item fully and translate it to English.
            - Do not omit any listed element.
            - Do not duplicate any element.
            - Base your answer strictly on explicit text.
            - If there is minor wording variation or typo in the question (e.g. spelling differences), interpret it according to the closest matching term in the provided text.
            - YOUR ENTIRE ANSWER MUST BE IN ENGLISH. Do not use any Arabic in your response.

            Provide your final answer now in English."""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content

        # Save this turn to conversation history
        self.conversation_history.append({"role": "user", "content": standalone_query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Trim history to max window
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        return answer
    
    def get_statistics(self):
        """Get statistics about the chunks"""
        total_chunks = len(self.chunks)
        articles = sum(1 for c in self.chunks if c.metadata['type'] == 'article')
        tables = sum(1 for c in self.chunks if c.metadata['type'] == 'table')
        sub_chunks = sum(1 for c in self.chunks if c.metadata.get('is_subchunk'))
        
        sizes = [c.metadata['chunk_size'] for c in self.chunks]
        
        # Get unique article numbers
        article_numbers = sorted(set(
            int(c.metadata['number']) 
            for c in self.chunks 
            if c.metadata['type'] == 'article' and c.metadata['number']
        ))
        
        return {
            'total_chunks': total_chunks,
            'articles': articles,
            'tables': tables,
            'sub_chunks': sub_chunks,
            'unique_article_numbers': len(article_numbers),
            'article_range': f"{min(article_numbers)} to {max(article_numbers)}" if article_numbers else "N/A",
            'avg_size': sum(sizes) / len(sizes) if sizes else 0,
            'max_size': max(sizes) if sizes else 0,
            'min_size': min(sizes) if sizes else 0
        }


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    import os
    
    # Initialize RAG engine
    engine = RAGEngine(
        document_path="arabic_text_and_tables.txt",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
          # Adjust as needed
    )
    
    # Print statistics
    stats = engine.get_statistics()
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test queries
    print("\n" + "="*80)
    print("EXAMPLE QUERIES")
    print("="*80)
    
    # Arabic query
    print("\n1. Arabic query:")
    answer = engine.answer_question("ما هي الإجازة الدورية للموظف؟", debug=True)
    print(f"\nAnswer: {answer}")
    
    # English query
    print("\n\n2. English query:")
    answer = engine.answer_question("What is the annual leave entitlement?", debug=True)
    print(f"\nAnswer: {answer}")