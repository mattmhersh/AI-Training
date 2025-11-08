# AI Infrastructure & Engineering Curriculum
## Part 4: Advanced AI Applications & Production Systems (Weeks 7-12)

**Master advanced AI techniques and production systems**

---

## Table of Contents - Part 4

- [Week 7: Vector Databases & RAG](#week-7-vector-databases--rag)
- [Week 8: Production ML Ops](#week-8-production-ml-ops)
- [Week 9: Multi-Agent Systems](#week-9-multi-agent-systems)
- [Week 10: Document Intelligence](#week-10-document-intelligence)
- [Week 11: Model Fine-Tuning](#week-11-model-fine-tuning)
- [Week 12: Advanced Optimization](#week-12-advanced-optimization)

---

## Week 7: Vector Databases & RAG

**Real-World Goal:** Build a document Q&A system that answers questions from your own documents

### Why RAG (Retrieval-Augmented Generation)?

**The Problem:** LLMs don't know your specific data
- Company knowledge bases
- Product documentation
- Legal documents
- Research papers
- Customer data

**The Solution:** RAG combines:
1. **Retrieval:** Find relevant documents from your database
2. **Augmentation:** Add those documents to the prompt
3. **Generation:** LLM answers using your data

### Setup

```bash
pip install sentence-transformers chromadb pypdf2 python-docx beautifulsoup4
```

### Day 1-3: Document Q&A System

**File:** `document_qa.py`

```python
"""
Document Q&A System - RAG Implementation
Real use case: Answer questions from company documentation
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from typing import List, Dict, Optional
from datetime import datetime
import PyPDF2
import docx
import re

class DocumentQA:
    def __init__(self, persist_directory: str = "vector_db"):
        """Initialize the Q&A system"""
        
        print("🔄 Loading embedding model...")
        # Use a good all-purpose embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Company documentation"}
        )
        
        print(f"✓ Vector database ready ({self.collection.count()} documents)")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from Word document"""
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from plain text file"""
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata
        
        Why chunking:
        - LLMs have token limits
        - Smaller chunks = more precise retrieval
        - Overlap = don't lose context at boundaries
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "char_count": len(chunk_text),
                    "chunk_index": len(chunks)
                })
        
        return chunks
    
    def add_document(
        self, 
        file_path: str, 
        metadata: Optional[Dict] = None,
        chunk_size: int = 500
    ) -> Dict:
        """
        Add a document to the knowledge base
        
        Real use case: Add company policy documents, product manuals, etc.
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
        
        print(f"\n📄 Processing: {path.name}")
        
        # Extract text based on file type
        if path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif path.suffix.lower() == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif path.suffix.lower() == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            return {"error": f"Unsupported file type: {path.suffix}"}
        
        if not text.strip():
            return {"error": "No text extracted from file"}
        
        print(f"  Extracted {len(text)} characters")
        
        # Chunk the text
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"  Generating embeddings...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True).tolist()
        
        # Prepare metadata
        base_metadata = {
            "filename": path.name,
            "file_path": str(path),
            "file_type": path.suffix,
            "added_at": datetime.now().isoformat(),
            "total_chunks": len(chunks)
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        # Generate IDs for chunks
        doc_id = path.stem
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadatas for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = base_metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "chunk_id": ids[i],
                "word_count": chunk["word_count"],
                "char_count": chunk["char_count"]
            })
            metadatas.append(chunk_meta)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Added to database")
        
        return {
            "status": "success",
            "filename": path.name,
            "chunks_added": len(chunks),
            "total_documents": self.collection.count()
        }
    
    def query(
        self,
        question: str,
        n_results: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the knowledge base
        
        Returns relevant document chunks with similarity scores
        """
        # Generate embedding for question
        question_embedding = self.embedding_model.encode([question])[0].tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "relevance_score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def answer_question(
        self,
        question: str,
        llm_client,
        n_results: int = 3,
        max_tokens: int = 512
    ) -> Dict:
        """
        Answer a question using RAG
        
        Real use case: Customer asks about company policy, get answer from docs
        """
        import time
        start_time = time.time()
        
        # Retrieve relevant documents
        print(f"\n🔍 Searching for: \"{question}\"")
        retrieved_docs = self.query(question, n_results=n_results)
        
        if not retrieved_docs:
            return {
                "success": False,
                "error": "No relevant documents found"
            }
        
        print(f"  Found {len(retrieved_docs)} relevant chunks")
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}] {doc['document']}")
            sources.append({
                "source_number": i,
                "filename": doc['metadata']['filename'],
                "relevance_score": round(doc['relevance_score'], 3),
                "chunk_index": doc['metadata']['chunk_index']
            })
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Answer the question based on the provided context. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get answer from LLM
        print(f"  Generating answer...")
        result = llm_client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            system_prompt="You are a helpful assistant that answers questions based on provided documents. Be accurate and cite sources."
        )
        
        elapsed_time = time.time() - start_time
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error")
            }
        
        return {
            "success": True,
            "question": question,
            "answer": result["text"],
            "sources": sources,
            "elapsed_time": round(elapsed_time, 2),
            "cost": result.get("cost", 0),
            "cached": result.get("cached", False)
        }
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        all_metadata = self.collection.get()['metadatas']
        
        if not all_metadata:
            return {"error": "Database is empty"}
        
        # Count unique documents
        unique_docs = set(meta['filename'] for meta in all_metadata)
        
        # Calculate total sizes
        total_words = sum(meta.get('word_count', 0) for meta in all_metadata)
        total_chars = sum(meta.get('char_count', 0) for meta in all_metadata)
        
        return {
            "total_chunks": len(all_metadata),
            "unique_documents": len(unique_docs),
            "documents": list(unique_docs),
            "total_words": total_words,
            "total_characters": total_chars,
            "avg_chunk_size": round(total_words / len(all_metadata), 0) if all_metadata else 0
        }

# Example: Company Knowledge Base
if __name__ == "__main__":
    from cached_llm_client import CachedLLMClient
    import os
    
    # Initialize systems
    qa_system = DocumentQA()
    
    llm_client = CachedLLMClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    print("="*70)
    print("📚 DOCUMENT Q&A SYSTEM DEMO")
    print("="*70)
    
    # Example: Add company documents
    print("\n1️⃣ Adding Documents to Knowledge Base")
    print("-"*70)
    
    # Create sample documents
    sample_docs = [
        {
            "filename": "company_policy.txt",
            "content": """
Company Remote Work Policy

Effective Date: January 1, 2024

1. Eligibility
All full-time employees are eligible for remote work after 90 days of employment.

2. Work Hours
Remote employees must be available during core hours: 10 AM - 3 PM in their timezone.

3. Equipment
The company provides a laptop and monitor. Additional equipment requests require manager approval.

4. Communication
Daily check-ins via Slack are required. Video cameras must be on for team meetings.

5. Home Office Requirements
Employees must have a dedicated workspace with reliable internet (minimum 25 Mbps).

6. Time Off
Remote work days do not replace PTO. Standard PTO policies apply.

7. Performance
Remote employees are evaluated on output and results, not hours worked.
            """
        },
        {
            "filename": "benefits_guide.txt",
            "content": """
Employee Benefits Guide 2024

Health Insurance
- Coverage starts first day of the month after hire date
- Company pays 80% of premiums for employees
- Coverage options: HMO, PPO, and HDHP plans
- Annual deductible: $1,500 individual / $3,000 family

Paid Time Off
- Vacation: 15 days per year (increases to 20 after 3 years)
- Sick leave: 10 days per year
- Holidays: 12 paid holidays including floating holidays
- Parental leave: 12 weeks paid for primary caregiver, 6 weeks for secondary

401(k) Retirement Plan
- Company matches 50% of contributions up to 6% of salary
- Vesting: Immediate 100% vesting
- Enrollment: Available after 30 days

Professional Development
- Annual stipend: $2,000 for courses, conferences, and certifications
- Conference attendance: 2 conferences per year with pre-approval
            """
        }
    ]
    
    # Write and add sample documents
    for doc in sample_docs:
        # Write file
        Path(doc["filename"]).write_text(doc["content"])
        
        # Add to database
        result = qa_system.add_document(
            doc["filename"],
            metadata={"department": "HR", "category": "policy"}
        )
        print(f"✓ {result['filename']}: {result['chunks_added']} chunks")
    
    # Show database stats
    print("\n2️⃣ Database Statistics")
    print("-"*70)
    stats = qa_system.get_database_stats()
    print(f"Total Documents: {stats['unique_documents']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Words: {stats['total_words']:,}")
    print(f"Avg Chunk Size: {stats['avg_chunk_size']} words")
    
    # Ask questions
    print("\n3️⃣ Question & Answer Examples")
    print("-"*70)
    
    questions = [
        "What is the company's remote work policy?",
        "How many vacation days do employees get?",
        "What equipment does the company provide for remote workers?",
        "How does the 401k matching work?",
        "What are the requirements for a home office?"
    ]
    
    total_cost = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        result = qa_system.answer_question(
            question=question,
            llm_client=llm_client,
            n_results=2
        )
        
        if result["success"]:
            cached_icon = "💾" if result["cached"] else "🌐"
            print(f"\n{cached_icon} Answer: {result['answer']}")
            
            print(f"\n📚 Sources:")
            for source in result['sources']:
                print(f"   [{source['source_number']}] {source['filename']} "
                      f"(relevance: {source['relevance_score']:.1%})")
            
            cost_str = "FREE" if result["cached"] else f"${result['cost']:.4f}"
            print(f"\n⏱️  Time: {result['elapsed_time']}s | 💰 Cost: {cost_str}")
            
            if not result["cached"]:
                total_cost += result["cost"]
        else:
            print(f"❌ Error: {result['error']}")
    
    print("\n" + "="*70)
    print(f"💰 Total Cost: ${total_cost:.4f}")
    print("="*70)
    
    # Cleanup sample files
    for doc in sample_docs:
        Path(doc["filename"]).unlink()
```

**Run it:**
```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
python3 document_qa.py
```

### Day 4-5: Advanced RAG Techniques

**File:** `advanced_rag.py`

```python
"""
Advanced RAG Techniques
- Hybrid search (keyword + semantic)
- Re-ranking for better results
- Query expansion
"""

from document_qa import DocumentQA
from typing import List, Dict
import re

class AdvancedRAG(DocumentQA):
    def __init__(self, persist_directory: str = "vector_db"):
        super().__init__(persist_directory)
    
    def expand_query(self, question: str) -> List[str]:
        """
        Expand query with related questions
        
        Why: Sometimes rephrasing finds better results
        """
        # Simple expansion (in production, use LLM to generate variations)
        expansions = [question]
        
        # Add keyword variations
        if "how" in question.lower():
            expansions.append(question.replace("how", "what is the process"))
        
        if "?" in question:
            # Add declarative version
            expansions.append(question.replace("?", "").strip())
        
        return expansions
    
    def hybrid_search(
        self,
        question: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Combine semantic and keyword search
        
        Real use case: Find documents with specific terms AND semantic meaning
        """
        # Semantic search
        semantic_results = self.query(question, n_results=n_results)
        
        # Keyword search (simple - in production use BM25)
        keywords = set(word.lower() for word in re.findall(r'\w+', question) if len(word) > 3)
        
        # Score documents by keyword presence
        for result in semantic_results:
            doc_text = result['document'].lower()
            keyword_score = sum(1 for keyword in keywords if keyword in doc_text)
            result['keyword_matches'] = keyword_score
            
            # Combine scores (weighted)
            result['combined_score'] = (
                result['relevance_score'] * 0.7 +  # Semantic: 70%
                (keyword_score / len(keywords) if keywords else 0) * 0.3  # Keywords: 30%
            )
        
        # Re-sort by combined score
        semantic_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return semantic_results
    
    def answer_with_citations(
        self,
        question: str,
        llm_client,
        n_results: int = 3
    ) -> Dict:
        """
        Answer with inline citations
        
        Real use case: Legal/compliance documents where citations are required
        """
        # Use hybrid search
        retrieved_docs = self.hybrid_search(question, n_results=n_results)
        
        if not retrieved_docs:
            return {"success": False, "error": "No relevant documents found"}
        
        # Build context with source markers
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source_marker = f"[{i}]"
            context_parts.append(f"{source_marker} {doc['document']}")
            sources.append({
                "id": i,
                "filename": doc['metadata']['filename'],
                "relevance": round(doc['combined_score'], 3),
                "chunk": doc['metadata']['chunk_index']
            })
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for citations
        prompt = f"""Answer the question using the provided sources. Include inline citations using [1], [2], etc. to reference specific sources.

If information comes from multiple sources, cite all relevant ones. If the answer isn't in the sources, say so.

Sources:
{context}

Question: {question}

Answer (with citations):"""
        
        result = llm_client.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.5,  # Lower temperature for factual accuracy
            system_prompt="You are a precise assistant that always cites sources. Include [N] citations in your answers."
        )
        
        if not result["success"]:
            return {"success": False, "error": result.get("error")}
        
        return {
            "success": True,
            "question": question,
            "answer": result["text"],
            "sources": sources,
            "cost": result.get("cost", 0)
        }

# Example usage
if __name__ == "__main__":
    from cached_llm_client import CachedLLMClient
    import os
    
    advanced_rag = AdvancedRAG()
    
    llm_client = CachedLLMClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    # Test hybrid search
    question = "What health insurance options are available?"
    
    print("🔍 Testing Hybrid Search")
    print("="*70)
    print(f"Question: {question}\n")
    
    results = advanced_rag.hybrid_search(question, n_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Relevance: {result['combined_score']:.3f}")
        print(f"   Semantic: {result['relevance_score']:.3f} | Keywords: {result['keyword_matches']}")
        print(f"   Source: {result['metadata']['filename']}")
        print(f"   Text: {result['document'][:100]}...\n")
    
    # Test answer with citations
    print("\n📝 Answer with Citations")
    print("="*70)
    
    result = advanced_rag.answer_with_citations(
        question=question,
        llm_client=llm_client
    )
    
    if result["success"]:
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for source in result['sources']:
            print(f"  [{source['id']}] {source['filename']} (relevance: {source['relevance']})")
```

### Day 6-7: Production RAG API

**File:** `rag_api.py`

```python
"""
Production RAG API
Real use case: Knowledge base API for customer support
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from advanced_rag import AdvancedRAG
from cached_llm_client import CachedLLMClient
import os

app = FastAPI(title="RAG API", version="1.0.0")

# Initialize systems
rag_system = AdvancedRAG()
llm_client = CachedLLMClient(
    endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
    api_key=os.getenv("RUNPOD_API_KEY")
)

class QueryRequest(BaseModel):
    question: str
    n_results: int = 3
    include_citations: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    cost: float
    cached: bool

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload a document to the knowledge base"""
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Add to RAG system
    meta = {}
    if metadata:
        import json
        meta = json.loads(metadata)
    
    result = rag_system.add_document(file_path, metadata=meta)
    
    return result

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base"""
    if request.include_citations:
        result = rag_system.answer_with_citations(
            question=request.question,
            llm_client=llm_client,
            n_results=request.n_results
        )
    else:
        result = rag_system.answer_question(
            question=request.question,
            llm_client=llm_client,
            n_results=request.n_results
        )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        cost=result["cost"],
        cached=result.get("cached", False)
    )

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics"""
    return rag_system.get_database_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Week 7 Milestone:** ✅ You've built a production RAG system that can:
- Ingest documents (PDF, Word, Text)
- Create vector embeddings
- Perform semantic search
- Answer questions with citations
- Serve via API

This enables you to build AI systems that work with your own data - the most valuable type of AI application.

---

## Week 8: Production ML Ops

**Real-World Goal:** Add monitoring, logging, and error handling to production AI systems

### Day 1-3: Comprehensive Logging

**File:** `ai_logger.py`

```python
"""
Production AI Logging System
Real use case: Track all AI operations for debugging and analysis
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import traceback

class AILogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        self.logger = logging.getLogger("ai_system")
        self.logger.setLevel(logging.INFO)
        
        # File handler with JSON formatting
        log_file = self.log_dir / f"ai_system_{datetime.now():%Y%m%d}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        self.logger.addHandler(console)
    
    def log_llm_request(
        self,
        prompt: str,
        response: str,
        cost: float,
        latency: float,
        tokens: Dict,
        cached: bool = False,
        error: Optional[str] = None
    ):
        """Log LLM API request"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "llm_request",
            "prompt_length": len(prompt),
            "response_length": len(response) if response else 0,
            "cost": cost,
            "latency_ms": latency * 1000,
            "tokens": tokens,
            "cached": cached,
            "status": "error" if error else "success",
            "error": error
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_rag_query(
        self,
        question: str,
        answer: str,
        sources_found: int,
        relevance_scores: List[float],
        cost: float,
        latency: float
    ):
        """Log RAG query"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "rag_query",
            "question_length": len(question),
            "answer_length": len(answer),
            "sources_found": sources_found,
            "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "cost": cost,
            "latency_ms": latency * 1000
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict = None
    ):
        """Log error with context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.logger.error(json.dumps(log_entry))
    
    def analyze_logs(self, date: Optional[str] = None) -> Dict:
        """Analyze logs for insights"""
        if not date:
            date = datetime.now().strftime("%Y%m%d")
        
        log_file = self.log_dir / f"ai_system_{date}.log"
        
        if not log_file.exists():
            return {"error": "Log file not found"}
        
        stats = {
            "llm_requests": 0,
            "rag_queries": 0,
            "errors": 0,
            "total_cost": 0,
            "avg_latency": 0,
            "cache_hit_rate": 0
        }
        
        latencies = []
        cached_count = 0
        total_requests = 0
        
        with open(log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    if entry["type"] == "llm_request":
                        stats["llm_requests"] += 1
                        if entry.get("cost"):
                            stats["total_cost"] += entry["cost"]
                        if entry.get("latency_ms"):
                            latencies.append(entry["latency_ms"])
                        if entry.get("cached"):
                            cached_count += 1
                        total_requests += 1
                    
                    elif entry["type"] == "rag_query":
                        stats["rag_queries"] += 1
                        if entry.get("cost"):
                            stats["total_cost"] += entry["cost"]
                        if entry.get("latency_ms"):
                            latencies.append(entry["latency_ms"])
                    
                    elif entry["type"] == "error":
                        stats["errors"] += 1
                
                except json.JSONDecodeError:
                    continue
        
        if latencies:
            stats["avg_latency"] = sum(latencies) / len(latencies)
        
        if total_requests > 0:
            stats["cache_hit_rate"] = (cached_count / total_requests) * 100
        
        return stats

# Example usage
if __name__ == "__main__":
    logger = AILogger()
    
    # Simulate some operations
    logger.log_llm_request(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        cost=0.001,
        latency=1.5,
        tokens={"input": 10, "output": 8},
        cached=False
    )
    
    logger.log_rag_query(
        question="What is our return policy?",
        answer="Our return policy allows 30 days...",
        sources_found=3,
        relevance_scores=[0.95, 0.87, 0.76],
        cost=0.002,
        latency=2.1
    )
    
    # Analyze logs
    stats = logger.analyze_logs()
    print("\n📊 Log Analysis:")
    print(f"LLM Requests: {stats['llm_requests']}")
    print(f"RAG Queries: {stats['rag_queries']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total Cost: ${stats['total_cost']:.4f}")
    print(f"Avg Latency: {stats['avg_latency']:.0f}ms")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
```

### Day 4-5: Error Handling & Retry Logic

**File:** `ai_resilience.py`

```python
"""
AI System Resilience - Retry logic and error handling
Real use case: Handle API failures gracefully in production
"""

import time
from typing import Callable, Any, Optional
import random

class AIResilience:
    @staticmethod
    def retry_with_exponential_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ) -> Any:
        """
        Retry a function with exponential backoff
        
        Why: APIs fail temporarily - retrying with delays usually succeeds
        
        Real use case: RunPod cold starts, network hiccups, rate limits
        """
        for attempt in range(max_retries + 1):
            try:
                return func()
            
            except Exception as e:
                if attempt == max_retries:
                    raise Exception(f"Failed after {max_retries + 1} attempts: {str(e)}")
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                
                # Add jitter to prevent thundering herd
                if jitter:
                    delay = delay * (0.5 + random.random())
                
                print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
                print(f"   Retrying in {delay:.1f}s...")
                
                time.sleep(delay)
    
    @staticmethod
    def fallback_chain(
        primary_func: Callable,
        fallback_func: Callable,
        error_handler: Optional[Callable] = None
    ) -> Any:
        """
        Try primary function, fall back to secondary if it fails
        
        Real use case: Use GPT-4 normally, fall back to GPT-3.5 if it fails
        """
        try:
            return primary_func()
        except Exception as e:
            if error_handler:
                error_handler(e)
            
            print(f"⚠️  Primary function failed: {str(e)}")
            print(f"   Using fallback...")
            
            try:
                return fallback_func()
            except Exception as fallback_error:
                raise Exception(f"Both primary and fallback failed. Primary: {str(e)}, Fallback: {str(fallback_error)}")
    
    @staticmethod
    def circuit_breaker(
        func: Callable,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0
    ):
        """
        Circuit breaker pattern to prevent cascading failures
        
        States:
        - CLOSED: Normal operation
        - OPEN: Too many failures, reject requests
        - HALF_OPEN: Testing if service recovered
        """
        # This is a simplified example - use a proper circuit breaker library in production
        class CircuitBreaker:
            def __init__(self):
                self.failure_count = 0
                self.success_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"
            
            def call(self, *args, **kwargs):
                # Check if circuit should reset
                if (self.state == "OPEN" and 
                    self.last_failure_time and 
                    time.time() - self.last_failure_time > timeout):
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                
                if self.state == "OPEN":
                    raise Exception("Circuit breaker is OPEN - too many failures")
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - reset or close circuit
                    self.failure_count = 0
                    if self.state == "HALF_OPEN":
                        self.success_count += 1
                        if self.success_count >= success_threshold:
                            self.state = "CLOSED"
                            print("✅ Circuit breaker CLOSED - service recovered")
                    
                    return result
                
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= failure_threshold:
                        self.state = "OPEN"
                        print(f"🚨 Circuit breaker OPEN - {failure_threshold} failures")
                    
                    raise e
        
        return CircuitBreaker()

# Example usage
if __name__ == "__main__":
    from cached_llm_client import CachedLLMClient
    import os
    
    client = CachedLLMClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    # Test retry logic
    print("Testing Retry Logic")
    print("="*70)
    
    def unstable_generation():
        """Simulates an unstable API"""
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("API temporarily unavailable")
        return client.generate("What is 2+2?", max_tokens=50)
    
    result = AIResilience.retry_with_exponential_backoff(
        unstable_generation,
        max_retries=3
    )
    
    print(f"✅ Success: {result['text']}")
    
    # Test fallback
    print("\n\nTesting Fallback Chain")
    print("="*70)
    
    def primary():
        # Simulate expensive model
        raise Exception("GPT-4 rate limit exceeded")
    
    def fallback():
        # Use cheaper model
        print("   Using GPT-3.5 instead...")
        return {"text": "2+2=4 (from fallback)", "cost": 0.0001}
    
    result = AIResilience.fallback_chain(primary, fallback)
    print(f"✅ Result: {result['text']}")
```

### Day 6-7: Performance Monitoring

**File:** `ai_monitor.py`

```python
"""
AI Performance Monitoring
Real use case: Track and optimize AI system performance
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict
import json
from pathlib import Path
import statistics

@dataclass
class PerformanceMetric:
    timestamp: str
    operation: str
    latency_ms: float
    cost: float
    tokens: Dict
    success: bool
    cached: bool = False

class AIMonitor:
    def __init__(self, metrics_file: str = "metrics.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics: List[PerformanceMetric] = []
    
    def record_metric(
        self,
        operation: str,
        latency_ms: float,
        cost: float,
        tokens: Dict,
        success: bool,
        cached: bool = False
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            latency_ms=latency_ms,
            cost=cost,
            tokens=tokens,
            success=success,
            cached=cached
        )
        
        self.metrics.append(metric)
        
        # Append to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({
                "timestamp": metric.timestamp,
                "operation": metric.operation,
                "latency_ms": metric.latency_ms,
                "cost": metric.cost,
                "tokens": metric.tokens,
                "success": metric.success,
                "cached": metric.cached
            }) + "\n")
    
    def get_metrics(self, hours: int = 24) -> List[PerformanceMetric]:
        """Load metrics from last N hours"""
        if not self.metrics_file.exists():
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        metrics = []
        
        with open(self.metrics_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    metric_time = datetime.fromisoformat(data["timestamp"])
                    
                    if metric_time >= cutoff:
                        metrics.append(PerformanceMetric(**data))
                except:
                    continue
        
        return metrics
    
    def generate_report(self, hours: int = 24) -> Dict:
        """Generate performance report"""
        metrics = self.get_metrics(hours)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        latencies = [m.latency_ms for m in metrics if m.success]
        costs = [m.cost for m in metrics if m.success]
        
        successful = len([m for m in metrics if m.success])
        failed = len([m for m in metrics if not m.success])
        cached = len([m for m in metrics if m.cached])
        
        # Group by operation
        by_operation = {}
        for metric in metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = {
                    "count": 0,
                    "success": 0,
                    "total_cost": 0,
                    "latencies": []
                }
            
            by_operation[metric.operation]["count"] += 1
            if metric.success:
                by_operation[metric.operation]["success"] += 1
                by_operation[metric.operation]["total_cost"] += metric.cost
                by_operation[metric.operation]["latencies"].append(metric.latency_ms)
        
        # Calculate operation stats
        operation_stats = {}
        for op, data in by_operation.items():
            operation_stats[op] = {
                "count": data["count"],
                "success_rate": (data["success"] / data["count"] * 100) if data["count"] > 0 else 0,
                "total_cost": round(data["total_cost"], 4),
                "avg_latency": round(statistics.mean(data["latencies"]), 2) if data["latencies"] else 0,
                "p95_latency": round(statistics.quantiles(data["latencies"], n=20)[18], 2) if len(data["latencies"]) > 20 else 0
            }
        
        return {
            "period_hours": hours,
            "summary": {
                "total_requests": len(metrics),
                "successful": successful,
                "failed": failed,
                "success_rate": round(successful / len(metrics) * 100, 2),
                "cached_responses": cached,
                "cache_hit_rate": round(cached / len(metrics) * 100, 2)
            },
            "performance": {
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
                "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
                "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 20 else 0,
                "p99_latency_ms": round(statistics.quantiles(latencies, n=100)[98], 2) if len(latencies) > 100 else 0
            },
            "costs": {
                "total": round(sum(costs), 4),
                "average": round(statistics.mean(costs), 6) if costs else 0,
                "projected_daily": round(sum(costs) / hours * 24, 4),
                "projected_monthly": round(sum(costs) / hours * 24 * 30, 2)
            },
            "by_operation": operation_stats
        }
    
    def check_health(self, thresholds: Dict = None) -> Dict:
        """Check system health against thresholds"""
        if thresholds is None:
            thresholds = {
                "max_error_rate": 5.0,  # 5% error rate
                "max_avg_latency": 3000,  # 3 seconds
                "max_p95_latency": 5000,  # 5 seconds
                "min_cache_hit_rate": 30.0  # 30% cache hits
            }
        
        report = self.generate_report(hours=1)  # Last hour
        
        if "error" in report:
            return {"status": "unknown", "reason": "No metrics available"}
        
        issues = []
        
        # Check error rate
        error_rate = 100 - report["summary"]["success_rate"]
        if error_rate > thresholds["max_error_rate"]:
            issues.append(f"High error rate: {error_rate:.1f}%")
        
        # Check latency
        avg_latency = report["performance"]["avg_latency_ms"]
        if avg_latency > thresholds["max_avg_latency"]:
            issues.append(f"High average latency: {avg_latency:.0f}ms")
        
        p95_latency = report["performance"]["p95_latency_ms"]
        if p95_latency > thresholds["max_p95_latency"]:
            issues.append(f"High P95 latency: {p95_latency:.0f}ms")
        
        # Check cache hit rate
        cache_rate = report["summary"]["cache_hit_rate"]
        if cache_rate < thresholds["min_cache_hit_rate"]:
            issues.append(f"Low cache hit rate: {cache_rate:.1f}%")
        
        if issues:
            return {
                "status": "unhealthy",
                "issues": issues,
                "metrics": report["summary"]
            }
        
        return {
            "status": "healthy",
            "metrics": report["summary"]
        }

# Example usage
if __name__ == "__main__":
    monitor = AIMonitor()
    
    # Simulate some operations
    import random
    
    print("Simulating operations...")
    for i in range(50):
        monitor.record_metric(
            operation="llm_generation",
            latency_ms=random.gauss(2000, 500),
            cost=random.uniform(0.001, 0.01),
            tokens={"input": random.randint(50, 200), "output": random.randint(100, 500)},
            success=random.random() > 0.05,  # 95% success rate
            cached=random.random() > 0.6  # 40% cache hit
        )
    
    # Generate report
    print("\n📊 PERFORMANCE REPORT (Last 24 hours)")
    print("="*70)
    
    report = monitor.generate_report()
    
    print(f"\n📈 Summary:")
    print(f"   Total Requests: {report['summary']['total_requests']}")
    print(f"   Success Rate: {report['summary']['success_rate']}%")
    print(f"   Cache Hit Rate: {report['summary']['cache_hit_rate']}%")
    
    print(f"\n⚡ Performance:")
    print(f"   Avg Latency: {report['performance']['avg_latency_ms']:.0f}ms")
    print(f"   P50 Latency: {report['performance']['p50_latency_ms']:.0f}ms")
    print(f"   P95 Latency: {report['performance']['p95_latency_ms']:.0f}ms")
    
    print(f"\n💰 Costs:")
    print(f"   Total: ${report['costs']['total']}")
    print(f"   Average: ${report['costs']['average']}")
    print(f"   Projected Daily: ${report['costs']['projected_daily']}")
    print(f"   Projected Monthly: ${report['costs']['projected_monthly']}")
    
    # Health check
    print("\n🏥 HEALTH CHECK")
    print("="*70)
    
    health = monitor.check_health()
    
    if health["status"] == "healthy":
        print("✅ System is healthy")
    else:
        print(f"⚠️  System is {health['status']}")
        print("\nIssues:")
        for issue in health["issues"]:
            print(f"   - {issue}")
```

**Week 8 Milestone:** ✅ You've added production-grade operational capabilities:
- Comprehensive logging system
- Error handling and retry logic
- Performance monitoring
- Health checks
- Cost tracking and projections

Your AI systems are now observable, resilient, and production-ready.

---

## Week 9-12: Advanced Topics Overview

### Week 9: Multi-Agent Systems
Build systems where multiple AI agents collaborate:
- **Orchestrator agent** coordinates other agents
- **Specialist agents** handle specific tasks
- **Tool-using agents** can call APIs, search databases
- **Real use case:** AI assistant that can research, write, and fact-check

### Week 10: Document Intelligence
Advanced document processing:
- **Table extraction** from PDFs
- **Image analysis** within documents
- **Multi-document synthesis**
- **Real use case:** Contract analysis system

### Week 11: Model Fine-Tuning
Customize models for your use case:
- **Prepare training data** from your domain
- **Fine-tune on RunPod** with your data
- **Evaluate model performance**
- **Real use case:** Customer service model trained on your data

### Week 12: Advanced Optimization
Squeeze maximum performance:
- **Batch processing** for throughput
- **Response streaming** for better UX
- **Model quantization** for faster inference
- **Load balancing** across multiple endpoints
- **Real use case:** High-traffic production system

---

## Summary: Part 4 Complete! 🎉

### What You've Built:
- **Week 7:** RAG system for document Q&A
- **Week 8:** Production monitoring and resilience
- **Weeks 9-12:** Advanced patterns (covered in learning resources)

### Skills Acquired:
✅ Vector databases and embeddings  
✅ Retrieval-Augmented Generation (RAG)  
✅ Hybrid search techniques  
✅ Production logging and monitoring  
✅ Error handling and retry logic  
✅ Performance optimization  
✅ System health checks  

### Real Value Created:
- Document Q&A system for any knowledge base
- Production-grade monitoring infrastructure
- Cost optimization strategies
- Resilient AI services

---

## Next Steps

Continue to **Part 5: Capstone Projects (Weeks 17-20)** where you'll build:
- Complete AI Customer Support Platform
- AI Content Analysis Pipeline

These integrate everything you've learned into production systems you can showcase! 🚀

---

## Additional Learning Resources

### Books
- "Designing Machine Learning Systems" by Chip Huyen
- "Building LLM Apps" by Shubham Saboo
- "The Hundred-Page Machine Learning Book" by Andriy Burkov

### Online Courses
- Fast.ai Practical Deep Learning
- DeepLearning.AI LLM Specialization
- Hugging Face Course (free)

### Communities
- r/MachineLearning
- r/LocalLLaMA
- Discord: RunPod Community
- Discord: LangChain

### Tools to Explore
- LangChain for complex AI workflows
- LlamaIndex for advanced RAG
- Weights & Biases for experiment tracking
- Modal for serverless Python

### Production Best Practices
1. **Always monitor costs** - Track every API call
2. **Implement caching** - Reduces costs by 50-80%
3. **Use logging** - Debug issues faster
4. **Add retry logic** - Handle transient failures
5. **Health checks** - Know when systems are degraded
6. **Load test** - Find breaking points before users do
7. **Version everything** - Code, models, prompts
8. **Document decisions** - Why you chose X over Y

---

**You've completed the advanced AI infrastructure curriculum!** 

You now have the skills to:
- Build production AI applications
- Deploy and manage LLMs
- Optimize costs effectively
- Monitor and debug AI systems
- Scale to handle real traffic

Ready for the capstone projects? Let's build something amazing! 🚀