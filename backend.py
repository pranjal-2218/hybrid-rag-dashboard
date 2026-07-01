import asyncio
import logging
import math
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid-rag-backend")

# --- Pydantic Schemas (Pydantic v2 compliant) ---

class IngestRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the parent document")
    text: str = Field(..., description="Full text content of the document")

class IngestResponse(BaseModel):
    status: str
    document_id: str
    num_chunks: int
    ingestion_time_seconds: float

class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int = Field(5, ge=1, description="Number of top fused results to return")

class QueryResponseItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float

class QueryResponse(BaseModel):
    results: List[QueryResponseItem]

# --- Sliding Window Chunker ---

def sliding_window_chunk(text: str, window_size: int = 600, overlap: int = 200) -> List[str]:
    """
    Segments text into multilingual sliding windows using character-level boundaries.
    Character-level sliding windows ensure language independence (e.g., CJK and Western characters).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= window_size:
        return [text]
        
    chunks: List[str] = []
    step = window_size - overlap
    i = 0
    while i < len(text):
        chunk = text[i:i + window_size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
        # If remaining text length is smaller than a step size, capture it and break
        if len(text) - i < step:
            final_chunk = text[i:].strip()
            if final_chunk and final_chunk not in chunks:
                chunks.append(final_chunk)
            break
    return chunks

# --- Concurrent In-Memory Document Store ---

class InMemoryDocumentStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        
        # Chunk details
        self.chunks: Dict[str, Dict] = {}  # chunk_id -> {"document_id": str, "text": str, "chunk_index": int}
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> np.ndarray (float32, normalized L2)
        
        # BM25 Statistics
        self.doc_frequencies: Dict[str, int] = {}  # token -> frequency across all chunks
        self.chunk_token_counts: Dict[str, Dict[str, int]] = {}  # chunk_id -> {token -> count}
        self.chunk_lengths: Dict[str, int] = {}  # chunk_id -> token count
        self.total_tokens: int = 0
        self.num_chunks: int = 0
        
    def _tokenize(self, text: str) -> List[str]:
        # Multilingual Unicode word-level tokenization
        return re.findall(r"\w+", text.lower(), re.UNICODE)

    def delete_document_unlocked(self, document_id: str) -> None:
        """Removes previous chunks and metrics of a document ID (Thread-unsafe, must hold lock)."""
        chunk_ids_to_delete = [
            cid for cid, chunk in self.chunks.items()
            if chunk["document_id"] == document_id
        ]
        
        for cid in chunk_ids_to_delete:
            token_counts = self.chunk_token_counts.get(cid, {})
            for token in token_counts.keys():
                if token in self.doc_frequencies:
                    self.doc_frequencies[token] -= 1
                    if self.doc_frequencies[token] <= 0:
                        del self.doc_frequencies[token]
            
            self.total_tokens -= self.chunk_lengths.get(cid, 0)
            self.num_chunks -= 1
            
            self.chunks.pop(cid, None)
            self.embeddings.pop(cid, None)
            self.chunk_token_counts.pop(cid, None)
            self.chunk_lengths.pop(cid, None)

    def add_chunks(self, document_id: str, chunk_texts: List[str], chunk_embeddings: List[np.ndarray]) -> None:
        """Atomically registers chunks and compute dense/sparse statistics."""
        with self.lock:
            # Overwrite previous version of the document if it exists
            self.delete_document_unlocked(document_id)
            
            for idx, (text, emb) in enumerate(zip(chunk_texts, chunk_embeddings)):
                chunk_id = f"{document_id}_chunk_{idx}"
                self.chunks[chunk_id] = {
                    "document_id": document_id,
                    "text": text,
                    "chunk_index": idx
                }
                self.embeddings[chunk_id] = emb
                
                # Tokenize & update BM25 stats
                tokens = self._tokenize(text)
                token_counts: Dict[str, int] = {}
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
                
                self.chunk_token_counts[chunk_id] = token_counts
                self.chunk_lengths[chunk_id] = len(tokens)
                self.total_tokens += len(tokens)
                self.num_chunks += 1
                
                for token in token_counts.keys():
                    self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Performs BM25 search over token frequencies."""
        with self.lock:
            if self.num_chunks == 0:
                return []
                
            avgdl = self.total_tokens / self.num_chunks
            k1 = 1.5
            b = 0.75
            
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
                
            scores: Dict[str, float] = {}
            N = self.num_chunks
            
            for cid, token_counts in self.chunk_token_counts.items():
                score = 0.0
                dl = self.chunk_lengths[cid]
                
                for token in query_tokens:
                    if token in token_counts:
                        tf = token_counts[token]
                        df = self.doc_frequencies.get(token, 0)
                        
                        # BM25 IDF formulation
                        idf = max(0.0001, math.log(1.0 + (N - df + 0.5) / (df + 0.5)))
                        
                        # BM25 TF contribution
                        term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)))
                        score += term_score
                        
                if score > 0:
                    scores[cid] = score
            
            # Sort descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_scores[:top_k]

    def dense_search(self, query_emb: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Computes Cosine Similarity using vectorized NumPy operations."""
        with self.lock:
            if not self.embeddings:
                return []
                
            chunk_ids = list(self.embeddings.keys())
            emb_matrix = np.array([self.embeddings[cid] for cid in chunk_ids], dtype=np.float32)
            
            # Embeddings are pre-normalized, hence dot-product is equivalent to Cosine Similarity
            similarities = np.dot(emb_matrix, query_emb)
            
            results = [(cid, float(sim)) for cid, sim in zip(chunk_ids, similarities)]
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def hybrid_search(self, query: str, query_emb: np.ndarray, top_k: int) -> List[Dict]:
        """Fuses sparse and dense rankings using Reciprocal Rank Fusion (RRF)."""
        pool_size = max(100, top_k * 2)
        
        # Parallel-logical retrieval (lock is held incrementally per search)
        sparse_res = self.sparse_search(query, pool_size)
        dense_res = self.dense_search(query_emb, pool_size)
        
        # Store ranks (1-indexed)
        sparse_ranks = {cid: rank for rank, (cid, _) in enumerate(sparse_res, start=1)}
        dense_ranks = {cid: rank for rank, (cid, _) in enumerate(dense_res, start=1)}
        
        # Merge via RRF
        candidates = set(sparse_ranks.keys()).union(dense_ranks.keys())
        rrf_scores: Dict[str, float] = {}
        
        for cid in candidates:
            score = 0.0
            if cid in sparse_ranks:
                score += 1.0 / (60.0 + sparse_ranks[cid])
            if cid in dense_ranks:
                score += 1.0 / (60.0 + dense_ranks[cid])
            rrf_scores[cid] = score
            
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Map back to structural context
        results: List[Dict] = []
        RRF_THRESHOLD = 0.018  # Filter out low-confidence "garbage" results
        
        with self.lock:
            for cid, rrf_score in sorted_candidates[:top_k]:
                if rrf_score < RRF_THRESHOLD:
                    continue  # Drop low-scoring chunks entirely
                    
                chunk_info = self.chunks.get(cid)
                if chunk_info:
                    results.append({
                        "chunk_id": cid,
                        "document_id": chunk_info["document_id"],
                        "text": chunk_info["text"],
                        "score": rrf_score
                    })
        return results

# --- App Lifespan and Engine Setup ---

model: SentenceTransformer = None
document_store = InMemoryDocumentStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Initializing SentenceTransformer model 'sentence-transformers/all-MiniLM-L6-v2'...")
    # Initialize the model on startup in a separate thread to keep loop free
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(
        None,
        lambda: SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    )
    logger.info("Model load complete.")
    yield
    logger.info("Shutdown sequence initiated.")

app = FastAPI(lifespan=lifespan, title="Production Hybrid RAG Engine")

# --- FastAPI API Endpoints ---

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(payload: IngestRequest) -> IngestResponse:
    start_time = time.perf_counter()
    document_id = payload.document_id.strip()
    text = payload.text.strip()
    
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id cannot be empty.")
    if not text:
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")
        
    try:
        chunks = sliding_window_chunk(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Text split produced zero chunks.")
            
        # CRITICAL PERF: Encode is CPU-bound; run in loop's thread executor
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(chunks, normalize_embeddings=True)
        )
        
        # Convert output vectors
        emb_arrays = [np.array(e, dtype=np.float32) for e in embeddings]
        
        # Register in-memory
        document_store.add_chunks(document_id, chunks, emb_arrays)
        
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Ingestion Metric | Document: %s | Chunks Created: %d | Time: %.4fs",
            document_id, len(chunks), elapsed
        )
        
        return IngestResponse(
            status="success",
            document_id=document_id,
            num_chunks=len(chunks),
            ingestion_time_seconds=elapsed
        )
    except Exception as e:
        logger.error("Ingestion error for %s: %s", document_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion processing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_index(payload: QueryRequest) -> QueryResponse:
    start_time = time.perf_counter()
    query_str = payload.query.strip()
    top_k = payload.top_k
    
    if not query_str:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        # Check storage size
        with document_store.lock:
            store_size = document_store.num_chunks
            
        if store_size == 0:
            return QueryResponse(results=[])
            
        # CRITICAL PERF: Model inference running in executor
        loop = asyncio.get_running_loop()
        query_emb = await loop.run_in_executor(
            None,
            lambda: model.encode(query_str, normalize_embeddings=True)
        )
        
        results = document_store.hybrid_search(query_str, query_emb, top_k)
        
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Query Metric | Query: '%s' | Top K: %d | Found: %d | Time: %.4fs",
            query_str, top_k, len(results), elapsed
        )
        
        response_items = [
            QueryResponseItem(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                score=item["score"]
            )
            for item in results
        ]
        
        return QueryResponse(results=response_items)
    except Exception as e:
        logger.error("Query execution error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query matching failed: {str(e)}")
