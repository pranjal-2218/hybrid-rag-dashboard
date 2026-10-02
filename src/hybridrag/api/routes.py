import asyncio
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from .schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, QueryResponseItem
from ..store.db import SQLiteStore
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseRetriever
from ..retrieval.fusion import rrf_fusion
from ..retrieval.rerank import Reranker
from ..generate import Generator
from ..ingest.chunker import sliding_window_chunk
from ..cache import LRUCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid-rag-api")

model: SentenceTransformer = None
store = SQLiteStore("hybridrag.db")
bm25_retriever = BM25Retriever()
dense_retriever = DenseRetriever()
reranker = Reranker()
generator = Generator()
query_cache = LRUCache(capacity=500)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Initializing SentenceTransformer...")
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(
        None,
        lambda: SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    )
    
    logger.info("Warming up indices from SQLite...")
    
    logger.info("Warming up Reranker...")
    loop.run_in_executor(None, reranker.load)
    
    all_chunks = store.load_all_chunks()
    all_embs = store.load_all_embeddings()
    
    # Load into BM25
    for cid, data in all_chunks.items():
        bm25_retriever.add_chunk(cid, data["text"])
        
    # Load into Dense
    cids = list(all_embs.keys())
    embs = [all_embs[cid] for cid in cids]
    if cids:
        dense_retriever.add_embeddings(cids, embs)
        
    logger.info(f"Loaded {len(all_chunks)} chunks into memory indices.")
    yield
    logger.info("Shutdown sequence initiated.")

app = FastAPI(lifespan=lifespan, title="Production Hybrid RAG Engine")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(payload: IngestRequest) -> IngestResponse:
    start_time = time.perf_counter()
    document_id = payload.document_id.strip()
    text = payload.text.strip()
    
    if not document_id or not text:
        raise HTTPException(status_code=400, detail="Invalid input.")
        
    try:
        chunks = sliding_window_chunk(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Text split produced zero chunks.")
            
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(chunks, normalize_embeddings=True)
        )
        
        emb_arrays = [np.array(e, dtype=np.float32) for e in embeddings]
        
        # Persist to DB
        store.save_chunks(document_id, chunks, emb_arrays)
        
        # Add to memory retrievers
        chunk_ids = []
        for i, (chunk_text, emb) in enumerate(zip(chunks, emb_arrays)):
            cid = f"{document_id}_chunk_{i}"
            bm25_retriever.add_chunk(cid, chunk_text)
            chunk_ids.append(cid)
            
        dense_retriever.add_embeddings(chunk_ids, emb_arrays)
        
        elapsed = time.perf_counter() - start_time
        return IngestResponse(
            status="success",
            document_id=document_id,
            num_chunks=len(chunks),
            ingestion_time_seconds=elapsed
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_index(payload: QueryRequest) -> QueryResponse:
    start_time = time.perf_counter()
    query_str = payload.query.strip()
    top_k = payload.top_k
    
    # Check cache
    cache_key = f"{query_str}_{top_k}"
    cached_res = query_cache.get(cache_key)
    if cached_res:
        return QueryResponse(results=cached_res["results"], answer=cached_res["answer"])
        
    try:
        if bm25_retriever.num_chunks == 0:
            return QueryResponse(results=[])
            
        loop = asyncio.get_running_loop()
        query_emb = await loop.run_in_executor(
            None,
            lambda: model.encode(query_str, normalize_embeddings=True)
        )
        
        pool_size = max(100, top_k * 2)
        sparse_res = bm25_retriever.search(query_str, pool_size)
        dense_res = dense_retriever.search(query_emb, pool_size)
        
        fused = rrf_fusion(sparse_res, dense_res, top_k)
        
        results = []
        with store._get_conn() as conn:
            for cid, score in fused:
                cursor = conn.execute("SELECT document_id, text FROM chunks WHERE id = ?", (cid,))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "chunk_id": cid,
                        "document_id": row[0],
                        "text": row[1],
                        "score": score
                    })
                    
        # Apply Cross-Encoder Reranking
        reranked_results = await reranker.rerank_async(query_str, results, top_k)
        
        # Generate LLM Answer
        answer_text = generator.generate_answer(query_str, reranked_results)
        
        final_items = [
            QueryResponseItem(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=r.get("rerank_score", r["score"])
            ) for r in reranked_results
        ]
                    
        query_cache.put(cache_key, {"results": final_items, "answer": answer_text})
        return QueryResponse(results=final_items, answer=answer_text)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
