import time
import uuid
import numpy as np
from src.hybridrag.retrieval.bm25 import BM25Retriever
from src.hybridrag.retrieval.dense import DenseRetriever

def run_benchmark():
    print("--- Performance Benchmarking ---")
    sizes = [1000, 10000, 50000]
    
    for size in sizes:
        print(f"\nBenchmarking with {size} chunks...")
        
        # Setup BM25
        bm25 = BM25Retriever()
        for i in range(size):
            bm25.add_chunk(f"doc_{i}", "this is a benchmark chunk with some artificial text.")
            
        # Setup Dense
        dense = DenseRetriever(initial_capacity=size + 10)
        chunk_ids = [f"doc_{i}" for i in range(size)]
        embs = [np.random.rand(384).astype(np.float32) for _ in range(size)]
        dense.add_embeddings(chunk_ids, embs)
        
        # Bench BM25
        t0 = time.perf_counter()
        bm25.search("benchmark text", top_k=50)
        t_bm25 = time.perf_counter() - t0
        
        # Bench Dense
        query_emb = np.random.rand(384).astype(np.float32)
        t0 = time.perf_counter()
        dense.search(query_emb, top_k=50)
        t_dense = time.perf_counter() - t0
        
        print(f"BM25 Latency: {t_bm25*1000:.2f} ms")
        print(f"Dense Latency: {t_dense*1000:.2f} ms")

if __name__ == "__main__":
    run_benchmark()
