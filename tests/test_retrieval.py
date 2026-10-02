import pytest
import numpy as np
from src.hybridrag.retrieval.bm25 import BM25Retriever
from src.hybridrag.retrieval.dense import DenseRetriever
from src.hybridrag.retrieval.fusion import rrf_fusion
from src.hybridrag.ingest.chunker import sliding_window_chunk

def test_sliding_window_chunker():
    text = "A" * 1000
    chunks = sliding_window_chunk(text, window_size=600, overlap=200)
    assert len(chunks) > 1
    assert len(chunks[0]) <= 600

def test_bm25_retrieval():
    bm25 = BM25Retriever()
    bm25.add_chunk("c1", "The quick brown fox jumps over the lazy dog.")
    bm25.add_chunk("c2", "A fast brown fox")
    
    res = bm25.search("brown fox", top_k=2)
    assert len(res) == 2
    assert res[0][0] in ["c1", "c2"]

def test_dense_retrieval():
    dense = DenseRetriever()
    embs = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    dense.add_embeddings(["c1", "c2"], embs)
    
    query = np.array([1.0, 0.0], dtype=np.float32)
    res = dense.search(query, top_k=1)
    assert len(res) == 1
    assert res[0][0] == "c1"

def test_rrf_fusion():
    sparse = [("c1", 10.0), ("c2", 5.0), ("c3", 1.0)]
    dense = [("c3", 0.9), ("c1", 0.8), ("c2", 0.7)]
    
    fused = rrf_fusion(sparse, dense, top_k=2)
    assert len(fused) == 2
    assert fused[0][0] == "c1" # Ranked 1st in sparse, 2nd in dense
