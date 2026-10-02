import numpy as np
import heapq
from typing import List, Tuple

class DenseRetriever:
    def __init__(self, embedding_dim: int = 384, initial_capacity: int = 1000):
        self.embedding_dim = embedding_dim
        self.capacity = initial_capacity
        self.matrix = np.empty((self.capacity, self.embedding_dim), dtype=np.float32)
        self.chunk_ids: List[str] = []
        self.count = 0

    def add_embeddings(self, chunk_ids: List[str], embeddings: List[np.ndarray]):
        num_new = len(embeddings)
        if num_new == 0:
            return
            
        while self.count + num_new > self.capacity:
            self.capacity *= 2
            new_matrix = np.empty((self.capacity, self.embedding_dim), dtype=np.float32)
            new_matrix[:self.count] = self.matrix[:self.count]
            self.matrix = new_matrix
            
        for i, (cid, emb) in enumerate(zip(chunk_ids, embeddings)):
            self.chunk_ids.append(cid)
            self.matrix[self.count + i] = emb
            
        self.count += num_new

    def search(self, query_emb: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self.count == 0:
            return []
            
        active_matrix = self.matrix[:self.count]
        similarities = np.dot(active_matrix, query_emb)
        
        results = zip(self.chunk_ids, similarities)
        # Assuming query_emb and active_matrix are normalized, sim is cosine sim
        return heapq.nlargest(top_k, results, key=lambda x: x[1])
