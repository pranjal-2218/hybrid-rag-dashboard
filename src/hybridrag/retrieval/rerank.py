import logging
import asyncio
from typing import List
from sentence_transformers import CrossEncoder

logger = logging.getLogger("hybrid-rag-reranker")

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Using ms-marco-MiniLM-L-6-v2 as it provides excellent ranking impact 
        while remaining lightweight enough (~80MB) for constrained deployments like Render.
        """
        self.model_name = model_name
        self.model = None

    def load(self):
        logger.info(f"Loading CrossEncoder model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        
    async def rerank_async(self, query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
        """
        Reranks the given documents against the query.
        documents should be a list of dicts containing at least 'text' and 'chunk_id'.
        """
        if not documents:
            return []
            
        if not self.model:
            self.load()
            
        pairs = [[query, doc["text"]] for doc in documents]
        
        loop = asyncio.get_running_loop()
        # CrossEncoder scoring is CPU intensive, run in executor
        scores = await loop.run_in_executor(None, lambda: self.model.predict(pairs))
        
        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            
        # Sort by rerank score descending
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
