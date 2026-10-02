import math
import re
import heapq
from typing import List, Dict, Tuple

class BM25Retriever:
    def __init__(self):
        # Inverted index: token -> {chunk_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.chunk_lengths: Dict[str, int] = {}
        self.total_tokens: int = 0
        self.num_chunks: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower(), re.UNICODE)

    def add_chunk(self, chunk_id: str, text: str):
        tokens = self._tokenize(text)
        if not tokens:
            return
            
        self.chunk_lengths[chunk_id] = len(tokens)
        self.total_tokens += len(tokens)
        self.num_chunks += 1
        
        token_counts: Dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            
        for token, count in token_counts.items():
            if token not in self.inverted_index:
                self.inverted_index[token] = {}
            self.inverted_index[token][chunk_id] = count

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self.num_chunks == 0:
            return []
            
        avgdl = self.total_tokens / self.num_chunks
        k1 = 1.5
        b = 0.75
        N = self.num_chunks
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
            
        scores: Dict[str, float] = {}
        
        for token in query_tokens:
            if token in self.inverted_index:
                chunk_tfs = self.inverted_index[token]
                df = len(chunk_tfs)
                idf = max(0.0001, math.log(1.0 + (N - df + 0.5) / (df + 0.5)))
                
                for cid, tf in chunk_tfs.items():
                    dl = self.chunk_lengths[cid]
                    term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)))
                    scores[cid] = scores.get(cid, 0.0) + term_score
                    
        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
