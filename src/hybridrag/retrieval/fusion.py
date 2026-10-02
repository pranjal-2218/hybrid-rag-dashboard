import heapq
from typing import List, Dict, Tuple

def rrf_fusion(sparse_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]], top_k: int, k_param: int = 60) -> List[Tuple[str, float]]:
    sparse_ranks = {cid: rank for rank, (cid, _) in enumerate(sparse_results, start=1)}
    dense_ranks = {cid: rank for rank, (cid, _) in enumerate(dense_results, start=1)}
    
    candidates = set(sparse_ranks.keys()).union(dense_ranks.keys())
    rrf_scores: Dict[str, float] = {}
    
    for cid in candidates:
        score = 0.0
        if cid in sparse_ranks:
            score += 1.0 / (k_param + sparse_ranks[cid])
        if cid in dense_ranks:
            score += 1.0 / (k_param + dense_ranks[cid])
        rrf_scores[cid] = score
        
    return heapq.nlargest(top_k, rrf_scores.items(), key=lambda x: x[1])
