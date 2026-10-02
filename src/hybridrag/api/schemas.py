from pydantic import BaseModel, Field
from typing import List

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
    answer: str = ""
