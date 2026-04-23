from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    k: int = 10
    top_k: int = 5


class SourceItem(BaseModel):
    content: str
    type: str
    source: Optional[str]
    page_number: Optional[int]
    rerank_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    query: str


class StatusResponse(BaseModel):
    collection: str
    point_count: int
    ready: bool


class IngestResponse(BaseModel):
    status: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_indexed: int


class DocumentStats(BaseModel):
    filename: str
    path: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    page_count: int
    estimated_tokens: int


class StatsResponse(BaseModel):
    total_chunks: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    document_count: int
    total_chars: int
    estimated_tokens: int
    documents: List[DocumentStats]
