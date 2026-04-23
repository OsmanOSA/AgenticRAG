const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface SourceItem {
  content: string;
  type: "Text" | "Table";
  source: string | null;
  page_number: number | null;
  rerank_score: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceItem[];
  query: string;
}

export interface StatusResponse {
  collection: string;
  point_count: number;
  ready: boolean;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

export async function queryRAG(
  query: string,
  k = 10,
  top_k = 5
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k, top_k }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? "Query failed");
  }
  return res.json();
}

export async function triggerIngest(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/ingest`, { method: "POST" });
  if (!res.ok) throw new Error("Ingestion failed");
}

export interface DocumentStats {
  filename: string;
  path: string;
  text_chunks: number;
  table_chunks: number;
  image_chunks: number;
  page_count: number;
  estimated_tokens: number;
}

export interface StatsResponse {
  total_chunks: number;
  text_chunks: number;
  table_chunks: number;
  image_chunks: number;
  document_count: number;
  total_chars: number;
  estimated_tokens: number;
  documents: DocumentStats[];
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/api/stats`);
  if (!res.ok) throw new Error("Stats unavailable");
  return res.json();
}
