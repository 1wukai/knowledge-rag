export interface SystemStatus {
  mode: "local" | "server";
  features: {
    conversations: boolean;
    async_ingestion: boolean;
    answer_history: boolean;
    index_management: boolean;
  };
  active_generation_id: string | null;
}

export interface Generation {
  generation_id: string;
  collection: string;
  status: string;
  chunk_count: number;
  embedding_model: string;
  embedding_dim?: number;
  created_at: string | null;
  activated_at: string | null;
}

export interface GenerationList {
  generations: Generation[];
  active: string | null;
  revision: number;
}

export interface AnswerSummary {
  answer_id: string;
  conversation_id: string | null;
  question: string;
  status: string;
  refused: boolean;
  top_score: number | null;
  created_at: string | null;
}

export interface AnswerDetail extends AnswerSummary {
  answer: string | null;
  generation_id: string | null;
  embedding_model: string | null;
  generation_model: string | null;
  reranked: boolean;
  citations: Citation[];
}

export interface ProviderConfig {
  base_url: string;
  api_key: string;
  model: string;
}

export interface RuntimeSettings {
  embedding: ProviderConfig;
  generation: ProviderConfig;
  rerank?: ProviderConfig | null;
}

export interface CorpusDoc {
  doc_id: string;
  path: string;
  title: string;
  hash: string;
  category: string;
  enabled: boolean;
  indexed: boolean;
}

export interface CorpusCandidate {
  path: string;
  title: string;
  category: string;
  hash: string;
  format?: string;
  size_bytes?: number;
  error?: string | null;
}

export interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface Message {
  role: string;
  content: string;
  created_at: string | null;
}

export interface SseEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface IndexDocument {
  doc_id: string;
  title: string;
  hash: string;
  path: string;
}

export interface IndexState {
  collection: string;
  generated_at: string;
  embedding_model: string;
  embedding_dim: number;
  chunk_count: number;
  documents: IndexDocument[];
}

export interface CorpusResponse {
  documents: CorpusDoc[];
  index: IndexState | null;
}

export interface IngestResult {
  doc_ids: string[];
  chunk_count: number;
  embedding_model: string;
  embedding_dim: number;
  collection: string;
  generated_at: string;
  missing: string[];
}

export interface IngestionJob {
  job_id: string;
  status: string;
  manifest_hash: string;
  target_generation_id: string | null;
  error: string | null;
  progress: { chunks?: number } | null;
  created_at: string | null;
  finished_at: string | null;
  chunk_count?: number;
  embedding_model?: string;
  embedding_dim?: number;
  reused?: string[];
  removed?: string[];
}

export interface Citation {
  index: number;
  doc_id: string;
  title: string;
  heading: string[];
  start_line: number;
  end_line: number;
  source_hash: string;
  snippet: string;
}

export interface ChatResponse {
  answer: string | null;
  citations: Citation[];
  index: IndexState;
  refused: boolean;
  top_score: number | null;
  threshold: number | null;
  reranked?: boolean;
}

export interface ProviderTestResult {
  embedding: { ok: boolean; message: string };
  generation: { ok: boolean; message: string };
  rerank?: { ok: boolean; message: string };
}
