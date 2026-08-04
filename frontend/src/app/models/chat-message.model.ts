/**
 * Domain models shared across the chatbot UI.
 * These mirror the shape we expect the RAG backend to return so the UI can be
 * wired to the real API without changing the components.
 */

export type MessageRole = 'user' | 'assistant';

export type MessageStatus = 'sending' | 'streaming' | 'done' | 'error';

/** A citation pointing back to the source document a RAG answer was grounded in. */
export interface Source {
  /** Human-readable document name, e.g. "Leave Policy 2025.pdf". */
  title: string;
  /** Optional section / page reference within the document. */
  section?: string;
  /** Optional URL or path to open the source document. */
  url?: string;
  /** The retrieved passage the answer drew from. */
  snippet?: string;
  /** Retriever relevance score (0..1) when available. */
  score?: number;
}

/** Token usage + latency metrics surfaced alongside an assistant answer. */
export interface ChatMetrics {
  /** Tokens in the user prompt sent to the model. */
  inputTokens: number;
  /** Tokens in the model's response. */
  outputTokens: number;
  /** Wall-clock time from request to response, in milliseconds. */
  responseTimeMs: number;
  /** True when token counts are estimated client-side rather than reported by the backend. */
  estimated: boolean;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: Date;
  status: MessageStatus;
  /** Grounding citations returned by the RAG pipeline (assistant messages only). */
  sources?: Source[];
  /** Token usage + latency, shown under assistant messages when available. */
  metrics?: ChatMetrics;
}

/** Request payload sent to the RAG backend. */
export interface ChatRequest {
  query: string;
  user_id: string;
  session_id: string;
  role: MessageRole | 'system';
  /** Prior turns for multi-turn context; kept minimal for now. */
  history?: { role: MessageRole; text: string }[];
}

/** Token accounting the backend may report for a turn. */
export interface TokenUsage {
  /** Prompt / input tokens. */
  inputTokens?: number;
  /** Completion / output tokens. */
  outputTokens?: number;
}

/** Response payload returned by the RAG backend. */
export interface ChatResponse {
  answer: string;
  sources?: Source[];
  /** Token usage reported by the backend, when available. */
  usage?: TokenUsage;
}
