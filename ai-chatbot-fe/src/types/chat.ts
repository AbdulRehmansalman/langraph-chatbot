export interface ChatMessage {
  message: string;
  document_ids?: string[];
}

export interface ChatResponse {
  id: string;
  user_message: string;
  bot_response: string;
  document_ids?: string[];
  created_at: string;
  metadata?: {
    processing_time?: number;
    model_used?: string;
    confidence_score?: number;
    provider?: string;
    langgraph_enabled?: boolean;
    langsmith_enabled?: boolean;
  };
}

export interface ChatHistory {
  messages: ChatResponse[];
  total_count?: number;
}

export interface ChatState {
  messages: ChatResponse[];
  loading: boolean;
  error: string | null;
  currentMessage: string;
}

export type MessageRole = 'user' | 'assistant';

export interface MessageMetadata {
  timestamp: string;
  sources?: string[];
  confidence?: number;
}
