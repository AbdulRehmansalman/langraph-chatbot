export interface Document {
  id: string;
  filename: string;
  content_type: string;
  file_path: string;
  storage_url: string;
  user_id: string;
  processed: boolean;
  created_at: string;
  updated_at?: string;
  file_size?: number;
  metadata?: DocumentMetadata;
}

export interface DocumentMetadata {
  page_count?: number;
  word_count?: number;
  language?: string;
  tags?: string[];
  description?: string;
}

export interface DocumentUploadResponse {
  message: string;
  document_id: string;
  filename: string;
  storage_url: string;
}

export interface DocumentState {
  documents: Document[];
  selectedDocuments: string[];
  loading: boolean;
  uploadProgress: number;
  error: string | null;
}

export type DocumentFilter = {
  processed?: boolean;
  content_type?: string;
  created_after?: string;
  created_before?: string;
  search?: string;
};

export type SupportedFileType = 'pdf' | 'txt' | 'docx';

export const SUPPORTED_FILE_TYPES: Record<SupportedFileType, string> = {
  pdf: 'application/pdf',
  txt: 'text/plain',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
};

export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
