// Re-export all types from specific modules
export * from './api';
export * from './auth';
export * from './chat';
export * from './document';
export * from './common';
export * from './google';

// Legacy exports for backward compatibility (to be removed gradually)
export type { User, LoginCredentials, RegisterData, AuthResponse } from './auth';
export type { Document, DocumentUploadResponse as DocumentUpload } from './document';
export type { ChatMessage, ChatResponse, ChatHistory } from './chat';
export type { ApiError } from './api';
