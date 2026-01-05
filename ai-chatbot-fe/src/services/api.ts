import type {
  AuthResponse,
  ChangePasswordData,
  ChatHistory,
  ChatMessage,
  ChatResponse,
  Document,
  DocumentUploadResponse,
  EmailVerificationData,
  LoginCredentials,
  PasswordResetData,
  RegisterData,
  User,
} from '../types';
import { apiClient } from './apiClient';

// Auth API
export const authAPI = {
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    return apiClient.post('/api/auth/login', credentials);
  },

  async register(data: RegisterData): Promise<{ message: string; user_id: string }> {
    return apiClient.post('/api/auth/register', data);
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get('/api/auth/me');
  },

  async verifyEmail(email: string, otp: string): Promise<{ message: string }> {
    const data: EmailVerificationData = { email, otp };
    return apiClient.post('/api/auth/verify-email', data);
  },

  async resendVerification(email: string): Promise<{ message: string }> {
    return apiClient.post('/api/auth/resend-verification', { email });
  },

  async forgotPassword(email: string): Promise<{ message: string }> {
    return apiClient.post('/api/auth/forgot-password', { email });
  },

  async resetPassword(
    email: string,
    otp: string,
    new_password: string
  ): Promise<{ message: string }> {
    const data: PasswordResetData = { email, otp, new_password };
    return apiClient.post('/api/auth/reset-password', data);
  },

  async changePassword(
    current_password: string,
    new_password: string
  ): Promise<{ message: string }> {
    const data: ChangePasswordData = { current_password, new_password };
    return apiClient.post('/api/auth/change-password', data);
  },
};

// Documents API
export const documentsAPI = {
  async upload(
    file: File,
    onProgress?: (percentage: number) => void
  ): Promise<DocumentUploadResponse> {
    return apiClient.uploadFile('/api/documents/upload', file, onProgress);
  },

  async getAll(): Promise<Document[]> {
    return apiClient.get('/api/documents/');
  },

  async delete(documentId: string): Promise<{ message: string }> {
    return apiClient.delete(`/api/documents/${documentId}`);
  },

  async getById(documentId: string): Promise<Document> {
    // Stub implementation as backend endpoint is missing
    return Promise.reject(new Error(`Get document ${documentId} not implemented on backend`));
  },
};

// Chat API
export const chatAPI = {
  async getHistory(limit: number = 50): Promise<ChatHistory> {
    return apiClient.get(`/api/chat/history?limit=${limit}`);
  },

  async deleteMessage(messageId: string): Promise<{ message: string }> {
    // Stub implementation
    return Promise.reject(new Error(`Delete message ${messageId} not implemented on backend`));
  },

  async clearHistory(): Promise<{ message: string }> {
    // Stub implementation
    return Promise.reject(new Error('Clear history not implemented on backend'));
  },
};

// Google Calendar API
export const googleCalendarAPI = {
  async startAuth(): Promise<{ auth_url: string; message: string; redirect_uri: string }> {
    return apiClient.get('/api/auth/google/authorize');
  },

  async getStatus(): Promise<{ connected: boolean; expires_at?: string; scopes?: string[] }> {
    return apiClient.get('/api/auth/google/status');
  },

  async disconnect(): Promise<{ message: string }> {
    return apiClient.delete('/api/auth/google/disconnect');
  },
};

// Export the client for custom requests
export { apiClient } from './apiClient';
