export const mockUser = {
  id: '1',
  email: 'test@example.com',
  full_name: 'Test User',
  created_at: '2024-01-01T00:00:00Z',
  is_active: true,
  email_verified: true,
};

export const mockAuthResponse = {
  access_token: 'mock-jwt-token',
  token_type: 'bearer',
  user: mockUser,
};

export const mockRegisterRequest = {
  email: 'test@example.com',
  password: 'password123',
  full_name: 'Test User',
};

export const mockLoginRequest = {
  email: 'test@example.com',
  password: 'password123',
};

export const mockChatMessage = {
  id: '1',
  content: 'Hello, how are you?',
  role: 'user' as const,
  timestamp: '2024-01-01T00:00:00Z',
};

export const mockChatResponse = {
  id: '2',
  content: 'I am doing well, thank you for asking!',
  role: 'assistant' as const,
  timestamp: '2024-01-01T00:00:01Z',
};

export const mockDocument = {
  id: '1',
  name: 'test-document.pdf',
  size: 1024,
  type: 'application/pdf',
  upload_date: '2024-01-01T00:00:00Z',
  status: 'processed' as const,
};
