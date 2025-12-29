import { http, HttpResponse } from 'msw';
import { mockAuthResponse, mockUser } from './data';
import type {
  RegisterData,
  LoginCredentials,
  EmailVerificationData,
  PasswordResetData,
  ChangePasswordData,
  ChatMessage,
} from '../../types';

const BASE_URL = 'http://localhost:8000/api';

export const handlers = [
  // Auth endpoints
  http.post(`${BASE_URL}/auth/register`, async ({ request }) => {
    const body = (await request.json()) as RegisterData;

    if (body.email === 'existing@example.com') {
      return HttpResponse.json({ detail: 'User already exists' }, { status: 400 });
    }

    return HttpResponse.json({
      message: 'Registration successful. Please check your email for verification.',
    });
  }),

  http.post(`${BASE_URL}/auth/login`, async ({ request }) => {
    const body = (await request.json()) as LoginCredentials;

    if (body.email === 'unverified@example.com') {
      return HttpResponse.json(
        { detail: 'Please verify your email before logging in' },
        { status: 400 }
      );
    }

    if (body.email !== 'test@example.com' || body.password !== 'password123') {
      return HttpResponse.json({ detail: 'Invalid credentials' }, { status: 401 });
    }

    return HttpResponse.json(mockAuthResponse);
  }),

  http.post(`${BASE_URL}/auth/verify-email`, async ({ request }) => {
    const body = (await request.json()) as EmailVerificationData;

    if (body.otp !== '123456') {
      return HttpResponse.json({ detail: 'Invalid verification code' }, { status: 400 });
    }

    return HttpResponse.json({
      message: 'Email verified successfully',
    });
  }),

  http.post(`${BASE_URL}/auth/forgot-password`, () => {
    return HttpResponse.json({
      message: 'Password reset email sent successfully',
    });
  }),

  http.post(`${BASE_URL}/auth/reset-password`, async ({ request }) => {
    const body = (await request.json()) as PasswordResetData;

    if (body.otp !== '123456') {
      return HttpResponse.json({ detail: 'Invalid reset code' }, { status: 400 });
    }

    return HttpResponse.json({
      message: 'Password reset successfully',
    });
  }),

  http.post(`${BASE_URL}/auth/resend-verification`, () => {
    return HttpResponse.json({
      message: 'Verification email sent successfully',
    });
  }),

  http.post(`${BASE_URL}/auth/change-password`, async ({ request }) => {
    const body = (await request.json()) as ChangePasswordData;

    if (body.current_password !== 'oldpassword') {
      return HttpResponse.json({ detail: 'Current password is incorrect' }, { status: 400 });
    }

    return HttpResponse.json({
      message: 'Password changed successfully',
    });
  }),

  http.get(`${BASE_URL}/auth/me`, () => {
    return HttpResponse.json(mockUser);
  }),

  // Chat endpoints
  http.post(`${BASE_URL}/chat/message`, async ({ request }) => {
    const body = (await request.json()) as ChatMessage;

    return HttpResponse.json({
      id: Date.now().toString(),
      content: `Echo: ${body.message}`,
      role: 'assistant',
      timestamp: new Date().toISOString(),
    });
  }),

  http.get(`${BASE_URL}/chat/history`, () => {
    return HttpResponse.json([]);
  }),

  // Documents endpoints
  http.get(`${BASE_URL}/documents`, () => {
    return HttpResponse.json([]);
  }),

  http.post(`${BASE_URL}/documents/upload`, () => {
    return HttpResponse.json({
      id: '1',
      name: 'test-document.pdf',
      message: 'Document uploaded successfully',
    });
  }),

  // System endpoints
  http.get(`${BASE_URL}/system/llm/status`, () => {
    return HttpResponse.json({
      status: 'healthy',
      model: 'test-model',
      environment: 'test',
    });
  }),
];
