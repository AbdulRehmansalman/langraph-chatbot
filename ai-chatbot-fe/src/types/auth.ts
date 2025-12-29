export interface User {
  id: string;
  email: string;
  full_name: string;
  created_at: string;
  is_active: boolean;
  email_verified?: boolean;
}

export interface LoginCredentials {
  email: string;
  password: string;
  timezone?: string;
}

export interface RegisterData {
  email: string;
  password: string;
  full_name: string;
  timezone?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user?: User;
}

export interface EmailVerificationData {
  email: string;
  otp: string;
}

export interface PasswordResetData {
  email: string;
  otp: string;
  new_password: string;
}

export interface ChangePasswordData {
  current_password: string;
  new_password: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
}

export type AuthAction =
  | 'LOGIN'
  | 'LOGOUT'
  | 'REGISTER'
  | 'VERIFY_EMAIL'
  | 'RESET_PASSWORD'
  | 'CHANGE_PASSWORD';
