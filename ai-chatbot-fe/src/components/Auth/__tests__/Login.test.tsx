import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '../../../tests/utils';
import Login from '../Login';
import { useAuthStore } from '../../../stores/authStore';

// Mock the auth store
vi.mock('../../../stores/authStore', () => ({
  useAuthStore: vi.fn(),
}));

// Mock react-router-dom
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('Login Component', () => {
  const mockLogin = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockNavigate.mockClear();

    // Mock useAuthStore return value
    vi.mocked(useAuthStore).mockReturnValue({
      login: mockLogin,
      user: null,
      token: null,
      isAuthenticated: false,
      loading: false,
      logout: vi.fn(),
      initialize: vi.fn(),
      setLoading: vi.fn(),
      setAuth: vi.fn(),
    });
  });

  it('renders login form correctly', () => {
    render(<Login />);

    expect(screen.getByText('Login to AI Chatbot')).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByDisplayValue('')).toBeInTheDocument(); // Password input
    expect(screen.getByRole('button', { name: /login/i })).toBeInTheDocument();
    expect(screen.getByText(/forgot your password/i)).toBeInTheDocument();
    expect(screen.getByText(/don't have an account/i)).toBeInTheDocument();
  });

  it('shows password toggle functionality', async () => {
    const user = userEvent.setup();
    render(<Login />);

    const passwordInput = screen.getByPlaceholderText(/enter your password/i);
    const toggleButton = screen.getByLabelText(/show password/i);

    // Initially password type
    expect(passwordInput).toHaveAttribute('type', 'password');

    // Toggle to show password
    await user.click(toggleButton);
    expect(passwordInput).toHaveAttribute('type', 'text');
    expect(screen.getByLabelText(/hide password/i)).toBeInTheDocument();

    // Toggle back to hide password
    const hideToggleButton = screen.getByLabelText(/hide password/i);
    await user.click(hideToggleButton);
    expect(passwordInput).toHaveAttribute('type', 'password');
    expect(screen.getByLabelText(/show password/i)).toBeInTheDocument();
  });

  it('handles successful login', async () => {
    const user = userEvent.setup();
    mockLogin.mockResolvedValue(undefined);

    render(<Login />);

    // Fill form
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByPlaceholderText(/enter your password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /login/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('mock-jwt-token');
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
  });

  it('handles login failure with invalid credentials', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Fill form with invalid credentials
    await user.type(screen.getByLabelText(/email/i), 'wrong@example.com');
    await user.type(screen.getByLabelText(/password/i), 'wrongpassword');

    // Submit form
    await user.click(screen.getByRole('button', { name: /login/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });

  it('handles unverified email', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Fill form with unverified email
    await user.type(screen.getByLabelText(/email/i), 'unverified@example.com');
    await user.type(screen.getByPlaceholderText(/enter your password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /login/i }));

    await waitFor(() => {
      expect(screen.getByText(/please verify your email/i)).toBeInTheDocument();
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });
  });

  it('handles email verification flow', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Trigger email verification screen
    await user.type(screen.getByLabelText(/email/i), 'unverified@example.com');
    await user.type(screen.getByPlaceholderText(/enter your password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /login/i }));

    await waitFor(() => {
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });

    // Fill verification code
    const otpInput = screen.getByLabelText(/verification code/i);
    await user.type(otpInput, '123456');

    // Submit verification
    await user.click(screen.getByRole('button', { name: /verify email/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('mock-jwt-token');
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
  });

  it('validates required fields', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Try to submit empty form
    await user.click(screen.getByRole('button', { name: /login/i }));

    // Form should not submit (browser validation)
    expect(mockLogin).not.toHaveBeenCalled();
  });

  it('shows loading state during submission', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Fill form
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByPlaceholderText(/enter your password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /login/i }));

    // Check for loading state
    expect(screen.getByText(/logging in/i)).toBeInTheDocument();
  });

  it('allows resending verification email', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Trigger email verification screen
    await user.type(screen.getByLabelText(/email/i), 'unverified@example.com');
    await user.type(screen.getByPlaceholderText(/enter your password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /login/i }));

    await waitFor(() => {
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });

    // Click resend verification
    await user.click(screen.getByText(/resend verification code/i));

    await waitFor(() => {
      expect(screen.getByText(/verification email sent successfully/i)).toBeInTheDocument();
    });
  });

  it('allows going back to login from verification screen', async () => {
    const user = userEvent.setup();
    render(<Login />);

    // Trigger email verification screen
    await user.type(screen.getByLabelText(/email/i), 'unverified@example.com');
    await user.type(screen.getByPlaceholderText(/enter your password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /login/i }));

    await waitFor(() => {
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });

    // Click back to login
    await user.click(screen.getByText(/back to login/i));

    await waitFor(() => {
      expect(screen.getByText('Login to AI Chatbot')).toBeInTheDocument();
    });
  });
});
