import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { setupServer } from 'msw/node';
import { render } from '../../../tests/utils';
import Register from '../Register';
import { handlers } from '../../../tests/mocks/handlers';

// Mock react-router-dom
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const server = setupServer(...handlers);

describe('Register Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockNavigate.mockClear();
    server.listen();
  });

  afterEach(() => {
    server.resetHandlers();
    server.close();
  });

  it('renders registration form correctly', () => {
    render(<Register />);

    expect(screen.getByText('Register for AI Chatbot')).toBeInTheDocument();
    expect(screen.getByLabelText(/full name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /register/i })).toBeInTheDocument();
    expect(screen.getByText(/already have an account/i)).toBeInTheDocument();
  });

  it('shows password toggle functionality for both password fields', async () => {
    const user = userEvent.setup();
    render(<Register />);

    const passwordInput = screen.getByLabelText(/^password$/i);
    const confirmPasswordInput = screen.getByLabelText(/confirm password/i);
    const toggleButtons = screen.getAllByLabelText(/show password/i);

    // Initially both password type
    expect(passwordInput).toHaveAttribute('type', 'password');
    expect(confirmPasswordInput).toHaveAttribute('type', 'password');

    // Toggle first password field
    await user.click(toggleButtons[0]);
    expect(passwordInput).toHaveAttribute('type', 'text');
    expect(confirmPasswordInput).toHaveAttribute('type', 'password');

    // Toggle second password field
    await user.click(toggleButtons[1]);
    expect(passwordInput).toHaveAttribute('type', 'text');
    expect(confirmPasswordInput).toHaveAttribute('type', 'text');
  });

  it('validates password match', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Fill form with mismatched passwords
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/^password$/i), 'password123');
    await user.type(screen.getByLabelText(/confirm password/i), 'different123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  it('validates minimum password length', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Fill form with short password
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/^password$/i), '123');
    await user.type(screen.getByLabelText(/confirm password/i), '123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText(/password must be at least 6 characters/i)).toBeInTheDocument();
    });
  });

  it('handles successful registration', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Fill form correctly
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/^password$/i), 'password123');
    await user.type(screen.getByLabelText(/confirm password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText(/registration successful/i)).toBeInTheDocument();
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });
  });

  it('handles registration failure with existing email', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Fill form with existing email
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'existing@example.com');
    await user.type(screen.getByLabelText(/^password$/i), 'password123');
    await user.type(screen.getByLabelText(/confirm password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText(/user already exists/i)).toBeInTheDocument();
    });
  });

  it('handles email verification after registration', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Complete registration
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/^password$/i), 'password123');
    await user.type(screen.getByLabelText(/confirm password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });

    // Fill verification code
    const otpInput = screen.getByLabelText(/verification code/i);
    await user.type(otpInput, '123456');

    // Submit verification
    await user.click(screen.getByRole('button', { name: /verify email/i }));

    await waitFor(
      () => {
        expect(screen.getByText(/email verified successfully/i)).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Should navigate to login after delay
    await waitFor(
      () => {
        expect(mockNavigate).toHaveBeenCalledWith('/login');
      },
      { timeout: 3000 }
    );
  });

  it('shows loading state during registration', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Fill form
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/^password$/i), 'password123');
    await user.type(screen.getByLabelText(/confirm password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /register/i }));

    // Check for loading state
    expect(screen.getByText(/creating account/i)).toBeInTheDocument();
  });

  it('validates required fields', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Try to submit empty form
    await user.click(screen.getByRole('button', { name: /register/i }));

    // Form should not submit (browser validation)
    expect(screen.queryByText(/registration successful/i)).not.toBeInTheDocument();
  });

  it('allows resending verification email after registration', async () => {
    const user = userEvent.setup();
    render(<Register />);

    // Complete registration to get to verification screen
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/^password$/i), 'password123');
    await user.type(screen.getByLabelText(/confirm password/i), 'password123');
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText('Email Verification')).toBeInTheDocument();
    });

    // Click resend verification
    await user.click(screen.getByText(/resend verification code/i));

    await waitFor(() => {
      expect(screen.getByText(/verification email sent successfully/i)).toBeInTheDocument();
    });
  });
});
