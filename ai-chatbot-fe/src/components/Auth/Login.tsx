import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useLogin, useResendVerification, useVerifyEmail } from '../../hooks/queries';
import { usePasswordToggle } from '../../hooks/usePasswordToggle';
import { useSEO } from '../../hooks/useSEO';
import type { AxiosError } from '../../types';
import { ErrorMessage } from '../ErrorMessage';

const Login: React.FC = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [showEmailVerification, setShowEmailVerification] = useState(false);
  const [otp, setOtp] = useState('');

  const loginMutation = useLogin();
  const verifyEmailMutation = useVerifyEmail();
  const resendVerificationMutation = useResendVerification();
  const { showPassword, togglePasswordVisibility, inputType } = usePasswordToggle();

  useSEO({
    title: 'Login - AI Chatbot',
    description:
      'Sign in to your AI Chatbot account to access intelligent document analysis and chat features.',
    keywords: 'login, sign in, AI chatbot, authentication, account access',
    ogTitle: 'Login - AI Chatbot',
    ogDescription:
      'Sign in to your AI Chatbot account to access intelligent document analysis and chat features.',
    canonical: 'https://ai-chatbot.example.com/login',
  });

  const loading = loginMutation.isPending || verifyEmailMutation.isPending;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Detect user's timezone automatically
    const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

    // Add timezone to login data
    const loginData = {
      ...formData,
      timezone: userTimezone,
    };

    loginMutation.mutate(loginData, {
      onError: (err: unknown) => {
        const error = err as AxiosError;
        // Check for new error format first
        const errorMessage = error.response?.data?.error?.message || error.response?.data?.detail;

        if (errorMessage === 'Please verify your email before logging in') {
          setShowEmailVerification(true);
        }
        // Error will be displayed by the mutation's error property
      },
    });
  };

  const handleEmailVerification = (e: React.FormEvent) => {
    e.preventDefault();

    verifyEmailMutation.mutate(
      { email: formData.email, otp },
      {
        onSuccess: () => {
          // After successful verification, login automatically
          loginMutation.mutate(formData);
        },
      }
    );
  };

  const handleResendVerification = () => {
    resendVerificationMutation.mutate(formData.email);
  };

  if (showEmailVerification) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-500 to-secondary-500 px-4">
        <div className="card w-full max-w-md">
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">Email Verification</h2>
          <p className="text-gray-600 text-center mb-6">
            Please enter the verification code sent to {formData.email}
          </p>

          <form onSubmit={handleEmailVerification} className="space-y-4">
            <div>
              <label htmlFor="otp" className="block text-sm font-medium text-gray-700 mb-2">
                Verification Code
              </label>
              <input
                type="text"
                id="otp"
                value={otp}
                onChange={e => setOtp(e.target.value)}
                placeholder="Enter 6-digit code"
                className="input-field"
                required
              />
            </div>

            {resendVerificationMutation.isSuccess && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-700">
                Verification email sent successfully
              </div>
            )}

            <ErrorMessage error={verifyEmailMutation.error} className="mb-4" />

            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? 'Verifying...' : 'Verify Email'}
            </button>

            <div className="flex flex-col space-y-2 text-center text-sm">
              <button
                type="button"
                onClick={handleResendVerification}
                className="text-primary-500 hover:text-primary-600 underline"
              >
                Resend verification code
              </button>
              <button
                type="button"
                onClick={() => setShowEmailVerification(false)}
                className="text-gray-500 hover:text-gray-600 underline"
              >
                Back to login
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-500 to-secondary-500 px-4">
      <div className="card w-full max-w-md">
        <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">Login to AI Chatbot</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="Enter your email"
              className="input-field"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
              Password
            </label>
            <div className="relative">
              <input
                type={inputType}
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password"
                className="input-field pr-10"
                required
              />
              <button
                type="button"
                onClick={togglePasswordVisibility}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 focus:outline-none"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L12 12m-2.122-2.122L7.05 7.05M12 12l2.121 2.121M12 12V9m0 0H9m3 0h3m-3 0v3m0-3l2.121-2.121"
                    />
                  </svg>
                ) : (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <ErrorMessage error={loginMutation.error} className="mb-4" />

          <button type="submit" disabled={loading} className="btn-primary w-full">
            {loading ? 'Logging in...' : 'Login'}
          </button>

          <div className="text-center text-sm space-y-2">
            <Link
              to="/forgot-password"
              className="text-primary-500 hover:text-primary-600 underline block"
            >
              Forgot your password?
            </Link>
            <Link to="/register" className="text-gray-500 hover:text-gray-600 underline block">
              Don't have an account? Register
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
