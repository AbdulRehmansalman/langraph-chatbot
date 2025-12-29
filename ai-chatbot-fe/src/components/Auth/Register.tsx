import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { usePasswordToggle } from '../../hooks/usePasswordToggle';
import { useSEO } from '../../hooks/useSEO';
import { authAPI } from '../../services/api';
import type { AxiosError } from '../../types';

const Register: React.FC = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    full_name: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showEmailVerification, setShowEmailVerification] = useState(false);
  const [otp, setOtp] = useState('');

  const navigate = useNavigate();
  const {
    showPassword: showPassword1,
    togglePasswordVisibility: togglePassword1,
    inputType: inputType1,
  } = usePasswordToggle();
  const {
    showPassword: showPassword2,
    togglePasswordVisibility: togglePassword2,
    inputType: inputType2,
  } = usePasswordToggle();

  useSEO({
    title: 'Register - AI Chatbot',
    description:
      'Create a new AI Chatbot account to start using intelligent document analysis and AI-powered chat features.',
    keywords: 'register, sign up, AI chatbot, create account, new user',
    ogTitle: 'Register - AI Chatbot',
    ogDescription:
      'Create a new AI Chatbot account to start using intelligent document analysis and AI-powered chat features.',
    canonical: 'https://ai-chatbot.example.com/register',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      setLoading(false);
      return;
    }

    try {
      // Detect user's timezone automatically
      const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

      const response = await authAPI.register({
        email: formData.email,
        password: formData.password,
        full_name: formData.full_name,
        timezone: userTimezone, // Send detected timezone
      });
      setSuccess(response.message);
      setShowEmailVerification(true);
    } catch (err: unknown) {
      const error = err as AxiosError;
      setError(error.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const handleEmailVerification = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await authAPI.verifyEmail(formData.email, otp);
      setSuccess('Email verified successfully! You can now login.');
      setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (err: unknown) {
      const error = err as AxiosError;
      setError(error.response?.data?.detail || 'Verification failed');
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = async () => {
    try {
      await authAPI.resendVerification(formData.email);
      setSuccess('Verification email sent successfully');
    } catch (err: unknown) {
      const error = err as AxiosError;
      setError(error.response?.data?.detail || 'Failed to resend verification');
    }
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

            {error && <div className="error-message">{error}</div>}
            {success && <div className="success-message">{success}</div>}

            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? 'Verifying...' : 'Verify Email'}
            </button>

            <div className="text-center text-sm">
              <button
                type="button"
                onClick={handleResendVerification}
                className="text-primary-500 hover:text-primary-600 underline"
              >
                Resend verification code
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
        <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
          Register for AI Chatbot
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="full_name" className="block text-sm font-medium text-gray-700 mb-2">
              Full Name
            </label>
            <input
              type="text"
              id="full_name"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              placeholder="Enter your full name"
              className="input-field"
              required
            />
          </div>

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
                type={inputType1}
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password (min 6 characters)"
                className="input-field pr-10"
                required
              />
              <button
                type="button"
                onClick={togglePassword1}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 focus:outline-none"
                aria-label={showPassword1 ? 'Hide password' : 'Show password'}
              >
                {showPassword1 ? (
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

          <div>
            <label
              htmlFor="confirmPassword"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Confirm Password
            </label>
            <div className="relative">
              <input
                type={inputType2}
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm your password"
                className="input-field pr-10"
                required
              />
              <button
                type="button"
                onClick={togglePassword2}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 focus:outline-none"
                aria-label={showPassword2 ? 'Hide password' : 'Show password'}
              >
                {showPassword2 ? (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 711.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L12 12m-2.122-2.122L7.05 7.05M12 12l2.121 2.121M12 12V9m0 0H9m3 0h3m-3 0v3m0-3l2.121-2.121"
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

          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}

          <button type="submit" disabled={loading} className="btn-primary w-full">
            {loading ? 'Creating Account...' : 'Register'}
          </button>

          <div className="text-center text-sm">
            <Link to="/login" className="text-primary-500 hover:text-primary-600 underline">
              Already have an account? Login
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Register;
