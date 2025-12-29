import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authAPI } from '../../services/api';
import { useSEO } from '../../hooks/useSEO';
import type { AxiosError } from '../../types';

const ForgotPassword: React.FC = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const navigate = useNavigate();

  useSEO({
    title: 'Forgot Password - AI Chatbot',
    description:
      'Reset your AI Chatbot account password. Enter your email to receive a password reset code.',
    keywords: 'forgot password, password reset, account recovery, AI chatbot',
    ogTitle: 'Forgot Password - AI Chatbot',
    ogDescription:
      'Reset your AI Chatbot account password. Enter your email to receive a password reset code.',
    canonical: 'https://ai-chatbot.example.com/forgot-password',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await authAPI.forgotPassword(email);
      setSuccess(response.message);
      setTimeout(() => {
        navigate('/reset-password', { state: { email } });
      }, 2000);
    } catch (err: unknown) {
      const error = err as AxiosError;
      setError(error.response?.data?.detail || 'Failed to send reset email');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to from-primary-500 to-secondary-500 px-4">
      <div className="card w-full max-w-md">
        <div className="text-center mb-6">
          <div className="text-4xl mb-4">🔑</div>
          <h2 className="text-2xl font-bold text-gray-800">Forgot Password</h2>
          <p className="text-gray-600 mt-2">
            Enter your email address and we'll send you a code to reset your password.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
              Email Address
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="Enter your email address"
              className="input-field"
              required
            />
          </div>

          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}

          <button type="submit" disabled={loading || !email} className="btn-primary w-full">
            {loading ? 'Sending...' : 'Send Reset Code'}
          </button>

          <div className="text-center text-sm space-y-2">
            <Link to="/login" className="text-primary-500 hover:text-primary-600 underline block">
              Back to Login
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

export default ForgotPassword;
