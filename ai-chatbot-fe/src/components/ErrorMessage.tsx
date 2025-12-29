import React from 'react';
import type { ApiError } from '../types';

interface ErrorMessageProps {
  error: ApiError | Error | string | null;
  className?: string;
  showRetry?: boolean;
  onRetry?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  error,
  className = '',
  showRetry = false,
  onRetry,
}) => {
  if (!error) return null;

  const getErrorMessage = (error: ApiError | Error | string): string => {
    if (typeof error === 'string') return error;
    if (error instanceof Error) return error.message;
    return error.detail || 'An unexpected error occurred';
  };

  const getErrorCode = (error: ApiError | Error | string): string | undefined => {
    if (typeof error === 'object' && 'code' in error) {
      return error.code;
    }
    return undefined;
  };

  const getErrorField = (error: ApiError | Error | string): string | undefined => {
    if (typeof error === 'object' && 'field' in error) {
      return error.field;
    }
    return undefined;
  };

  const errorMessage = getErrorMessage(error);
  const errorCode = getErrorCode(error);
  const errorField = getErrorField(error);

  return (
    <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0">
          <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium text-red-800">
            {errorCode ? `Error (${errorCode})` : 'Error'}
          </h3>
          <p className="mt-1 text-sm text-red-700">{errorMessage}</p>
          {errorField && <p className="mt-1 text-xs text-red-600">Field: {errorField}</p>}
        </div>
        {showRetry && onRetry && (
          <div className="ml-3 flex-shrink-0">
            <button
              type="button"
              onClick={onRetry}
              className="text-sm font-medium text-red-600 hover:text-red-500 focus:outline-none focus:underline"
            >
              Try again
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
