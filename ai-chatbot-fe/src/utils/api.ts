import type { ApiError } from '../types';

export class ApiException extends Error {
  public status: number;
  public field?: string;
  public code?: string;
  public details?: any;

  constructor(error: ApiError) {
    super(error.detail);
    this.name = 'ApiException';
    this.status = error.status || 500;
    this.field = error.field;
    this.code = error.code;
    this.details = error.details;
  }
}

export const createApiError = (error: unknown): ApiError => {
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosError = error as {
      response?: {
        data?: {
          // New backend error format
          error?: {
            code?: string;
            message?: string;
            field?: string;
            details?: any;
          };
          // Old format fallback
          detail?: string;
          field?: string;
        };
        status?: number;
      };
    };

    const responseData = axiosError.response?.data;

    // Debug logging for error handling
    if (import.meta.env.DEV) {
      console.log('API Error Response:', {
        status: axiosError.response?.status,
        data: responseData,
      });
    }

    // Handle new backend error format
    if (responseData?.error) {
      return {
        detail: responseData.error.message || 'An unexpected error occurred',
        status: axiosError.response?.status || 500,
        field: responseData.error.field,
        code: responseData.error.code,
        details: responseData.error.details,
      };
    }

    // Handle old format for backward compatibility
    if (responseData?.detail) {
      return {
        detail: responseData.detail,
        status: axiosError.response?.status || 500,
        field: responseData.field,
      };
    }

    return {
      detail: 'An unexpected error occurred',
      status: axiosError.response?.status || 500,
    };
  }

  if (error instanceof Error) {
    return {
      detail: error.message,
      status: 500,
    };
  }

  return {
    detail: 'An unexpected error occurred',
    status: 500,
  };
};

export const handleApiError = (error: unknown): never => {
  const apiError = createApiError(error);
  throw new ApiException(apiError);
};

export const withRetry = async <T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry on 4xx errors (client errors)
      if (error && typeof error === 'object' && 'response' in error) {
        const status = (error as { response?: { status?: number } }).response?.status;
        if (status && status >= 400 && status < 500) {
          break;
        }
      }

      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, delay * attempt));
      }
    }
  }

  return handleApiError(lastError);
};

export const isTokenExpired = (error: unknown): boolean => {
  if (error && typeof error === 'object' && 'response' in error) {
    const status = (error as { response?: { status?: number } }).response?.status;
    return status === 401;
  }
  return false;
};
