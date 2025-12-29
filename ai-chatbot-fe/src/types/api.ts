export interface ApiResponse<T = unknown> {
  data: T;
  message?: string;
  status: number;
}

export interface ApiError {
  detail: string;
  status?: number;
  field?: string;
  code?: string;
  details?: any;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export interface ApiRequestConfig {
  timeout?: number;
  retries?: number;
  skipAuth?: boolean;
}

export interface AxiosError {
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
  message?: string;
}
