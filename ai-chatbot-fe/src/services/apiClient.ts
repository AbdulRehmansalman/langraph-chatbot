import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import axios from 'axios';
import type { ApiRequestConfig } from '../types';
import { handleApiError, isTokenExpired, withRetry } from '../utils/api';
import { storage } from '../utils/storage';

// Get API URL from environment variable (Vite)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private instance: AxiosInstance;

  constructor(baseURL: string = API_BASE_URL) {
    this.instance = axios.create({
      baseURL,
      timeout: 300000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.instance.interceptors.request.use(
      config => {
        const token = storage.get<string>('token');
        if (token && !config.headers?.skipAuth) {
          config.headers.Authorization = `Bearer ${token}`;
          console.log('Request with token:', {
            url: config.url,
            hasToken: !!token,
            tokenPreview: token.substring(0, 20) + '...',
          });
        } else {
          console.log('Request without token:', {
            url: config.url,
            skipAuth: config.headers?.skipAuth,
          });
        }
        return config;
      },
      error => {
        console.error('Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.instance.interceptors.response.use(
      response => response,
      async error => {
        const originalRequest = error.config;

        // Handle token expiration - but only for authenticated requests
        if (isTokenExpired(error) && !originalRequest._retry) {
          originalRequest._retry = true;

          // Only redirect if this was an authenticated request (had a token)
          const hadToken = originalRequest.headers?.Authorization;
          if (hadToken) {
            storage.remove('token');

            // Don't redirect if we're already on the login page or if this is a login request
            const isLoginRequest = originalRequest.url?.includes('/auth/login');
            const isLoginPage = window.location.pathname === '/login';

            if (typeof window !== 'undefined' && !isLoginRequest && !isLoginPage) {
              window.location.href = '/login';
            }
          }

          return Promise.reject(error);
        }

        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig & ApiRequestConfig): Promise<T> {
    try {
      const response = await this.makeRequest<T>('get', url, undefined, config);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  }

  async post<T>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig & ApiRequestConfig
  ): Promise<T> {
    try {
      const response = await this.makeRequest<T>('post', url, data, config);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  }

  async put<T>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig & ApiRequestConfig
  ): Promise<T> {
    try {
      const response = await this.makeRequest<T>('put', url, data, config);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  }

  async delete<T>(url: string, config?: AxiosRequestConfig & ApiRequestConfig): Promise<T> {
    try {
      const response = await this.makeRequest<T>('delete', url, undefined, config);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  }

  private async makeRequest<T>(
    method: 'get' | 'post' | 'put' | 'delete',
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig & ApiRequestConfig
  ): Promise<AxiosResponse<T>> {
    const { retries = 0, ...axiosConfig } = config || {};

    const requestFn = async () => {
      switch (method) {
        case 'get':
          return this.instance.get<T>(url, axiosConfig);
        case 'post':
          return this.instance.post<T>(url, data, axiosConfig);
        case 'put':
          return this.instance.put<T>(url, data, axiosConfig);
        case 'delete':
          return this.instance.delete<T>(url, axiosConfig);
        default:
          throw new Error(`Unsupported method: ${method}`);
      }
    };

    if (retries > 0) {
      return withRetry(requestFn, retries);
    }

    return requestFn();
  }

  // Upload file with progress tracking
  async uploadFile<T>(
    url: string,
    file: File,
    onProgress?: (percentage: number) => void,
    config?: AxiosRequestConfig & ApiRequestConfig
  ): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.instance.post<T>(url, formData, {
        ...config,
        headers: {
          'Content-Type': 'multipart/form-data',
          ...config?.headers,
        },
        onUploadProgress: progressEvent => {
          if (onProgress && progressEvent.total) {
            const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(percentage);
          }
        },
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  }
}

export const apiClient = new ApiClient();
