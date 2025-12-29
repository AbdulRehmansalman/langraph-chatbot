import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../../services/api';
import { useAuthStore } from '../../stores/authStore';
import { storage } from '../../utils/storage';
import type { LoginCredentials, RegisterData } from '../../types';

// Query Keys
export const authKeys = {
  all: ['auth'] as const,
  user: () => [...authKeys.all, 'user'] as const,
};

// Queries
export const useCurrentUser = () => {
  const { isAuthenticated } = useAuthStore();

  return useQuery({
    queryKey: authKeys.user(),
    queryFn: authAPI.getCurrentUser,
    enabled: isAuthenticated,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Mutations
export const useLogin = () => {
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (credentials: LoginCredentials) => authAPI.login(credentials),
    onSuccess: async response => {
      console.log('Login response:', response);

      if (response && response.access_token) {
        if (response.user) {
          // If user data is included in the response
          setAuth(response.user, response.access_token);
          queryClient.setQueryData(authKeys.user(), response.user);
          navigate('/dashboard');
        } else {
          // If user data is not included, store token first then fetch user data
          storage.set('token', response.access_token);
          try {
            const user = await authAPI.getCurrentUser();
            setAuth(user, response.access_token);
            queryClient.setQueryData(authKeys.user(), user);
            navigate('/dashboard');
          } catch (error) {
            console.error('Failed to fetch user data after login:', error);
            storage.remove('token');
          }
        }
      }
    },
    onError: error => {
      console.error('Login failed:', error);
    },
  });
};

export const useRegister = () => {
  return useMutation({
    mutationFn: (data: RegisterData) => authAPI.register(data),
    onError: error => {
      console.error('Registration failed:', error);
    },
  });
};

export const useVerifyEmail = () => {
  return useMutation({
    mutationFn: ({ email, otp }: { email: string; otp: string }) => authAPI.verifyEmail(email, otp),
    onError: error => {
      console.error('Email verification failed:', error);
    },
  });
};

export const useResendVerification = () => {
  return useMutation({
    mutationFn: (email: string) => authAPI.resendVerification(email),
    onError: error => {
      console.error('Resend verification failed:', error);
    },
  });
};

export const useForgotPassword = () => {
  return useMutation({
    mutationFn: (email: string) => authAPI.forgotPassword(email),
    onError: error => {
      console.error('Forgot password failed:', error);
    },
  });
};

export const useResetPassword = () => {
  return useMutation({
    mutationFn: ({
      email,
      otp,
      newPassword,
    }: {
      email: string;
      otp: string;
      newPassword: string;
    }) => authAPI.resetPassword(email, otp, newPassword),
    onError: error => {
      console.error('Reset password failed:', error);
    },
  });
};

export const useChangePassword = () => {
  return useMutation({
    mutationFn: ({
      currentPassword,
      newPassword,
    }: {
      currentPassword: string;
      newPassword: string;
    }) => authAPI.changePassword(currentPassword, newPassword),
    onError: error => {
      console.error('Change password failed:', error);
    },
  });
};

export const useLogout = () => {
  const navigate = useNavigate();
  const { logout: logoutStore } = useAuthStore();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      // No API call needed for logout, just clear local state
      return Promise.resolve();
    },
    onSuccess: () => {
      logoutStore();
      queryClient.clear(); // Clear all cached data
      navigate('/login');
    },
  });
};
