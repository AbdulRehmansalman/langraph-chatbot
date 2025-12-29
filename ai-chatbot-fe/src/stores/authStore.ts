import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User } from '../types';
import { authAPI } from '../services/api';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  logout: () => void;
  initialize: () => Promise<void>;
  setAuth: (user: User, token: string) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, _get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      loading: true,

      logout: () => {
        localStorage.removeItem('token');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          loading: false,
        });
      },

      initialize: async () => {
        const token = localStorage.getItem('token');
        if (token) {
          try {
            const user = await authAPI.getCurrentUser();
            set({
              user,
              token,
              isAuthenticated: true,
              loading: false,
            });
          } catch (error) {
            console.error('Failed to initialize auth:', error);
            localStorage.removeItem('token');
            set({
              user: null,
              token: null,
              isAuthenticated: false,
              loading: false,
            });
          }
        } else {
          set({ loading: false });
        }
      },

      setAuth: (user: User, token: string) => {
        localStorage.setItem('token', token);
        set({
          user,
          token,
          isAuthenticated: true,
          loading: false,
        });
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: state => ({
        token: state.token,
      }),
    }
  )
);
