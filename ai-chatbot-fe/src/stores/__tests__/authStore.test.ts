import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useAuthStore } from '../authStore';
import { authAPI } from '../../services/api';
import { mockUser } from '../../tests/mocks/data';

// Mock authAPI
vi.mock('../../services/api', () => ({
  authAPI: {
    getCurrentUser: vi.fn(),
  },
}));

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('authStore', () => {
  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);

    // Reset store state
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      loading: true,
    });
  });

  describe('initial state', () => {
    it('should have correct initial state', () => {
      const state = useAuthStore.getState();

      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.loading).toBe(true);
    });
  });

  describe('logout', () => {
    it('should logout and clear state', () => {
      // Set authenticated state
      useAuthStore.setState({
        user: mockUser,
        token: 'some-token',
        isAuthenticated: true,
        loading: false,
      });

      const { logout } = useAuthStore.getState();
      logout();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.loading).toBe(false);
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('token');
    });
  });

  describe('initialize', () => {
    it('should initialize with stored token', async () => {
      const storedToken = 'stored-token';
      localStorageMock.getItem.mockReturnValue(storedToken);
      vi.mocked(authAPI.getCurrentUser).mockResolvedValue(mockUser);

      const { initialize } = useAuthStore.getState();
      await initialize();

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.token).toBe(storedToken);
      expect(state.isAuthenticated).toBe(true);
      expect(state.loading).toBe(false);
    });

    it('should clear invalid stored token', async () => {
      const invalidToken = 'invalid-token';
      localStorageMock.getItem.mockReturnValue(invalidToken);
      vi.mocked(authAPI.getCurrentUser).mockRejectedValue(new Error('Unauthorized'));

      const { initialize } = useAuthStore.getState();
      await initialize();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.loading).toBe(false);
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('token');
    });

    it('should handle no stored token', async () => {
      localStorageMock.getItem.mockReturnValue(null);

      const { initialize } = useAuthStore.getState();
      await initialize();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.loading).toBe(false);
    });
  });

  describe('setAuth', () => {
    it('should set authentication state', () => {
      const mockToken = 'new-token';

      const { setAuth } = useAuthStore.getState();
      setAuth(mockUser, mockToken);

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.token).toBe(mockToken);
      expect(state.isAuthenticated).toBe(true);
      expect(state.loading).toBe(false);
      expect(localStorageMock.setItem).toHaveBeenCalledWith('token', mockToken);
    });
  });
});
