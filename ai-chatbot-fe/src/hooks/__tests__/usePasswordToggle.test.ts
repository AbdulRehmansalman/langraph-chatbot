import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePasswordToggle } from '../usePasswordToggle';

describe('usePasswordToggle', () => {
  it('should initialize with password hidden', () => {
    const { result } = renderHook(() => usePasswordToggle());

    expect(result.current.showPassword).toBe(false);
    expect(result.current.inputType).toBe('password');
  });

  it('should toggle password visibility', () => {
    const { result } = renderHook(() => usePasswordToggle());

    // Initially hidden
    expect(result.current.showPassword).toBe(false);
    expect(result.current.inputType).toBe('password');

    // Toggle to show
    act(() => {
      result.current.togglePasswordVisibility();
    });

    expect(result.current.showPassword).toBe(true);
    expect(result.current.inputType).toBe('text');

    // Toggle back to hide
    act(() => {
      result.current.togglePasswordVisibility();
    });

    expect(result.current.showPassword).toBe(false);
    expect(result.current.inputType).toBe('password');
  });

  it('should return correct input type based on visibility state', () => {
    const { result } = renderHook(() => usePasswordToggle());

    // Test multiple toggles
    for (let i = 0; i < 5; i++) {
      // Toggle for each iteration
      act(() => {
        result.current.togglePasswordVisibility();
      });

      const shouldShow = (i + 1) % 2 === 1; // After i+1 toggles, should show if odd number of toggles
      expect(result.current.showPassword).toBe(shouldShow);
      expect(result.current.inputType).toBe(shouldShow ? 'text' : 'password');
    }
  });
});
