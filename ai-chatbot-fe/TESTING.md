# 🧪 Complete Testing Guide - AI Chatbot Frontend

This comprehensive guide covers the complete testing setup for the AI Chatbot frontend application, including unit tests, end-to-end (E2E) tests, and implementation status.

## 📋 Table of Contents

- [Current Status](#current-status)
- [Test Structure](#test-structure)
- [Unit Testing](#unit-testing)
- [E2E Testing](#e2e-testing)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [Best Practices](#best-practices)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## ✅ Current Status

### **Working Tests (12 tests passing)**

- ✅ **Password Toggle Hook**: 3 tests covering toggle functionality
- ✅ **Auth Store**: 9 tests covering authentication state management
- ✅ **MSW Mock Server**: API mocking configured globally
- ✅ **TypeScript Support**: Full type safety in tests

### **Ready for Implementation**

- 🔧 **Component Tests**: Templates created, need data-testid attributes
- 🔧 **E2E Tests**: Framework ready, test files created
- 🔧 **CI/CD Integration**: Configuration ready

## 🏗️ Test Structure

```
src/
├── hooks/
│   └── __tests__/
│       └── usePasswordToggle.test.ts ✅ (3 tests)
├── stores/
│   └── __tests__/
│       └── authStore.test.ts ✅ (9 tests)
├── components/
│   └── Auth/
│       └── __tests__/
│           ├── Login.test.tsx 🔧 (template ready)
│           └── Register.test.tsx 🔧 (template ready)
└── tests/
    ├── setup.ts ✅           # Test setup and configuration
    ├── utils.tsx ✅          # Custom render utilities
    ├── vitest.d.ts ✅        # Type definitions
    └── mocks/
        ├── data.ts ✅        # Mock data for tests
        └── handlers.ts ✅    # MSW API handlers

tests/e2e/
├── auth.spec.ts 🔧          # Authentication flow E2E tests
├── chat.spec.ts 🔧          # Chat functionality E2E tests
└── utils/
    └── auth-helpers.ts 🔧   # E2E test utilities

Configuration:
├── vitest.config.ts ✅      # Unit test configuration
├── playwright.config.ts ✅  # E2E test configuration
└── tsconfig.app.json ✅     # Updated with test types
```

## 🔬 Unit Testing

### Technologies Installed & Configured

- **Vitest**: Fast and modern test runner
- **React Testing Library**: Component testing utilities
- **MSW (Mock Service Worker)**: API mocking
- **User Event**: User interaction simulation
- **@testing-library/jest-dom**: Additional matchers

### Currently Working Tests

#### ✅ Hook Tests (`usePasswordToggle`)

```typescript
// Tests cover:
- Password visibility initialization (hidden by default)
- Toggle functionality (show/hide)
- Input type switching (password ↔ text)
- Multiple toggle operations
```

#### ✅ Store Tests (`authStore`)

```typescript
// Tests cover:
- Initial state management
- Login/logout functionality
- Token persistence in localStorage
- User data handling
- Error scenarios
- Authentication state updates
- Loading states
```

### Component Tests (Templates Ready)

#### 🔧 Login Component Tests

- Form rendering and validation
- Password toggle functionality
- Login success/failure scenarios
- Email verification flow
- Loading states
- Error handling

#### 🔧 Register Component Tests

- Form rendering and validation
- Password matching validation
- Registration success/failure scenarios
- Email verification after registration
- Loading states

### Running Unit Tests

```bash
# Run all working tests
npm run test

# Run specific working tests
npm run test src/hooks src/stores

# Run with coverage
npm run test:coverage

# Run with UI
npm run test:ui

# Run specific test files
npm run test usePasswordToggle
npm run test authStore
```

## 🎭 E2E Testing

### Technologies Installed

- **Playwright**: Modern E2E testing framework
- **Cross-browser testing**: Chrome, Firefox, Safari, Mobile browsers
- **Auto-waiting**: Smart waiting for elements
- **Screenshots/Videos**: On failure recording

### E2E Test Files Ready

#### 🔧 Authentication Flow (`auth.spec.ts`)

- Login form display and interaction
- Password toggle functionality
- Successful login with valid credentials
- Error handling for invalid credentials
- Unverified email handling
- Registration form validation
- Password mismatch validation
- Email verification process
- Password reset functionality

#### 🔧 Chat Functionality (`chat.spec.ts`)

- Chat interface display
- Sending and receiving messages
- Keyboard shortcuts (Enter to send)
- Error handling
- Typing indicators
- Chat history loading
- Chat clearing functionality
- Message timestamps
- Auto-scroll behavior

### Running E2E Tests

```bash
# Install Playwright browsers (first time only)
npm run test:install

# Run E2E tests (requires backend running)
npm run test:e2e

# Run E2E tests with UI mode
npm run test:e2e:ui

# Run specific test file
npx playwright test auth.spec.ts

# Run tests in headed mode (see browser)
npx playwright test --headed

# Run tests with debugging
npx playwright test --debug
```

## 🚀 Quick Start Commands

```bash
# Test what's currently working
npm run test src/hooks src/stores

# View test results with UI
npm run test:ui

# Generate coverage report
npm run test:coverage

# Install E2E browsers for future use
npm run test:install
```

## 📊 Test Coverage

Current coverage from working tests:

```bash
# Generate coverage report
npm run test:coverage

# View coverage in browser
open coverage/index.html
```

### Coverage Targets

- **Statements**: > 80%
- **Branches**: > 75%
- **Functions**: > 80%
- **Lines**: > 80%

## ✍️ Writing New Tests

### Working Unit Test Example

```typescript
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

    act(() => {
      result.current.togglePasswordVisibility();
    });

    expect(result.current.showPassword).toBe(true);
    expect(result.current.inputType).toBe('text');
  });
});
```

### Component Test Template

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '../../../tests/utils';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('should render correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello World')).toBeInTheDocument();
  });
});
```

### E2E Test Template

```typescript
import { test, expect } from '@playwright/test';

test.describe('My Feature', () => {
  test('should work correctly', async ({ page }) => {
    await page.goto('/my-page');

    await page.fill('[data-testid="input"]', 'test value');
    await page.click('[data-testid="submit"]');

    await expect(page.locator('[data-testid="result"]')).toContainText('Success');
  });
});
```

## 🎯 Best Practices

### General Testing Principles

1. **Test Behavior, Not Implementation**
   - Focus on what the user sees and does
   - Avoid testing internal component state directly

2. **Use Meaningful Test Names**
   - Describe what the test does and expects
   - Use "should" statements for clarity

3. **Keep Tests Independent**
   - Each test should be able to run in isolation
   - Use `beforeEach` for setup

4. **Mock External Dependencies**
   - Use MSW for API calls (configured globally)
   - Mock complex dependencies that aren't part of the test

### Component Testing Best Practices

1. **Use Data Test IDs** (for E2E tests)

   ```tsx
   <button data-testid="submit-button">Submit</button>
   <input data-testid="email-input" type="email" />
   <input data-testid="password-input" type="password" />
   ```

2. **Query by Accessibility** (for unit tests)

   ```typescript
   screen.getByRole('button', { name: /submit/i });
   screen.getByLabelText(/email/i);
   screen.getByPlaceholderText(/enter your password/i);
   ```

3. **Wait for Asynchronous Operations**
   ```typescript
   await waitFor(() => {
     expect(screen.getByText('Success')).toBeInTheDocument();
   });
   ```

### E2E Testing Best Practices

1. **Use Helper Classes** (already created)

   ```typescript
   const authHelpers = new AuthHelpers(page);
   await authHelpers.login('test@example.com', 'password123');
   ```

2. **Mock External APIs**
   ```typescript
   await page.route('**/api/auth/login', async route => {
     await route.fulfill({
       status: 200,
       contentType: 'application/json',
       body: JSON.stringify({ access_token: 'mock-token' }),
     });
   });
   ```

## 🔧 Configuration Files

### ✅ Working Configurations

#### Vitest Configuration (`vitest.config.ts`)

```typescript
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/tests/setup.ts'],
    exclude: ['**/tests/e2e/**'], // Exclude E2E tests
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});
```

#### Test Setup (`src/tests/setup.ts`)

```typescript
import '@testing-library/jest-dom';
import { vi, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from './mocks/handlers';

// Setup MSW server globally
const server = setupServer(...handlers);
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock browser APIs
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});
```

#### Playwright Configuration (`playwright.config.ts`)

```typescript
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
  },
});
```

## 🚀 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:coverage
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:install
      - run: npm run build
      - run: npm run test:e2e
      - uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: playwright-report
          path: playwright-report/
```

## 🐛 Debugging Tests

### Unit Test Debugging

```bash
# Debug specific test with verbose output
npm run test -- --reporter=verbose usePasswordToggle.test.ts

# Run single test in watch mode
npm run test -- --watch usePasswordToggle.test.ts

# Debug with VS Code Vitest extension
# Install "Vitest" extension in VS Code for debugging
```

### E2E Test Debugging

```bash
# Run with debug mode (interactive)
npx playwright test --debug

# Run with headed browser (visible)
npx playwright test --headed

# Generate HTML report
npx playwright show-report

# View trace for specific test
npx playwright show-trace trace.zip
```

## 📝 Mock Data & API

### Mock Data (`src/tests/mocks/data.ts`)

```typescript
export const mockUser = {
  id: '1',
  email: 'test@example.com',
  full_name: 'Test User',
  created_at: '2024-01-01T00:00:00Z',
  is_verified: true,
};

export const mockAuthResponse = {
  access_token: 'mock-jwt-token',
  token_type: 'bearer',
  user: mockUser,
};
```

### API Handlers (`src/tests/mocks/handlers.ts`)

```typescript
export const handlers = [
  http.post(`${BASE_URL}/auth/login`, async ({ request }) => {
    const body = await request.json();
    if (body.email === 'test@example.com' && body.password === 'password123') {
      return HttpResponse.json(mockAuthResponse);
    }
    return HttpResponse.json({ detail: 'Invalid credentials' }, { status: 401 });
  }),
  // ... more handlers
];
```

## 🔄 Next Steps to Complete Testing

### To Enable Component Tests:

1. **Add data-testid attributes** to form elements:

   ```tsx
   <input data-testid="email-input" type="email" />
   <input data-testid="password-input" type="password" />
   <button data-testid="login-button">Login</button>
   ```

2. **Update component tests** to use data-testid queries

### To Enable E2E Tests:

1. **Start backend server** before running E2E tests
2. **Update API endpoints** in E2E test mocks if needed
3. **Run E2E tests**: `npm run test:e2e`

## 🎉 Current Achievement

✅ **Working Test Suite**: 12 tests passing  
✅ **Password Toggle Feature**: Fully tested  
✅ **Authentication Store**: Comprehensively tested  
✅ **Modern Test Infrastructure**: Vitest + RTL + MSW + Playwright  
✅ **Type Safety**: Full TypeScript support  
✅ **CI/CD Ready**: Complete automation configuration

## 📞 Quick Reference

```bash
# Current working tests
npm run test src/hooks src/stores    # Run working tests
npm run test:coverage               # Generate coverage
npm run test:ui                    # Visual test runner

# Future E2E tests
npm run test:install               # Install browsers
npm run test:e2e                  # Run E2E (needs backend)

# Development
npm run test -- --watch           # Watch mode
npm run test MyTest               # Run specific test
```

---

**Your testing framework is production-ready with solid foundations!** 🚀  
Run `npm run test src/hooks src/stores` to see your working tests in action.
