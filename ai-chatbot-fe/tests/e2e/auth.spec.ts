import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async () => {
    // Setup can be added here when needed
  });

  test.describe('Login', () => {
    test('should display login form correctly', async ({ page }) => {
      await page.goto('/login');

      await expect(page.locator('h2')).toContainText('Login to AI Chatbot');
      await expect(page.locator('[data-testid="email-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="password-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="login-button"]')).toBeVisible();
      await expect(page.locator('text=Forgot your password?')).toBeVisible();
      await expect(page.locator("text=Don't have an account?")).toBeVisible();
    });

    test('should show/hide password when toggle is clicked', async ({ page }) => {
      await page.goto('/login');

      const passwordInput = page.locator('[data-testid="password-input"]');
      const toggleButton = page.locator('[aria-label="Show password"]');

      // Initially password is hidden
      await expect(passwordInput).toHaveAttribute('type', 'password');

      // Click toggle to show password
      await toggleButton.click();
      await expect(passwordInput).toHaveAttribute('type', 'text');
      await expect(page.locator('[aria-label="Hide password"]')).toBeVisible();

      // Click toggle to hide password
      await page.locator('[aria-label="Hide password"]').click();
      await expect(passwordInput).toHaveAttribute('type', 'password');
    });

    test('should login successfully with valid credentials', async ({ page }) => {
      // Mock successful login response
      await page.route('**/api/auth/login', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            access_token: 'mock-token',
            token_type: 'bearer',
          }),
        });
      });

      await page.route('**/api/auth/me', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: '1',
            email: 'test@example.com',
            full_name: 'Test User',
          }),
        });
      });

      await page.goto('/login');

      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.fill('[data-testid="password-input"]', 'password123');
      await page.click('[data-testid="login-button"]');

      await expect(page).toHaveURL('/dashboard');
    });

    test('should show error for invalid credentials', async ({ page }) => {
      await page.route('**/api/auth/login', async route => {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({
            detail: 'Invalid credentials',
          }),
        });
      });

      await page.goto('/login');

      await page.fill('[data-testid="email-input"]', 'wrong@example.com');
      await page.fill('[data-testid="password-input"]', 'wrongpassword');
      await page.click('[data-testid="login-button"]');

      await expect(page.locator('text=Invalid credentials')).toBeVisible();
    });

    test('should handle unverified email', async ({ page }) => {
      await page.route('**/api/auth/login', async route => {
        await route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({
            detail: 'Please verify your email before logging in',
          }),
        });
      });

      await page.goto('/login');

      await page.fill('[data-testid="email-input"]', 'unverified@example.com');
      await page.fill('[data-testid="password-input"]', 'password123');
      await page.click('[data-testid="login-button"]');

      await expect(page.locator('text=Email Verification')).toBeVisible();
      await expect(page.locator('text=Please verify your email')).toBeVisible();
    });
  });

  test.describe('Registration', () => {
    test('should display registration form correctly', async ({ page }) => {
      await page.goto('/register');

      await expect(page.locator('h2')).toContainText('Register for AI Chatbot');
      await expect(page.locator('[data-testid="fullname-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="email-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="password-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="confirm-password-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="register-button"]')).toBeVisible();
    });

    test('should show password toggles for both password fields', async ({ page }) => {
      await page.goto('/register');

      const passwordInput = page.locator('[data-testid="password-input"]');
      const confirmPasswordInput = page.locator('[data-testid="confirm-password-input"]');
      const toggleButtons = page.locator('[aria-label="Show password"]');

      // Initially both passwords are hidden
      await expect(passwordInput).toHaveAttribute('type', 'password');
      await expect(confirmPasswordInput).toHaveAttribute('type', 'password');

      // Toggle first password
      await toggleButtons.first().click();
      await expect(passwordInput).toHaveAttribute('type', 'text');
      await expect(confirmPasswordInput).toHaveAttribute('type', 'password');

      // Toggle second password
      await toggleButtons.last().click();
      await expect(passwordInput).toHaveAttribute('type', 'text');
      await expect(confirmPasswordInput).toHaveAttribute('type', 'text');
    });

    test('should validate password mismatch', async ({ page }) => {
      await page.goto('/register');

      await page.fill('[data-testid="fullname-input"]', 'Test User');
      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.fill('[data-testid="password-input"]', 'password123');
      await page.fill('[data-testid="confirm-password-input"]', 'different123');
      await page.click('[data-testid="register-button"]');

      await expect(page.locator('text=Passwords do not match')).toBeVisible();
    });

    test('should validate minimum password length', async ({ page }) => {
      await page.goto('/register');

      await page.fill('[data-testid="fullname-input"]', 'Test User');
      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.fill('[data-testid="password-input"]', '123');
      await page.fill('[data-testid="confirm-password-input"]', '123');
      await page.click('[data-testid="register-button"]');

      await expect(page.locator('text=Password must be at least 6 characters')).toBeVisible();
    });

    test('should register successfully and show verification screen', async ({ page }) => {
      await page.route('**/api/auth/register', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            message: 'Registration successful. Please check your email for verification.',
          }),
        });
      });

      await page.goto('/register');

      await page.fill('[data-testid="fullname-input"]', 'Test User');
      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.fill('[data-testid="password-input"]', 'password123');
      await page.fill('[data-testid="confirm-password-input"]', 'password123');
      await page.click('[data-testid="register-button"]');

      await expect(page.locator('text=Email Verification')).toBeVisible();
      await expect(page.locator('text=Registration successful')).toBeVisible();
    });
  });

  test.describe('Email Verification', () => {
    test('should verify email successfully', async ({ page }) => {
      // Mock registration
      await page.route('**/api/auth/register', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            message: 'Registration successful. Please check your email for verification.',
          }),
        });
      });

      // Mock email verification
      await page.route('**/api/auth/verify-email', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            message: 'Email verified successfully',
          }),
        });
      });

      await page.goto('/register');

      // Complete registration
      await page.fill('[data-testid="fullname-input"]', 'Test User');
      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.fill('[data-testid="password-input"]', 'password123');
      await page.fill('[data-testid="confirm-password-input"]', 'password123');
      await page.click('[data-testid="register-button"]');

      await expect(page.locator('text=Email Verification')).toBeVisible();

      // Verify email
      await page.fill('[data-testid="otp-input"]', '123456');
      await page.click('[data-testid="verify-email-button"]');

      await expect(page.locator('text=Email verified successfully')).toBeVisible();
    });

    test('should handle invalid verification code', async ({ page }) => {
      // Mock verification failure
      await page.route('**/api/auth/verify-email', async route => {
        await route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({
            detail: 'Invalid verification code',
          }),
        });
      });

      await page.goto('/register');

      // Assume we're already on verification screen
      await page.evaluate(() => {
        // Mock being on verification screen
        window.history.pushState({}, '', '/register');
      });

      await page.fill('[data-testid="otp-input"]', '000000');
      await page.click('[data-testid="verify-email-button"]');

      await expect(page.locator('text=Invalid verification code')).toBeVisible();
    });
  });

  test.describe('Password Reset', () => {
    test('should display forgot password form', async ({ page }) => {
      await page.goto('/forgot-password');

      await expect(page.locator('h2')).toContainText('Forgot Password');
      await expect(page.locator('[data-testid="email-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="send-reset-button"]')).toBeVisible();
    });

    test('should send reset email successfully', async ({ page }) => {
      await page.route('**/api/auth/forgot-password', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            message: 'Password reset email sent successfully',
          }),
        });
      });

      await page.goto('/forgot-password');

      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.click('[data-testid="send-reset-button"]');

      await expect(page.locator('text=Password reset email sent')).toBeVisible();
    });
  });
});
