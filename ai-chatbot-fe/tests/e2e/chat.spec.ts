import { test, expect } from '@playwright/test';
import { AuthHelpers } from './utils/auth-helpers';

test.describe('Chat Functionality', () => {
  let authHelpers: AuthHelpers;

  test.beforeEach(async ({ page }) => {
    authHelpers = new AuthHelpers(page);

    // Mock authentication
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

    // Login before each test
    await authHelpers.login();
  });

  test('should display chat interface correctly', async ({ page }) => {
    await page.goto('/dashboard');

    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="send-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="chat-history"]')).toBeVisible();
  });

  test('should send and receive messages', async ({ page }) => {
    await page.route('**/api/chat/message', async route => {
      const request = await route.request();
      const body = await request.postDataJSON();

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: Date.now().toString(),
          content: `Echo: ${body.message}`,
          role: 'assistant',
          timestamp: new Date().toISOString(),
        }),
      });
    });

    await page.goto('/dashboard');

    const userMessage = 'Hello, how are you?';

    // Send message
    await page.fill('[data-testid="message-input"]', userMessage);
    await page.click('[data-testid="send-button"]');

    // Check user message appears
    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(userMessage);

    // Check assistant response appears
    await expect(page.locator('[data-testid="assistant-message"]').last()).toContainText(
      `Echo: ${userMessage}`
    );

    // Input should be cleared
    await expect(page.locator('[data-testid="message-input"]')).toHaveValue('');
  });

  test('should handle send message with Enter key', async ({ page }) => {
    await page.route('**/api/chat/message', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: Date.now().toString(),
          content: 'Response via Enter key',
          role: 'assistant',
          timestamp: new Date().toISOString(),
        }),
      });
    });

    await page.goto('/dashboard');

    const messageInput = page.locator('[data-testid="message-input"]');

    await messageInput.fill('Message sent with Enter');
    await messageInput.press('Enter');

    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(
      'Message sent with Enter'
    );
    await expect(page.locator('[data-testid="assistant-message"]').last()).toContainText(
      'Response via Enter key'
    );
  });

  test('should handle chat errors gracefully', async ({ page }) => {
    await page.route('**/api/chat/message', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Internal server error',
        }),
      });
    });

    await page.goto('/dashboard');

    await page.fill('[data-testid="message-input"]', 'This will fail');
    await page.click('[data-testid="send-button"]');

    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toContainText(
      'Failed to send message'
    );
  });

  test('should show typing indicator during response', async ({ page }) => {
    let resolveResponse: (value: unknown) => void;
    const responsePromise = new Promise(resolve => {
      resolveResponse = resolve;
    });

    await page.route('**/api/chat/message', async route => {
      // Wait for test to resolve the promise
      await responsePromise;

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: Date.now().toString(),
          content: 'Delayed response',
          role: 'assistant',
          timestamp: new Date().toISOString(),
        }),
      });
    });

    await page.goto('/dashboard');

    await page.fill('[data-testid="message-input"]', 'Test typing indicator');
    await page.click('[data-testid="send-button"]');

    // Should show typing indicator
    await expect(page.locator('[data-testid="typing-indicator"]')).toBeVisible();

    // Resolve the API call
    resolveResponse(true);

    // Typing indicator should disappear
    await expect(page.locator('[data-testid="typing-indicator"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="assistant-message"]').last()).toContainText(
      'Delayed response'
    );
  });

  test('should load chat history on page load', async ({ page }) => {
    await page.route('**/api/chat/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: '1',
            content: 'Previous message 1',
            role: 'user',
            timestamp: '2024-01-01T10:00:00Z',
          },
          {
            id: '2',
            content: 'Previous response 1',
            role: 'assistant',
            timestamp: '2024-01-01T10:01:00Z',
          },
          {
            id: '3',
            content: 'Previous message 2',
            role: 'user',
            timestamp: '2024-01-01T10:02:00Z',
          },
        ]),
      });
    });

    await page.goto('/dashboard');

    // Should display chat history
    await expect(page.locator('[data-testid="user-message"]').first()).toContainText(
      'Previous message 1'
    );
    await expect(page.locator('[data-testid="assistant-message"]').first()).toContainText(
      'Previous response 1'
    );
    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(
      'Previous message 2'
    );
  });

  test('should clear chat history', async ({ page }) => {
    await page.route('**/api/chat/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: '1',
            content: 'Message to be cleared',
            role: 'user',
            timestamp: '2024-01-01T10:00:00Z',
          },
        ]),
      });
    });

    await page.route('**/api/chat/clear', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Chat history cleared' }),
      });
    });

    await page.goto('/dashboard');

    // Verify history is loaded
    await expect(page.locator('[data-testid="user-message"]')).toContainText(
      'Message to be cleared'
    );

    // Clear chat
    await page.click('[data-testid="clear-chat-button"]');
    await page.click('[data-testid="confirm-clear-button"]');

    // Verify chat is cleared
    await expect(page.locator('[data-testid="user-message"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="assistant-message"]')).not.toBeVisible();
  });

  test('should handle empty message submission', async ({ page }) => {
    await page.goto('/dashboard');

    // Try to send empty message
    await page.click('[data-testid="send-button"]');

    // Should not send message (button should be disabled or no API call made)
    await expect(page.locator('[data-testid="user-message"]')).not.toBeVisible();
  });

  test('should display message timestamps', async ({ page }) => {
    await page.route('**/api/chat/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: '1',
            content: 'Timestamped message',
            role: 'user',
            timestamp: '2024-01-01T10:00:00Z',
          },
        ]),
      });
    });

    await page.goto('/dashboard');

    // Should display timestamp
    await expect(page.locator('[data-testid="message-timestamp"]')).toBeVisible();
  });

  test('should scroll to bottom when new message is added', async ({ page }) => {
    await page.route('**/api/chat/message', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: Date.now().toString(),
          content: 'New message',
          role: 'assistant',
          timestamp: new Date().toISOString(),
        }),
      });
    });

    await page.goto('/dashboard');

    // Send multiple messages to create scroll
    for (let i = 1; i <= 10; i++) {
      await page.fill('[data-testid="message-input"]', `Message ${i}`);
      await page.click('[data-testid="send-button"]');
      await page.waitForTimeout(500); // Small delay for each message
    }

    // Check that the chat container is scrolled to bottom
    const chatContainer = page.locator('[data-testid="chat-history"]');
    const scrollTop = await chatContainer.evaluate(el => el.scrollTop);
    const scrollHeight = await chatContainer.evaluate(el => el.scrollHeight);
    const clientHeight = await chatContainer.evaluate(el => el.clientHeight);

    expect(scrollTop + clientHeight).toBeGreaterThanOrEqual(scrollHeight - 50); // Allow for small margin
  });
});
