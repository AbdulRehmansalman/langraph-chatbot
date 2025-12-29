import { Page, expect } from '@playwright/test';

export class AuthHelpers {
  constructor(private page: Page) {}

  async login(email: string = 'test@example.com', password: string = 'password123') {
    await this.page.goto('/login');

    await this.page.fill('[data-testid="email-input"]', email);
    await this.page.fill('[data-testid="password-input"]', password);

    await this.page.click('[data-testid="login-button"]');

    // Wait for navigation to dashboard
    await expect(this.page).toHaveURL('/dashboard');
  }

  async register(
    fullName: string = 'Test User',
    email: string = 'test@example.com',
    password: string = 'password123'
  ) {
    await this.page.goto('/register');

    await this.page.fill('[data-testid="fullname-input"]', fullName);
    await this.page.fill('[data-testid="email-input"]', email);
    await this.page.fill('[data-testid="password-input"]', password);
    await this.page.fill('[data-testid="confirm-password-input"]', password);

    await this.page.click('[data-testid="register-button"]');
  }

  async verifyEmail(otp: string = '123456') {
    await this.page.fill('[data-testid="otp-input"]', otp);
    await this.page.click('[data-testid="verify-email-button"]');
  }

  async logout() {
    await this.page.click('[data-testid="user-menu"]');
    await this.page.click('[data-testid="logout-button"]');

    // Wait for redirect to login
    await expect(this.page).toHaveURL('/login');
  }

  async togglePasswordVisibility(fieldTestId: string) {
    const toggleButton = this.page.locator(`[data-testid="${fieldTestId}"] + button`);
    await toggleButton.click();
  }

  async expectPasswordVisible(fieldTestId: string, visible: boolean = true) {
    const passwordInput = this.page.locator(`[data-testid="${fieldTestId}"]`);
    const expectedType = visible ? 'text' : 'password';
    await expect(passwordInput).toHaveAttribute('type', expectedType);
  }
}

export class ChatHelpers {
  constructor(private page: Page) {}

  async sendMessage(message: string) {
    await this.page.fill('[data-testid="message-input"]', message);
    await this.page.click('[data-testid="send-button"]');
  }

  async expectMessageInChat(message: string, role: 'user' | 'assistant' = 'user') {
    const messageElement = this.page.locator(`[data-testid="${role}-message"]`, {
      hasText: message,
    });
    await expect(messageElement).toBeVisible();
  }

  async waitForAssistantResponse() {
    await this.page.waitForSelector('[data-testid="assistant-message"]', {
      timeout: 10000,
    });
  }

  async clearChat() {
    await this.page.click('[data-testid="clear-chat-button"]');
    await this.page.click('[data-testid="confirm-clear-button"]');
  }
}

export class DocumentHelpers {
  constructor(private page: Page) {}

  async uploadDocument(filePath: string) {
    const fileInput = this.page.locator('[data-testid="file-input"]');
    await fileInput.setInputFiles(filePath);

    await this.page.click('[data-testid="upload-button"]');

    // Wait for upload success
    await expect(this.page.locator('[data-testid="upload-success"]')).toBeVisible();
  }

  async expectDocumentInList(fileName: string) {
    const documentItem = this.page.locator('[data-testid="document-item"]', {
      hasText: fileName,
    });
    await expect(documentItem).toBeVisible();
  }

  async deleteDocument(fileName: string) {
    const documentItem = this.page.locator('[data-testid="document-item"]', {
      hasText: fileName,
    });

    await documentItem.locator('[data-testid="delete-button"]').click();
    await this.page.click('[data-testid="confirm-delete-button"]');
  }
}
