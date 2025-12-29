export class Storage {
  private prefix: string;

  constructor(prefix: string = 'ai-chatbot') {
    this.prefix = prefix;
  }

  private getKey(key: string): string {
    return `${this.prefix}:${key}`;
  }

  get<T>(key: string): T | null {
    try {
      const item = localStorage.getItem(this.getKey(key));
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error(`Error reading from storage:`, error);
      return null;
    }
  }

  set<T>(key: string, value: T): void {
    try {
      localStorage.setItem(this.getKey(key), JSON.stringify(value));
    } catch (error) {
      console.error(`Error writing to storage:`, error);
    }
  }

  remove(key: string): void {
    try {
      localStorage.removeItem(this.getKey(key));
    } catch (error) {
      console.error(`Error removing from storage:`, error);
    }
  }

  clear(): void {
    try {
      const keys = Object.keys(localStorage);
      const prefixedKeys = keys.filter(key => key.startsWith(this.prefix));
      prefixedKeys.forEach(key => localStorage.removeItem(key));
    } catch (error) {
      console.error(`Error clearing storage:`, error);
    }
  }

  exists(key: string): boolean {
    return localStorage.getItem(this.getKey(key)) !== null;
  }
}

export const storage = new Storage();
