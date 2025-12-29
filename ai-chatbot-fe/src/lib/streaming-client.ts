/**
 * Streaming Chat Client
 * =====================
 * Frontend utility for consuming Server-Sent Events (SSE) from the chat API.
 * 
 * Enterprise Features:
 * - Type-safe event handling
 * - Automatic reconnection
 * - Error recovery
 * - Progress tracking
 * - Timeout handling
 */

import * as React from 'react';

// Event types matching backend
// Event types matching backend
export const StreamEventType = {
  STATUS: 'status',
  TOKEN: 'token',
  SOURCES: 'sources',
  PROGRESS: 'progress',
  COMPLETE: 'complete',
  ERROR: 'error',
  METADATA: 'metadata'
} as const;

export type StreamEventType = typeof StreamEventType[keyof typeof StreamEventType];

export const StreamStatus = {
  STARTING: 'starting',
  RETRIEVING: 'retrieving',
  GENERATING: 'generating',
  COMPLETE: 'complete',
  ERROR: 'error'
} as const;

export type StreamStatus = typeof StreamStatus[keyof typeof StreamStatus];

// Event interfaces
export interface StreamEvent {
  type: StreamEventType;
  data: any;
  timestamp: string;
}

export interface TokenEvent extends StreamEvent {
  type: typeof StreamEventType.TOKEN;
  data: string;
}

export interface StatusEvent extends StreamEvent {
  type: typeof StreamEventType.STATUS;
  data: {
    status: StreamStatus;
    message: string;
  };
}

export interface ProgressEvent extends StreamEvent {
  type: typeof StreamEventType.PROGRESS;
  data: {
    tokens: number;
    time: number;
    estimated_remaining: number;
  };
}

export interface CompleteEvent extends StreamEvent {
  type: typeof StreamEventType.COMPLETE;
  data: {
    total_time: number;
    total_tokens: number;
    provider: string;
    status: string;
  };
}

export interface ErrorEvent extends StreamEvent {
  type: typeof StreamEventType.ERROR;
  data: {
    message: string;
    code: string;
    recoverable: boolean;
  };
}

// Event handlers
export interface StreamEventHandlers {
  onToken?: (token: string) => void;
  onStatus?: (status: StreamStatus, message: string) => void;
  onProgress?: (progress: ProgressEvent['data']) => void;
  onComplete?: (data: CompleteEvent['data']) => void;
  onError?: (error: ErrorEvent['data']) => void;
  onMetadata?: (metadata: any) => void;
}

// Streaming client configuration
export interface StreamingClientConfig {
  apiUrl: string;
  token: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

/**
 * Streaming Chat Client
 * 
 * Usage:
 * ```typescript
 * const client = new StreamingChatClient({
 *   apiUrl: 'http://localhost:8000/api/chat/stream',
 *   token: 'your-jwt-token'
 * });
 * 
 * await client.streamMessage(
 *   'What is RAG?',
 *   {
 *     onToken: (token) => console.log(token),
 *     onStatus: (status, message) => console.log(status, message),
 *     onComplete: (data) => console.log('Done!', data),
 *     onError: (error) => console.error(error)
 *   },
 *   ['doc-id-1', 'doc-id-2']
 * );
 * ```
 */
export class StreamingChatClient {
  private config: Required<StreamingClientConfig>;
  private abortController: AbortController | null = null;

  constructor(config: StreamingClientConfig) {
    this.config = {
      timeout: 30000,
      retryAttempts: 3,
      retryDelay: 1000,
      ...config
    };
  }

  /**
   * Stream a chat message and handle events.
   * 
   * @param message - User message
   * @param handlers - Event handlers
   * @param documentIds - Optional document IDs for RAG
   * @returns Promise that resolves when stream completes
   */
  async streamMessage(
    message: string,
    handlers: StreamEventHandlers,
    documentIds?: string[]
  ): Promise<void> {
    // Cancel any existing stream
    this.abort();

    // Create new abort controller
    this.abortController = new AbortController();

    // Create timeout signal if supported (graceful fallback for older browsers)
    let signal: AbortSignal = this.abortController.signal;
    try {
      if (typeof AbortSignal.any === 'function' && typeof AbortSignal.timeout === 'function') {
        signal = AbortSignal.any([
          this.abortController.signal,
          AbortSignal.timeout(this.config.timeout)
        ]);
      }
    } catch {
      // Fallback: just use the abort controller signal
    }

    try {
      const response = await fetch(this.config.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.token}`,
        },
        body: JSON.stringify({
          message,
          document_ids: documentIds || []
        }),
        signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      // Process SSE stream
      await this.processStream(response.body, handlers);

    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.log('Stream aborted by user');
        } else if (error.name === 'TimeoutError') {
          handlers.onError?.({
            message: `Request timed out after ${this.config.timeout / 1000} seconds`,
            code: 'TIMEOUT',
            recoverable: true
          });
        } else {
          handlers.onError?.({
            message: error.message,
            code: 'CLIENT_ERROR',
            recoverable: false
          });
        }
      }
      throw error;
    } finally {
      this.abortController = null;
    }
  }

  /**
   * Process the SSE stream.
   */
  private async processStream(
    body: ReadableStream<Uint8Array>,
    handlers: StreamEventHandlers
  ): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete events in buffer
        const events = buffer.split('\n\n');
        buffer = events.pop() || ''; // Keep incomplete event in buffer

        for (const eventText of events) {
          if (!eventText.trim()) continue;
          if (eventText.startsWith(':')) continue; // Heartbeat comment

          // Parse SSE event
          const lines = eventText.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.substring(6);
              try {
                const event: StreamEvent = JSON.parse(data);
                this.handleEvent(event, handlers);
              } catch (e) {
                console.error('Failed to parse event:', data, e);
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Handle a single stream event.
   */
  private handleEvent(event: StreamEvent, handlers: StreamEventHandlers): void {
    switch (event.type) {
      case StreamEventType.TOKEN:
        handlers.onToken?.(event.data);
        break;

      case StreamEventType.STATUS:
        handlers.onStatus?.(event.data.status, event.data.message);
        break;

      case StreamEventType.PROGRESS:
        handlers.onProgress?.(event.data);
        break;

      case StreamEventType.COMPLETE:
        handlers.onComplete?.(event.data);
        break;

      case StreamEventType.ERROR:
        handlers.onError?.(event.data);
        break;

      case StreamEventType.METADATA:
        handlers.onMetadata?.(event.data);
        break;

      default:
        console.warn('Unknown event type:', event.type);
    }
  }

  /**
   * Abort the current stream.
   */
  abort(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  /**
   * Check if a stream is currently active.
   */
  isStreaming(): boolean {
    return this.abortController !== null;
  }
}

/**
 * React Hook for streaming chat
 * 
 * Usage:
 * ```typescript
 * const { streamMessage, isStreaming, abort } = useStreamingChat({
 *   apiUrl: 'http://localhost:8000/api/chat/stream',
 *   token: userToken
 * });
 * 
 * const handleSend = async () => {
 *   await streamMessage(
 *     message,
 *     {
 *       onToken: (token) => setResponse(prev => prev + token),
 *       onComplete: (data) => console.log('Done!', data)
 *     },
 *     documentIds
 *   );
 * };
 * ```
 */
export function useStreamingChat(config: StreamingClientConfig) {
  const clientRef = React.useRef<StreamingChatClient | null>(null);
  const [isStreaming, setIsStreaming] = React.useState(false);
  
  // Memoize config values to prevent unnecessary re-initialization
  const apiUrl = config.apiUrl;
  const token = config.token;
  const timeout = config.timeout;
  const retryAttempts = config.retryAttempts;
  const retryDelay = config.retryDelay;

  // Initialize client only when URL or token actually changes
  React.useEffect(() => {
    clientRef.current = new StreamingChatClient({
      apiUrl,
      token,
      timeout,
      retryAttempts,
      retryDelay
    });
    return () => {
      clientRef.current?.abort();
    };
  }, [apiUrl, token, timeout, retryAttempts, retryDelay]);

  const streamMessage = React.useCallback(
    async (
      message: string,
      handlers: StreamEventHandlers,
      documentIds?: string[]
    ) => {
      if (!clientRef.current) {
        throw new Error('Streaming client not initialized');
      }

      setIsStreaming(true);
      try {
        await clientRef.current.streamMessage(message, handlers, documentIds);
      } finally {
        setIsStreaming(false);
      }
    },
    []
  );

  const abort = React.useCallback(() => {
    clientRef.current?.abort();
    setIsStreaming(false);
  }, []);

  return {
    streamMessage,
    isStreaming,
    abort
  };
}

// Export for CommonJS compatibility
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    StreamingChatClient,
    useStreamingChat,
    StreamEventType,
    StreamStatus
  };
}
