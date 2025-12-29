import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { useChatHistory, useDocuments } from '../../hooks/queries';
import { StreamStatus, useStreamingChat } from '../../lib/streaming-client';
import { useDocumentStore } from '../../stores/documentStore';
import MessageBubble from './MessageBubble';

// Unique ID for tracking stream sessions to prevent race conditions
let streamSessionId = 0;

// Token batching for smoother UI updates (prevents render thrashing)
const useTokenBatcher = () => {
  const pendingTokens = useRef('');
  const rafId = useRef<number | null>(null);
  const setterRef = useRef<((updater: (prev: string) => string) => void) | null>(null);

  const flush = useCallback(() => {
    if (pendingTokens.current && setterRef.current) {
      const tokens = pendingTokens.current;
      pendingTokens.current = '';
      setterRef.current(prev => prev + tokens);
    }
    rafId.current = null;
  }, []);

  const addToken = useCallback((token: string, setter: (updater: (prev: string) => string) => void) => {
    setterRef.current = setter;
    pendingTokens.current += token;
    
    if (rafId.current === null) {
      rafId.current = requestAnimationFrame(flush);
    }
  }, [flush]);

  const reset = useCallback(() => {
    if (rafId.current !== null) {
      cancelAnimationFrame(rafId.current);
      rafId.current = null;
    }
    pendingTokens.current = '';
  }, []);

  return { addToken, reset };
};

const ChatInterface: React.FC = () => {
  const [message, setMessage] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<StreamStatus | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const currentSessionRef = useRef<number>(0);

  const { selectedDocuments } = useDocumentStore();
  const { data: documents = [] } = useDocuments();
  const { data: chatHistory, isLoading: historyLoading, error, refetch } = useChatHistory();
  const tokenBatcher = useTokenBatcher();

  // Get auth token from localStorage
  const token = localStorage.getItem('token') || '';
  
  // Initialize streaming client
  const { streamMessage, isStreaming } = useStreamingChat({
    apiUrl: `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/chat/stream`,
    token
  });

  const messages = useMemo(
    () => chatHistory?.messages?.slice().reverse() || [],
    [chatHistory?.messages]
  );

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingResponse, pendingMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim() || isStreaming) return;

    const userMessage = message.trim();
    
    // Track this stream session to prevent race conditions
    streamSessionId++;
    const thisSession = streamSessionId;
    currentSessionRef.current = thisSession;
    
    setMessage('');
    setPendingMessage(userMessage);
    setStreamingResponse('');
    setStreamingStatus(StreamStatus.STARTING);
    setStreamError(null);
    tokenBatcher.reset();

    try {
      await streamMessage(
        userMessage,
        {
          onToken: (token) => {
            // Only update if this is still the current session
            if (currentSessionRef.current !== thisSession) return;
            // Use batched updates for smoother rendering
            tokenBatcher.addToken(token, setStreamingResponse);
            setStreamError(null);
          },
          onStatus: (status, statusMessage) => {
            if (currentSessionRef.current !== thisSession) return;
            setStreamingStatus(status);
            console.log(`Status: ${status} - ${statusMessage}`);
          },
          onProgress: (progress) => {
            console.log('Progress:', progress);
          },
          onComplete: async (data) => {
            // Verify this is still the active session before updating state
            if (currentSessionRef.current !== thisSession) {
              console.log('Ignoring stale stream completion');
              return;
            }
            
            console.log('Stream complete:', data);
            
            // Wait for history to refresh BEFORE clearing streaming state
            await refetch();
            
            // Double-check session is still valid after async operation
            if (currentSessionRef.current !== thisSession) return;
            
            // Clear state atomically
            setStreamingStatus(null);
            setStreamingResponse('');
            setPendingMessage(null);
            setStreamError(null);
          },
          onError: (error) => {
            if (currentSessionRef.current !== thisSession) return;
            
            console.error('Stream error:', error);
            setStreamingStatus(StreamStatus.ERROR);
            setStreamError(error.message || 'An error occurred while streaming');
            setStreamingResponse('');
            setPendingMessage(null);
            
            // Clear error after 5 seconds (only if still current session)
            setTimeout(() => {
              if (currentSessionRef.current !== thisSession) return;
              setStreamingStatus(null);
              setStreamError(null);
            }, 5000);
          }
        },
        selectedDocuments
      );
    } catch (error: unknown) {
      if (currentSessionRef.current !== thisSession) return;
      
      const errorMessage = error instanceof Error ? error.message : 'Failed to connect to streaming service';
      console.error('Failed to stream message:', error);
      setStreamingStatus(StreamStatus.ERROR);
      setStreamError(errorMessage);
      setStreamingResponse('');
      setPendingMessage(null);
      
      // Clear error after 5 seconds
      setTimeout(() => {
        if (currentSessionRef.current !== thisSession) return;
        setStreamingStatus(null);
        setStreamError(null);
      }, 5000);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as React.FormEvent);
    }
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      // On mobile, keep fixed height (single line)
      if (window.innerWidth < 768) {
        textareaRef.current.style.height = '44px';
        return;
      }

      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  const getSelectedDocumentNames = () => {
    return selectedDocuments
      .map(id => documents.find(doc => doc.id === id)?.filename)
      .filter(Boolean);
  };

  const getStatusMessage = (status: StreamStatus | null) => {
    switch (status) {
      case StreamStatus.STARTING:
        return 'Starting...';
      case StreamStatus.RETRIEVING:
        return 'Retrieving documents...';
      case StreamStatus.GENERATING:
        return 'Generating response...';
      case StreamStatus.ERROR:
        return 'Error occurred';
      default:
        return 'AI is thinking...';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-gray-200 p-4 bg-white">
        <h2 className="text-xl font-semibold text-gray-800">AI Assistant</h2>
        {selectedDocuments.length > 0 && (
          <p className="text-sm text-gray-600 mt-1">
            Chatting with: {getSelectedDocumentNames().join(', ')}
          </p>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {historyLoading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
            <span className="ml-3 text-gray-600">Loading chat history...</span>
          </div>
        ) : messages.length === 0 && !streamingResponse ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🤖</div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">Start a conversation</h3>
            <p className="text-gray-500">
              {selectedDocuments.length > 0
                ? 'Ask questions about your selected documents'
                : 'Upload and select documents to chat about them, or ask general questions'}
            </p>
          </div>
        ) : (
          <>
            {messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            
            {/* Pending User Message - Show immediately while streaming */}
            {pendingMessage && (
               <div className="flex justify-end space-y-4">
                <div className="max-w-[85%] md:max-w-3xl">
                  <div className="bg-primary-500 text-white rounded-2xl rounded-tr-md px-3 py-2 md:px-4 md:py-3">
                    <p className="whitespace-pre-wrap break-words text-sm md:text-base">
                      {pendingMessage}
                    </p>
                  </div>
                  <div className="text-xs text-gray-500 text-right mt-1">
                    Sending...
                  </div>
                </div>
              </div>
            )}
            
            {/* Streaming response bubble - Matched to MessageBubble style */}
            {streamingResponse && (
               <div className="flex justify-start">
                <div className="max-w-[90%] md:max-w-3xl">
                  <div className="flex items-start space-x-2 md:space-x-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-sm md:text-base">
                      🤖
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="bg-gray-100 rounded-2xl rounded-tl-md px-3 py-2 md:px-4 md:py-3">
                        <p className="whitespace-pre-wrap break-words text-gray-800 text-sm md:text-base">
                          {streamingResponse}
                          <span className="inline-block w-2 h-4 bg-primary-500 ml-1 animate-pulse align-middle"></span>
                        </p>
                      </div>
                      {streamingStatus && (
                         <div className="text-xs text-gray-500 mt-1">
                           {getStatusMessage(streamingStatus)}
                         </div>
                       )}
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Loading indicator when no streaming response yet */}
            {isStreaming && !streamingResponse && (
              <div className="flex items-center space-x-2 text-gray-500">
                <div className="flex space-x-1">
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0ms' }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: '150ms' }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: '300ms' }}
                  ></div>
                </div>
                <span className="text-sm">{getStatusMessage(streamingStatus)}</span>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error Display */}
      {(error || streamError) && (
        <div className="shrink-0 p-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start">
              <svg className="w-5 h-5 text-red-500 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">
                  {streamError || error?.message || 'An unexpected error occurred'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="flex-shrink-0 border-t border-gray-200 p-3 md:p-4 bg-white">
        <form onSubmit={handleSubmit} className="flex space-x-2 md:space-x-4">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={e => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                selectedDocuments.length > 0 ? 'Ask a question...' : 'Type your message...'
              }
              className="w-full px-3 py-2 md:px-4 md:py-3 text-sm md:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
              rows={1}
              disabled={isStreaming}
              style={{ minHeight: '44px' }}
            />
          </div>
          <button
            type="submit"
            disabled={!message.trim() || isStreaming}
            className="px-4 py-2 md:px-6 md:py-3 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
          >
            {isStreaming ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <>
                <span className="hidden md:inline">Send</span>
                <span className="md:hidden">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                </span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
