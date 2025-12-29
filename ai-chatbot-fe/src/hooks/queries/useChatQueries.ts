import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { chatAPI } from '../../services/api';
import type { ChatMessage } from '../../types';

// Query Keys
export const chatKeys = {
  all: ['chat'] as const,
  history: (limit?: number) => [...chatKeys.all, 'history', { limit }] as const,
};

// Queries
export const useChatHistory = (limit: number = 50) => {
  return useQuery({
    queryKey: chatKeys.history(limit),
    queryFn: () => chatAPI.getHistory(limit),
    staleTime: 1 * 60 * 1000, // 1 minute
  });
};

// Mutations
export const useSendMessage = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (message: ChatMessage) => chatAPI.sendMessage(message),
    onSuccess: () => {
      // Invalidate and refetch chat history
      queryClient.invalidateQueries({ queryKey: chatKeys.all });
    },
    onError: error => {
      console.error('Send message failed:', error);
    },
  });
};

export const useDeleteMessage = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (messageId: string) => chatAPI.deleteMessage(messageId),
    onSuccess: () => {
      // Invalidate and refetch chat history
      queryClient.invalidateQueries({ queryKey: chatKeys.all });
    },
    onError: error => {
      console.error('Delete message failed:', error);
    },
  });
};

export const useClearChatHistory = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => chatAPI.clearHistory(),
    onSuccess: () => {
      // Clear all chat-related queries
      queryClient.removeQueries({ queryKey: chatKeys.all });
    },
    onError: error => {
      console.error('Clear chat history failed:', error);
    },
  });
};
