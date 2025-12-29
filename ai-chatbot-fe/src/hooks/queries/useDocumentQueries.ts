import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { documentsAPI } from '../../services/api';

// Query Keys
export const documentKeys = {
  all: ['documents'] as const,
  lists: () => [...documentKeys.all, 'list'] as const,
  list: (filters: string) => [...documentKeys.lists(), { filters }] as const,
  details: () => [...documentKeys.all, 'detail'] as const,
  detail: (id: string) => [...documentKeys.details(), id] as const,
};

// Queries
export const useDocuments = () => {
  return useQuery({
    queryKey: documentKeys.lists(),
    queryFn: documentsAPI.getAll,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useDocument = (documentId: string) => {
  return useQuery({
    queryKey: documentKeys.detail(documentId),
    queryFn: () => documentsAPI.getById(documentId),
    enabled: !!documentId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Mutations
export const useUploadDocument = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ file, onProgress }: { file: File; onProgress?: (percentage: number) => void }) =>
      documentsAPI.upload(file, onProgress),
    onSuccess: () => {
      // Invalidate and refetch documents list
      queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
    onError: error => {
      console.error('Document upload failed:', error);
    },
  });
};

export const useDeleteDocument = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (documentId: string) => documentsAPI.delete(documentId),
    onSuccess: () => {
      // Invalidate and refetch documents list
      queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
    onError: error => {
      console.error('Document deletion failed:', error);
    },
  });
};
