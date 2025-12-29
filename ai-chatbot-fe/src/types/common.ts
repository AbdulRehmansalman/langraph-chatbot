export interface LoadingState {
  loading: boolean;
  error: string | null;
}

export interface FormState<T> extends LoadingState {
  data: T;
  isDirty: boolean;
  isValid: boolean;
  touched: Record<keyof T, boolean>;
  errors: Partial<Record<keyof T, string>>;
}

export interface TableState<T> {
  data: T[];
  loading: boolean;
  error: string | null;
  pagination: {
    page: number;
    limit: number;
    total: number;
  };
  sorting: {
    field: keyof T | null;
    direction: 'asc' | 'desc';
  };
  filters: Record<string, unknown>;
}

export type AsyncActionStatus = 'idle' | 'pending' | 'fulfilled' | 'rejected';

export interface AsyncAction<T = unknown> {
  status: AsyncActionStatus;
  data: T | null;
  error: string | null;
}

export type Theme = 'light' | 'dark' | 'system';

export interface NotificationOptions {
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
  persistent?: boolean;
}

export interface ModalState {
  isOpen: boolean;
  title?: string;
  content?: React.ReactNode;
  onClose?: () => void;
  onConfirm?: () => void;
}
