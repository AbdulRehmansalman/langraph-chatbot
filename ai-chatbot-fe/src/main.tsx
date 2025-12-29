import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import './index.css';
import App from './App.tsx';

// Prevent unhandled promise rejections from causing page refresh
window.addEventListener('unhandledrejection', event => {
  console.log('Unhandled promise rejection:', event.reason);
  event.preventDefault(); // Prevent the default behavior (page refresh)
});

// Prevent uncaught errors from causing page refresh
window.addEventListener('error', event => {
  console.log('Uncaught error:', event.error);
  event.preventDefault(); // Prevent the default behavior
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
      throwOnError: false, // Prevent errors from bubbling up
    },
    mutations: {
      retry: false,
      throwOnError: false, // Prevent errors from bubbling up
      onError: error => {
        console.log('Global mutation error:', error);
        // Don't let errors bubble up to cause page refresh
      },
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>
);
