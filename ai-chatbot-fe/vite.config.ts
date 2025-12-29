import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'dist/stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
    rollupOptions: {
      output: {
        manualChunks: id => {
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('react-dom')) {
              return 'react-vendor';
            }
            if (id.includes('react-router')) {
              return 'router';
            }
            if (id.includes('@tanstack/react-query')) {
              return 'query';
            }
            if (id.includes('zustand')) {
              return 'state';
            }
            if (id.includes('axios')) {
              return 'http';
            }
            return 'vendor';
          }

          // Split components into logical chunks
          if (id.includes('/components/Auth/')) {
            return 'auth';
          }
          if (
            id.includes('/components/Dashboard/') ||
            id.includes('/components/Documents/') ||
            id.includes('/components/Chat/')
          ) {
            return 'dashboard';
          }
          if (id.includes('/hooks/') || id.includes('/stores/')) {
            return 'utils';
          }
        },
      },
    },
    chunkSizeWarningLimit: 600,
    sourcemap: false,
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@tanstack/react-query',
      'zustand',
      'axios',
    ],
  },
});
