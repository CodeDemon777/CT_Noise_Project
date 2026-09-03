import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://127.0.0.1:8080',
      '/demo': 'http://127.0.0.1:8080',
      '/upload': 'http://127.0.0.1:8080',
      '/report': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
      '/static': 'http://127.0.0.1:8080',
    },
  },
});
