import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/run': 'http://localhost:8000',
      '/results': 'http://localhost:8000',
      '/upload-batch': 'http://localhost:8000',
    },
  },
})
