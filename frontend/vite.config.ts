import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// `npm run build:demo` собирает всё в один HTML-файл: его можно открыть без сервера или выложить на Netlify.
export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === 'demo' ? [viteSingleFile()] : [])],
  server: {
    port: 5173,
    // В разработке /api уходит на бэкенд. Адрес меняется переменной VITE_API_PROXY.
    proxy: { '/api': process.env.VITE_API_PROXY ?? 'http://localhost:8000' },
  },
}))
