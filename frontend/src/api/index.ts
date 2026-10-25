import type { Api } from './api'
import { httpApi } from './http'
import { mockApi } from './mock'

// По умолчанию работаем на моках. VITE_USE_MOCK=0 — настоящий бэкенд (docker compose и npm run dev с прокси на :8000).
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== '0'
export const api: Api = USE_MOCK ? mockApi : httpApi
