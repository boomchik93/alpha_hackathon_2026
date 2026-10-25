// Клиент настоящего бэкенда (backend/app/routes.py). Все ручки под /api, вход по Bearer-токену.
import type { Api } from './api'
import type { ApiError, InsightCard, JobView, Ready, Role, ScenarioSet } from './types'
import { token } from './token'

const BASE = import.meta.env.VITE_API_URL ?? '/api'

async function req(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const t = token.get()
  if (t) headers.set('Authorization', `Bearer ${t}`)
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  let res: Response
  try {
    res = await fetch(BASE + path, { ...init, headers })
  } catch {
    throw { status: 0, code: 'network', message: 'Нет связи с сервером' } satisfies ApiError
  }
  if (!res.ok && res.status !== 202) {
    const body = (await res.json().catch(() => ({}))) as Partial<ApiError>
    if (res.status === 401) token.clear()
    throw { status: res.status, code: body.code ?? 'internal_error', message: body.message ?? res.statusText, retry_after: body.retry_after } satisfies ApiError
  }
  return res
}
const get = async <T,>(p: string) => (await (await req(p)).json()) as T
const post = async <T,>(p: string, b?: unknown) => (await (await req(p, { method: 'POST', body: b === undefined ? undefined : JSON.stringify(b) })).json()) as T
const q = (params: Record<string, string | number | boolean | undefined>) => {
  const s = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v !== undefined && v !== '' && v !== false && s.set(k, String(v)))
  const str = s.toString()
  return str ? `?${str}` : ''
}

// 200 — готовый объект, 202 — задача, которую опрашиваем по poll_url
async function readyOrJob<T>(path: string): Promise<Ready<T>> {
  const res = await req(path)
  const body = await res.json()
  return res.status === 202 ? { status: 'pending', job: body as JobView } : { status: 'ready', data: body as T }
}

export const httpApi: Api = {
  async login(login, password) {
    const out = await post<import('./types').LoginOut>('/auth/login', { login, password })
    token.set(out.token)
    return out
  },
  async logout() { try { await post('/auth/logout') } finally { token.clear() } },
  me: () => get('/me'),
  companies: () => get('/companies'),
  company: id => get(`/companies/${encodeURIComponent(id)}`),
  feed: (id, f = {}) => get(`/companies/${encodeURIComponent(id)}/feed${q({ ...f })}`),
  summary: id => get(`/me/summary${q({ company_id: id })}`),
  card: (id, item, role?: Role) => readyOrJob<InsightCard>(`/companies/${encodeURIComponent(id)}/insights/${encodeURIComponent(item)}${q({ role })}`),
  scenarios: (id, item, role?: Role) => readyOrJob<ScenarioSet>(`/companies/${encodeURIComponent(id)}/insights/${encodeURIComponent(item)}/scenarios${q({ role })}`),
  job: id => get(`/jobs/${encodeURIComponent(id)}`),
  followup: (cardId, question, role?: Role) => post(`/cards/${encodeURIComponent(cardId)}/followup${q({ role })}`, { question }),
  feedback: (cardId, body) => post(`/cards/${encodeURIComponent(cardId)}/feedback`, body),
  cardTrace: cardId => get(`/cards/${encodeURIComponent(cardId)}/trace`),
  // PDF требует токен, поэтому обычная ссылка не подойдёт: качаем blob и открываем его с #page=N.
  sourcePdf: async docId => (await req(`/sources/${encodeURIComponent(docId)}/pdf`)).blob(),
  health: () => get('/health'),
  feedbackSummary: () => get('/feedback/summary'),
  audit: (limit = 50) => get(`/admin/audit${q({ limit })}`),
}
