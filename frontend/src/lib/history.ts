// История уточняющих вопросов. Ручки истории в бэкенде нет, поэтому храним в браузере, отдельно для каждого логина.
import { USE_MOCK } from '../api'
import { HISTORY_SEED } from '../api/fixtures'
import type { HistoryItem } from '../api/types'

const key = (login: string) => `alfa-history:${login}`
const memory = new Map<string, HistoryItem[]>()

export function readHistory(login: string): HistoryItem[] {
  let list: HistoryItem[] | undefined = memory.get(login)
  if (!list) {
    try { const raw = localStorage.getItem(key(login)); if (raw) list = JSON.parse(raw) as HistoryItem[] } catch { /* ignore */ }
  }
  if (!list) list = USE_MOCK ? HISTORY_SEED.filter(h => h.login === login).map((h, i) => ({ ...h, id: `seed-${i}` })) : []
  memory.set(login, list)
  return [...list].sort((a, b) => b.asked_at.localeCompare(a.asked_at))
}

export function addHistory(item: Omit<HistoryItem, 'id' | 'asked_at'>) {
  const list = readHistory(item.login)
  const next = [{ ...item, id: `h-${Date.now()}`, asked_at: new Date().toISOString() }, ...list].slice(0, 200)
  memory.set(item.login, next)
  try { localStorage.setItem(key(item.login), JSON.stringify(next)) } catch { /* ignore */ }
}
