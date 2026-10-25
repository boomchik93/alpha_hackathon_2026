// Токен сессии. localStorage может быть недоступен (приватный режим) — тогда держим в памяти до перезагрузки.
let memory: string | null = null
const KEY = 'alfa-token'

export const token = {
  get(): string | null { try { return localStorage.getItem(KEY) ?? memory } catch { return memory } },
  set(v: string) { memory = v; try { localStorage.setItem(KEY, v) } catch { /* ignore */ } },
  clear() { memory = null; try { localStorage.removeItem(KEY) } catch { /* ignore */ } },
}
