import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api } from '../api'
import { token } from '../api/token'
import type { Me, Role } from '../api/types'

interface Session {
  user: Me | null
  ready: boolean // проверили сохранённый токен
  companyId: string
  role: Role // для эксперта и админа — «смотреть как»; для пользователя — его роль
  signIn(login: string, password: string): Promise<void>
  signOut(): Promise<void>
  setCompanyId(id: string): void
  setRole(r: Role): void
}

const Ctx = createContext<Session | null>(null)
const store = {
  get(k: string) { try { return localStorage.getItem(k) } catch { return null } },
  set(k: string, v: string | null) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v) } catch { /* ignore */ } },
}
const defaultRole = (u: Me): Role => (u.role === 'other' ? 'cfo' : u.role)

export function SessionProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Me | null>(null)
  const [ready, setReady] = useState(false)
  const [companyId, setCompanyIdState] = useState(store.get('alfa-company') ?? '')
  const [role, setRole] = useState<Role>('cfo')

  function apply(u: Me | null, keepCompany = false) {
    setUser(u)
    if (!u) return
    setRole(defaultRole(u))
    const saved = store.get('alfa-company')
    const first = u.companies[0] ?? saved ?? ''
    const next = keepCompany && saved && (u.access !== 'user' || u.companies.includes(saved)) ? saved : first
    setCompanyIdState(next); store.set('alfa-company', next)
  }

  // Восстанавливаем сессию по сохранённому токену
  useEffect(() => {
    if (!token.get()) { setReady(true); return }
    api.me().then(u => apply(u, true)).catch(() => token.clear()).finally(() => setReady(true))
  }, [])

  // Эксперт и админ без привязанных компаний: берём первую доступную
  useEffect(() => {
    if (user && !companyId) api.companies().then(cs => { if (cs[0]) { setCompanyIdState(cs[0].company_id); store.set('alfa-company', cs[0].company_id) } }).catch(() => {})
  }, [user, companyId])

  const value: Session = {
    user, ready, companyId, role,
    async signIn(login, password) { const out = await api.login(login, password); store.set('alfa-company', null); apply(out.user) },
    async signOut() { await api.logout().catch(() => {}); setUser(null) },
    setCompanyId(id) { setCompanyIdState(id); store.set('alfa-company', id) },
    setRole,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useSession() {
  const s = useContext(Ctx)
  if (!s) throw new Error('SessionProvider missing')
  return s
}

// Роль для запросов: только эксперт и админ передают её явно (backend: effective_role).
export function useRequestRole(): Role | undefined {
  const s = useSession()
  return s.user && s.user.access !== 'user' ? s.role : undefined
}
