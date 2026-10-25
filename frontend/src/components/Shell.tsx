import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { ArrowRight, Bell, Building2, CircleAlert, Gauge, HelpCircle, Home, MessageSquareText, Moon, PanelLeftClose, Settings, Sparkles, Sun, TrendingDown } from 'lucide-react'
import { USE_MOCK } from '../api'
import { useCompanies, useCompany, useFeed, useSummary } from '../api/hooks'
import type { Role } from '../api/types'
import { useSession } from '../auth/session'
import { ROLE_LABEL, companyLabel, fmtDate, plural } from '../dict/labels'
import { Dropdown, Eyebrow, Logo, store } from './ui'

const ROLES = (Object.keys(ROLE_LABEL) as Role[]).filter(r => r !== 'other')

function initialTheme(): 'light' | 'dark' {
  const saved = store.get('alfa-theme')
  if (saved === 'light' || saved === 'dark') return saved
  try { return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' } catch { return 'light' }
}

export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(initialTheme)
  useEffect(() => { document.documentElement.dataset.theme = theme; store.set('alfa-theme', theme) }, [theme])
  return [theme, setTheme] as const
}

export function Shell({ children }: { children: ReactNode }) {
  const s = useSession()
  const nav = useNavigate()
  const loc = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const [notificationsOpen, setNotificationsOpen] = useState(false)
  const [theme, setTheme] = useTheme()
  const actionsRef = useRef<HTMLDivElement>(null)
  const companies = useCompanies(s.user?.login)
  const company = useCompany(s.companyId)
  const summary = useSummary(s.companyId)
  const feed = useFeed(s.companyId, { tier: 'main' })
  const newCount = summary.data?.new_count ?? 0

  useEffect(() => { window.scrollTo(0, 0); setNotificationsOpen(false); setProfileOpen(false) }, [loc.pathname])
  useEffect(() => {
    if (!notificationsOpen && !profileOpen) return
    const close = (event: MouseEvent) => { if (!actionsRef.current?.contains(event.target as Node)) { setNotificationsOpen(false); setProfileOpen(false) } }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [notificationsOpen, profileOpen])

  const items = [
    { to: '/', icon: Home, label: 'Главная' },
    { to: '/feed', icon: Sparkles, label: 'События для бизнеса', count: newCount },
    { to: '/profile', icon: Building2, label: 'Профиль компании' },
    { to: '/history', icon: MessageSquareText, label: 'История вопросов' },
    ...(s.user && s.user.access !== 'user' ? [{ to: '/monitoring', icon: Gauge, label: 'Мониторинг' }] : []),
  ]
  const isFeed = (to: string) => to === '/feed' && (loc.pathname.startsWith('/insight') || loc.pathname.startsWith('/trace') || loc.pathname.startsWith('/source'))
  const canSwitchRole = s.user?.access !== 'user'
  const initials = (s.user?.display_name ?? '').split(/[\s,]+/).filter(Boolean).slice(0, 2).map(x => x[0].toUpperCase()).join('')
  const top = feed.data?.entries ?? []
  const dataWarn = company.data?.data_quality.notes[0] ?? (company.data?.data_quality.missing_fields.length ? 'Часть полей профиля не заполнена: выводы будут осторожнее' : undefined)

  return <div className={`app-shell ${collapsed ? 'is-collapsed' : ''}`}>
    <aside className="sidebar">
      <div className="side-top"><Logo compact={collapsed} /><button className="icon-button collapse" onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? 'Развернуть меню' : 'Свернуть меню'}><PanelLeftClose size={19} /></button></div>
      <nav>
        {items.map(({ to, icon: Icon, label, count }) => <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav-item ${isActive || isFeed(to) ? 'active' : ''}`} title={label}>
          <Icon size={20} /><span>{label}</span>{!!count && <b className="nav-count">{count}</b>}
        </NavLink>)}
      </nav>
      <div className="side-bottom">
        {USE_MOCK && <div className="demo-flag"><span>Демо · тестовые данные</span></div>}
        <span className="nav-item muted-item" title="В прототипе не реализовано"><HelpCircle size={20} /><span>Помощь</span></span>
        <span className="nav-item muted-item" title="В прототипе не реализовано"><Settings size={20} /><span>Настройки</span></span>
      </div>
    </aside>
    <div className="app-main">
      <header className="topbar">
        <div className="mobile-logo"><Logo /></div>
        <div className="context-selectors">
          <div className="select-field"><span className="select-label">Компания</span><Dropdown ariaLabel="Компания" value={s.companyId} onChange={s.setCompanyId} options={(companies.data ?? []).map(c => ({ value: c.company_id, label: companyLabel(c) }))} compact /></div>
          {canSwitchRole
            ? <div className="select-field" title="Для демо: посмотреть карточку глазами другой роли"><span className="select-label">Смотреть как</span><Dropdown ariaLabel="Смотреть как" value={s.role} onChange={v => s.setRole(v as Role)} options={ROLES.map(r => ({ value: r, label: ROLE_LABEL[r] }))} compact /></div>
            : <div className="select-field"><span className="select-label">Моя роль</span><span className="role-static">{ROLE_LABEL[s.role]}</span></div>}
        </div>
        <div className="top-actions" ref={actionsRef}>
          <button className="icon-button theme-toggle" aria-label={theme === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему'} title={theme === 'dark' ? 'Светлая тема' : 'Тёмная тема'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={19} /> : <Moon size={19} />}</button>
          <button className={`icon-button ${newCount ? 'has-dot' : ''} ${notificationsOpen ? 'active' : ''}`} aria-label="Уведомления" aria-expanded={notificationsOpen} onClick={() => { setNotificationsOpen(!notificationsOpen); setProfileOpen(false) }}><Bell size={20} /></button>
          <button className="avatar" aria-label="Профиль пользователя" aria-expanded={profileOpen} onClick={() => { setProfileOpen(!profileOpen); setNotificationsOpen(false) }}>{initials}</button>
          {notificationsOpen && <section className="notifications-popover" aria-label="Уведомления">
            <div className="notifications-head"><div><Eyebrow>Центр событий</Eyebrow><h3>Уведомления</h3></div>{newCount > 0 && <span>{newCount} {plural(newCount, 'новое', 'новых', 'новых')}</span>}</div>
            <div className="notifications-list">
              {top.slice(0, 2).map((e, i) => <Link key={e.item_id} to={`/insight/${e.item_id}`} className={`notification-item ${e.is_new ? 'unread' : ''}`}>
                <span className={`notification-icon ${i === 0 ? 'critical' : ''}`}>{i === 0 ? <TrendingDown /> : <Sparkles />}</span>
                <div><b>{e.title}</b><p>Почему вам: {e.reasons[0]}</p><small>{fmtDate(e.published_at)}</small></div>{e.is_new ? <i /> : <span />}
              </Link>)}
              {dataWarn && <Link to="/profile" className="notification-item"><span className="notification-icon profile"><CircleAlert /></span><div><b>В профиле компании есть пробелы</b><p>{dataWarn}</p><small>Профиль компании</small></div><span /></Link>}
            </div>
            <Link to="/feed" className="notifications-footer">Перейти ко всем событиям <ArrowRight size={16} /></Link>
          </section>}
          {profileOpen && <div className="profile-popover">
            <strong>{s.user?.display_name}</strong>
            <span>{s.user?.access === 'user' ? ROLE_LABEL[s.user.role] : s.user?.access === 'expert' ? 'Эксперт банка' : 'Администратор'}</span>
            <span>{company.data ? companyLabel(company.data) : ''}</span>
            <hr />
            <button onClick={() => { s.signOut().then(() => nav('/login')) }}>Выйти</button>
          </div>}
        </div>
      </header>
      <main>{children}</main>
    </div>
    <div className="mobile-nav">{items.slice(0, 4).map(({ to, icon: Icon, label }) => <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive || isFeed(to) ? 'active' : '')}><Icon size={20} /><span>{label.split(' ')[0]}</span></NavLink>)}</div>
  </div>
}
