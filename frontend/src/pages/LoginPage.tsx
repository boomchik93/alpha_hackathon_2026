import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { ArrowRight, ShieldCheck, Sparkles } from 'lucide-react'
import { USE_MOCK } from '../api'
import { PROFILES, USERS, type SeedUser } from '../api/fixtures'
import type { ApiError } from '../api/types'
import { useSession } from '../auth/session'
import { useTheme } from '../components/Shell'
import { ErrorBanner, Eyebrow, Logo } from '../components/ui'
import { ROLE_LABEL, companyLabel } from '../dict/labels'

const accessLabel = (u: SeedUser) => (u.access === 'user' ? ROLE_LABEL[u.role] : u.access === 'expert' ? 'Эксперт банка' : 'Администратор')
const accountLine = (u: SeedUser) => (u.access === 'user' ? `${companyLabel(PROFILES[u.companies[0]])}${PROFILES[u.companies[0]]?.data_quality.missing_fields.length ? ' · данные с пропусками' : ''}` : u.access === 'expert' ? 'все компании, оценки' : 'все компании, мониторинг')

export default function LoginPage() {
  useTheme()
  const s = useSession()
  const nav = useNavigate()
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<ApiError | null>(null)

  if (s.ready && s.user && !busy) return <Navigate to="/" replace />

  async function enter(l = login, p = password) {
    setBusy(true); setErr(null)
    try { await s.signIn(l, p); nav('/') } catch (e) { setErr(e as ApiError) } finally { setBusy(false) }
  }

  return <div className="login-page">
    <div className="login-brand">
      <Logo />
      <div className="login-visual">
        <div className="signal-word">СИГНАЛ</div>
        <p>События рынка.<br /><b>В контексте вашего бизнеса.</b></p>
        <ul className="login-points">
          <li><ShieldCheck size={16} /> Позиция банка всегда отделена от выводов модели</li>
          <li><ShieldCheck size={16} /> Каждая цифра сверяется с источником</li>
          <li><ShieldCheck size={16} /> Сценарии: что будет при разных решениях</li>
        </ul>
        <div className="login-rings"><i /><i /><i /></div>
      </div>
      <small>Демо-продукт · Альфа Хак 2026</small>
    </div>
    <div className="login-panel">
      <form className="login-form" onSubmit={(e: FormEvent) => { e.preventDefault(); enter() }}>
        <Eyebrow>{USE_MOCK ? 'Демо-доступ' : 'Вход'}</Eyebrow>
        <h1>Войти в Альфа Бизнес</h1>
        <p>Лента и карточки подстраиваются под компанию и вашу роль.</p>
        <label className="login-field" htmlFor="login-input">Логин<input id="login-input" name="username" value={login} onChange={e => setLogin(e.target.value)} autoComplete="username" spellCheck={false} /></label>
        <label className="login-field" htmlFor="password-input">Пароль<input id="password-input" name="password" type="password" value={password} onChange={e => setPassword(e.target.value)} autoComplete="current-password" spellCheck={false} /></label>
        {err && <ErrorBanner error={err} />}
        <button className="primary-button" disabled={busy || !login.trim() || !password}>{busy ? 'Входим…' : <>Войти <ArrowRight /></>}</button>
        <>
          <div className="login-divider"><span>тестовые аккаунты для жюри</span></div>
          <div className="account-list">
            {USERS.map(u => (
              <button type="button" key={u.login} className={`account-option ${login === u.login ? 'selected' : ''}`} onClick={() => { setLogin(u.login); setPassword(u.password); enter(u.login, u.password) }}>
                <span className={`account-avatar ${u.access !== 'user' ? 'special' : ''}`}>{u.display_name.split(/[\s,]+/).slice(0, 2).map(x => x[0].toUpperCase()).join('')}</span>
                <span className="account-text"><b>{accessLabel(u)}</b><small>{accountLine(u)}</small></span>
                <ArrowRight size={16} />
              </button>
            ))}
          </div>
          <div className="demo-note"><Sparkles /><div><b>{USE_MOCK ? 'Демо без сервера' : 'Демо-аккаунты'}</b><p>Компании обезличены, новости — тестовый корпус. Демо-профили подготовлены заранее.</p></div></div>
        </>
      </form>
    </div>
  </div>
}
