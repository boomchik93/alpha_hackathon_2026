import { useQuery } from '@tanstack/react-query'
import { CircleDollarSign, FileChartColumn, Gauge, ThumbsUp } from 'lucide-react'
import { api } from '../api'
import type { ApiError, AuditRow, FeedbackSummary, Health, Role, VoteStats } from '../api/types'
import { useSession } from '../auth/session'
import { ErrorBanner, Eyebrow, PageHeader, Pill, Skel } from '../components/ui'
import { REASON_LABEL, ROLE_LABEL, fmtDate, fmtTime } from '../dict/labels'

const rub = (v: number) => v.toFixed(1).replace('.', ',') + ' ₽'
const useful = (g?: VoteStats) => (g && g.total ? 1 - (g.down_share ?? 0) : null)
const pctOr = (v: number | null) => (v === null ? '—' : `${Math.round(v * 100)}%`)
const ACTION: Record<string, [string, string]> = {
  login: ['Вход', 'neutral'], login_failed: ['Неудачный вход', 'red'], insight: ['Карточка', 'success'], scenarios: ['Сценарии', 'success'],
  followup: ['Уточняющий вопрос', 'neutral'], feedback: ['Оценка', 'warning'], trace: ['Трасса', 'neutral'], source: ['Первоисточник', 'neutral'],
}
const actionLabel = (a: string) => ACTION[a] ?? [a, 'neutral']

// Мониторинг качества и расхода (К08, О04): /health, /feedback/summary, /admin/audit. Для эксперта и администратора.
export default function MonitoringPage() {
  const s = useSession()
  const allowed = !!s.user && s.user.access !== 'user'
  const health = useQuery({ queryKey: ['health'], queryFn: () => api.health(), refetchInterval: 15_000, staleTime: 0 })
  const fb = useQuery({ queryKey: ['feedback-summary'], queryFn: () => api.feedbackSummary(), enabled: allowed, refetchInterval: 15_000, staleTime: 0 })
  const audit = useQuery({ queryKey: ['audit'], queryFn: () => api.audit(50), enabled: s.user?.access === 'admin', refetchInterval: 15_000, staleTime: 0 })
  const err = (fb.error ?? audit.error ?? health.error) as unknown as ApiError | null
  const isAdmin = s.user?.access === 'admin'
  return <div className="page">
    <PageHeader eyebrow="Состояние сервиса" title="Мониторинг качества" subtitle="Здоровье ML-слоя, расход на модели и обратная связь. Обновляется каждые 15 секунд." />
    {!allowed && <ErrorBanner error={{ status: 403, code: 'forbidden', message: 'Мониторинг доступен эксперту и администратору' }} />}
    {err && <ErrorBanner error={err} onRetry={() => { health.refetch(); fb.refetch(); audit.refetch() }} />}
    {allowed && (!health.data || !fb.data) && !err && <div className="metrics-grid">{[0, 1, 2, 3].map(i => <section key={i} className="surface metric"><Skel w="50%" /><Skel w="40%" h={26} /></section>)}</div>}
    {allowed && health.data && fb.data && <Dashboard h={health.data} f={fb.data} rows={audit.data ?? []} showAudit={isAdmin} />}
  </div>
}

function Dashboard({ h, f, rows, showAudit }: { h: Health; f: FeedbackSummary; rows: AuditRow[]; showAudit: boolean }) {
  const spent = h.spent_today_rub ?? 0
  const budget = h.budget_rub ?? 500
  const spend = spent / budget
  const all = f.ml?.['все'] ?? (f.user && f.expert ? { total: f.user.total + f.expert.total, down: f.user.down + f.expert.down, down_share: (f.user.down + f.expert.down) / Math.max(1, f.user.total + f.expert.total), reasons: {} } : undefined)
  const reasons = Object.entries(f.user?.reasons ?? {}).concat(Object.entries(f.expert?.reasons ?? {}))
    .reduce<Record<string, number>>((m, [k, v]) => ({ ...m, [k]: (m[k] ?? 0) + v }), {})
  const maxReason = Math.max(1, ...Object.values(reasons))
  const generated = rows.filter(r => r.action === 'insight').length
  const asked = rows.filter(r => r.action === 'followup').length
  const scen = rows.filter(r => r.action === 'scenarios').length
  const uU = useful(f.user), uE = useful(f.expert)
  const versions = Object.entries(f.ml ?? {}).filter(([k]) => k !== 'все')
  return <>
    <div className="metrics-grid">
      <section className={`surface metric ${h.ml_ok ? 'green' : 'red'}`}><div><Gauge /><span>ML-сервис</span></div><b>{h.ml_ok ? 'Работает' : 'Недоступен'}</b><small>{h.ml_ok ? (h.llm_mock ? 'модель в режиме заглушки' : 'модель подключена') : h.ml_error ?? 'нет ответа'}</small></section>
      <section className={`surface metric ${spend > 0.85 ? 'red' : 'dark'}`}><div><CircleDollarSign /><span>Расход сегодня</span></div><b>{rub(spent)}</b><small>из лимита {budget} ₽ · {Math.round(spend * 100)}%</small><div className="budget-bar"><i style={{ width: `${Math.min(100, spend * 100)}%` }} className={spend > 0.85 ? 'hot' : spend > 0.6 ? 'warm' : ''} /></div></section>
      <section className="surface metric green"><div><ThumbsUp /><span>Полезные карточки</span></div><b>{pctOr(useful(all))}</b><small>{all?.total ?? 0} оценок всего</small></section>
      {showAudit ? <section className="surface metric dark"><div><FileChartColumn /><span>Открытий карточек</span></div><b>{generated}</b><small>{scen} запросов сценариев · {asked} вопросов</small></section>
        : <section className="surface metric dark"><div><FileChartColumn /><span>Оценок экспертов</span></div><b>{f.expert?.total ?? 0}</b><small>журнал обращений доступен администратору</small></section>}
    </div>
    <div className="monitor-grid">
      <section className="surface issues-card">
        <div className="section-title"><div><Eyebrow>Оценки</Eyebrow><h3>Пользователи и эксперты</h3></div></div>
        {([['Пользователи', f.user], ['Эксперты банка', f.expert]] as const).map(([l, g]) => <div className="bar-row" key={l}><span>{l}</span><div><i className="good" style={{ width: `${(useful(g) ?? 0) * 100}%` }} /></div><b>{pctOr(useful(g))}</b></div>)}
        <div className="vote-split">
          {([['Пользователи', f.user], ['Эксперты', f.expert]] as const).map(([l, g]) => <div key={l}><span>{l}</span><b>{(g?.total ?? 0) - (g?.down ?? 0)}<small> полезно</small></b><b className="down">{g?.down ?? 0}<small> не полезно</small></b></div>)}
        </div>
        {uU !== null && uE !== null && <div className="gap-note"><b>{Math.round(Math.abs(uU - uE) * 100)} п. п.</b><span>расхождение оценок экспертов и пользователей. Экспертные оценки считаются отдельно: если эксперты заметно строже, стоит проверить качество фактов.</span></div>}
      </section>
      <div className="profile-col">
      <section className="surface quality-chart">
        <div className="section-title"><div><Eyebrow>Обратная связь</Eyebrow><h3>«Не полезно» по ролям</h3></div></div>
        {Object.entries(f.user?.by_role ?? {}).map(([r, g]) => <div className="bar-row" key={r}><span>{ROLE_LABEL[r as Role] ?? r}</span><div><i style={{ width: `${(g.down_share ?? 0) * 100}%` }} /></div><b>{pctOr(g.down_share)}</b></div>)}
        {!Object.keys(f.user?.by_role ?? {}).length && <p className="muted">Оценок пользователей пока нет.</p>}
      </section>
      <section className="surface issues-card">
        <div className="section-title"><div><Eyebrow>Причины</Eyebrow><h3>Почему «не полезно»</h3></div></div>
        {Object.entries(reasons).map(([r, n]) => <div className="issue-row" key={r}><span><i style={{ width: `${(n / maxReason) * 100}%` }} /></span><b>{REASON_LABEL[r as keyof typeof REASON_LABEL] ?? r}</b><em>{n}</em></div>)}
        {!Object.keys(reasons).length && <p className="muted">Отрицательных оценок нет.</p>}
        {versions.length > 0 && <div className="prompt-versions">{versions.map(([v, g]) => <div key={v}><b>Промпт {v}</b><span>{g.total} оценок · не полезно {pctOr(g.down_share)}</span></div>)}</div>}
        {!f.ml && <p className="muted">ML-сервис не вернул статистику по версиям промпта.</p>}
      </section>
      </div>
    </div>
    <div className="monitor-grid">
      <Usage f={f} />
      <ItemsByDay days={f.items_by_day} />
    </div>
    {showAudit && <section className="surface traces-card">
      <div className="section-title"><div><Eyebrow>Журнал</Eyebrow><h3>Последние обращения</h3></div><Pill>{rows.length}</Pill></div>
      <div className="traces-table">
        <div className="traces-row head"><span>Время</span><span>Что</span><span>Компания</span><span>Действие</span><span>Польз.</span></div>
        {rows.map(r => { const [label, tone] = actionLabel(r.action); return <div className="traces-row" key={r.id}>
          <span>{fmtTime(r.ts)}<small className="muted"> {fmtDate(r.ts, true)}</small></span>
          <span><b>{r.item_id ?? r.detail ?? '—'}</b><small>{r.trace_id ?? r.card_id ?? ''}</small></span>
          <span>{r.company_id ?? '—'}</span>
          <span><Pill tone={tone}>{label}</Pill></span>
          <span>{r.user_id ?? '—'}</span>
        </div> })}
      </div>
    </section>}
  </>
}

// Что пользователи делают после открытия карточки: вопрос, первоисточник, трасса, оценка (метрика из product/metrics.md).
const USAGE_ROWS: [string, string][] = [['insight', 'Открыли карточку'], ['followup', 'Задали вопрос'], ['source', 'Открыли первоисточник'], ['trace', 'Смотрели трассу'], ['feedback', 'Оценили']]

function Usage({ f }: { f: FeedbackSummary }) {
  const a = f.usage?.actions ?? {}
  const max = Math.max(1, ...USAGE_ROWS.map(([k]) => a[k]?.count ?? 0))
  return <section className="surface quality-chart">
    <div className="section-title"><div><Eyebrow>Вовлечённость · {f.usage?.days ?? 7} дней</Eyebrow><h3>Что делают после карточки</h3></div>
      {f.usage?.actions_per_card != null && <Pill tone="success">{String(f.usage.actions_per_card).replace('.', ',')} действия на карточку</Pill>}</div>
    {USAGE_ROWS.map(([k, l]) => <div className="bar-row" key={k}><span>{l}</span><div><i className="good" style={{ width: `${((a[k]?.count ?? 0) / max) * 100}%` }} /></div><b>{a[k]?.count ?? 0}</b></div>)}
    {!f.usage && <p className="muted">Бэкенд не вернул журнал обращений.</p>}
  </section>
}

// Поступление материалов по дням публикации: день без новых материалов означает остановку сборщика.
function ItemsByDay({ days }: { days?: Record<string, number> | null }) {
  const rows = Object.entries(days ?? {})
  const max = Math.max(1, ...rows.map(([, n]) => n))
  const empty = rows.filter(([, n]) => n === 0).length
  return <section className="surface quality-chart">
    <div className="section-title"><div><Eyebrow>Источники</Eyebrow><h3>Новые материалы по дням</h3></div>
      {rows.length > 0 && <Pill tone={empty ? 'warning' : 'success'}>{empty ? `${empty} дн. без материалов` : 'поступают каждый день'}</Pill>}</div>
    {rows.map(([d, n]) => <div className="bar-row" key={d}><span>{fmtDate(d, true)}</span><div><i className={n ? 'good' : ''} style={{ width: `${(n / max) * 100}%` }} /></div><b>{n}</b></div>)}
    {!rows.length && <p className="muted">ML-сервис не вернул поступление материалов.</p>}
  </section>
}
