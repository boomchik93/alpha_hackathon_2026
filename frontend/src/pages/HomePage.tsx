import { Link } from 'react-router-dom'
import { ArrowRight, ChevronRight, ShieldCheck, Sparkles, TrendingDown, TrendingUp, WalletCards } from 'lucide-react'
import { useCompany, useFeed, useSummary } from '../api/hooks'
import type { FeedEntry } from '../api/types'
import { useSession } from '../auth/session'
import { Eyebrow, PageHeader, Pill, ScoreRing, Skel, pct } from '../components/ui'
import { SOURCE_LABEL, companyLabel, fmtDate, plural } from '../dict/labels'

function greeting() {
  const h = new Date().getHours()
  return h < 5 ? 'Доброй ночи' : h < 12 ? 'Доброе утро' : h < 18 ? 'Добрый день' : 'Добрый вечер'
}

export const TASKS = [
  { title: 'Подписать 3 платежа', note: 'Поставщики сырья · 1,8 млн ₽', date: 'Сегодня, до 19:00', tone: 'urgent' },
  { title: 'Проверить запас по ковенантам', note: 'Сценарий сохранения ставки 14% до весны', date: 'До 28 октября', tone: 'normal' },
  { title: 'Обновить валютный прогноз', note: 'Валютные расчёты на I квартал 2027', date: 'До 30 октября', tone: 'normal' },
  { title: 'Сверить лимиты контрагентов', note: '6 компаний требуют пересмотра', date: 'До 1 ноября', tone: 'normal' },
]

export default function HomePage() {
  const s = useSession()
  const sum = useSummary(s.companyId)
  const company = useCompany(s.companyId)
  const n = sum.data?.new_count ?? 0
  const feed = useFeed(s.companyId, { tier: 'main' })
  const top = [...(feed.data?.entries ?? [])].sort((a, b) => b.score - a.score).slice(0, 3)
  const first = top.find(e => e.item_id === sum.data?.top?.item_id) ?? top[0]

  return <div className="page dashboard-page">
    <PageHeader eyebrow={fmtDate(new Date().toISOString())} title={greeting()} subtitle={company.data ? `${companyLabel(company.data)} · ${company.data.company_id} · ${company.data.geo.region ?? 'регион не указан'}` : ' '} />

    {/* Главная точка входа в сервис (К09) */}
    <section className="signal-hero">
      <div className="hero-copy">
        <div className="hero-badge"><Sparkles size={15} /> Персональная аналитика</div>
        <h2>{sum.isLoading ? 'Собираем события…' : n > 0 ? <>{n} {plural(n, 'новое событие может', 'новых события могут', 'новых событий могут')} повлиять<br />на ваш бизнес</> : <>Новых важных событий нет</>}</h2>
        <p>{first ? <>Самое важное — «{first.title}». Почему вам: {first.reasons.join(', ')}.</> : 'Последние материалы — в ленте событий.'}</p>
        <div className="hero-actions">
          {first && <Link className="primary-button light" to={`/insight/${first.item_id}`}>Разобрать главное <ArrowRight size={18} /></Link>}
          <Link className="text-button light-text" to="/feed">Вся лента</Link>
        </div>
      </div>
      <div className="impact-orbit" aria-hidden="true"><div className="orbit outer" /><div className="orbit inner" /><div className="orbit-core"><span>{first ? pct(first.score) : '—'}</span><small>влияние главного события</small></div><span className="orb-dot d1" /><span className="orb-dot d2" /><span className="orb-dot d3" /></div>
      {first && <div className="hero-depth-card debt"><span><TrendingDown /></span><div><small>{first.reasons[0]}</small><b>Высокое влияние</b></div></div>}
      {feed.data && <div className="hero-depth-card liquidity"><span><ShieldCheck /></span><div><small>Лента персональна</small><b>{feed.data.filtered_out} {plural(feed.data.filtered_out, 'материал отсеян', 'материала отсеяно', 'материалов отсеяно')}</b></div></div>}
    </section>

    <div className="dashboard-grid">
      <Link to="/finances" className="surface accounts-card interactive-card">
        <div className="section-title"><div><Eyebrow>Финансы</Eyebrow><h3>Счета и остатки</h3></div><span className="icon-button"><ChevronRight /></span></div>
        <div className="balance">1 284 560 312 ₽</div><div className="balance-caption">Доступно на 3 счетах · рубли, доллары, лиры</div>
        <div className="mini-chart"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
        <div className="card-footer"><span>Поступления сегодня</span><b className="positive">+ 48 600 000 ₽</b></div>
      </Link>
      <section className="surface action-card">
        <div className="section-title"><div><Eyebrow>На контроле</Eyebrow><h3>Ближайшие действия</h3></div><span className="tiny-count">{TASKS.length}</span></div>
        {TASKS.slice(0, 3).map((t, i) => <Link to={`/tasks?focus=${i}`} className="task-row" key={t.title}><span className={`task-icon t${i}`}>{i === 0 ? <WalletCards /> : i === 1 ? <ShieldCheck /> : <TrendingUp />}</span><div><b>{t.title}</b><small>{t.date}</small></div><ChevronRight size={18} /></Link>)}
      </section>
    </div>

    <section className="section-block">
      <div className="section-heading"><div><Eyebrow>В фокусе</Eyebrow><h2>Важное для компании</h2></div><Link to="/feed">Смотреть все <ArrowRight size={17} /></Link></div>
      <div className="focus-grid">
        {feed.isLoading && [0, 1, 2].map(i => <div key={i} className="focus-card"><Skel w="40%" /><Skel w="90%" h={20} /><Skel w="70%" /></div>)}
        {top.map((e, i) => <FocusCard e={e} index={i} key={e.item_id} />)}
      </div>
    </section>
  </div>
}

function FocusCard({ e, index }: { e: FeedEntry; index: number }) {
  return <Link to={`/insight/${e.item_id}`} className={`focus-card focus-${index}`}>
    <div className="focus-top"><Pill tone={e.source_kind === 'bank_analytics' ? 'bank' : 'neutral'}>{SOURCE_LABEL[e.source_kind]}</Pill><ScoreRing score={e.score} dark={index === 0} /></div>
    <h3>{e.title}</h3><p>Почему вам: {e.reasons[0]}</p>
    <div className="focus-bottom"><span>{fmtDate(e.published_at, true)}{e.is_new ? ' · новое' : ''}</span><ArrowRight size={19} /></div>
  </Link>
}
