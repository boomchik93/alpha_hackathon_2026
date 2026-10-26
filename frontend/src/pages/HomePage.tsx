import { Link } from 'react-router-dom'
import { ArrowRight, Building2, ChevronRight, ShieldCheck, Sparkles, TrendingDown } from 'lucide-react'
import { useCompany, useFeed, useSummary } from '../api/hooks'
import type { FeedEntry } from '../api/types'
import { useSession } from '../auth/session'
import { Eyebrow, PageHeader, Pill, ScoreRing, Skel, pct } from '../components/ui'
import { SOURCE_LABEL, companyLabel, fmtDate, fmtRub, plural } from '../dict/labels'

function greeting() {
  const h = new Date().getHours()
  return h < 5 ? 'Доброй ночи' : h < 12 ? 'Доброе утро' : h < 18 ? 'Добрый день' : 'Добрый вечер'
}

export default function HomePage() {
  const s = useSession()
  const sum = useSummary(s.companyId)
  const company = useCompany(s.companyId)
  const n = sum.data?.new_count ?? 0
  const feed = useFeed(s.companyId, { tier: 'main' })
  const top = [...(feed.data?.entries ?? [])].sort((a, b) => b.score - a.score).slice(0, 3)
  const first = top.find(e => e.item_id === sum.data?.top?.item_id) ?? top[0]

  return <div className="page dashboard-page">
    <PageHeader eyebrow={fmtDate(new Date().toISOString())} title={greeting()} subtitle={company.data ? `${companyLabel(company.data)} · ${company.data.geo.region ?? 'регион не указан'}` : ' '} />

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
      {company.data ? <Link to="/profile" className="surface company-summary-card interactive-card">
        <div className="section-title"><div><Eyebrow>О компании</Eyebrow><h3>{companyLabel(company.data)}</h3></div><span className="company-summary-icon"><Building2 /></span></div>
        <p className="company-summary-type">{company.data.level === 'group' ? 'Группа компаний' : company.data.client_type === 'PROSPECT' ? 'Потенциальный клиент' : 'Клиент банка'}</p>
        <div className="company-summary-grid">
          <div><span>Отрасль</span><b>{company.data.industry.okved_name ?? 'Не указана'}</b></div>
          <div><span>Регион</span><b>{company.data.geo.region ?? 'Не указан'}</b></div>
          <div><span>Выручка</span><b>{fmtRub(company.data.size.revenue_rub)}</b></div>
          <div><span>Полнота профиля</span><b className={company.data.data_quality.missing_fields.length ? '' : 'positive'}>{company.data.data_quality.missing_fields.length ? 'Есть пробелы' : 'Высокая'}</b></div>
        </div>
        <div className="company-summary-footer">Открыть профиль <ArrowRight size={17} /></div>
      </Link> : <section className="surface company-summary-card" aria-busy="true"><Skel w="45%" h={20} /><Skel w="75%" /><Skel w="100%" h={120} /></section>}
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
