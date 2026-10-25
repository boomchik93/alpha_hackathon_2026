import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, BookOpen, ChevronDown, Landmark, Lightbulb, ShieldCheck, SlidersHorizontal } from 'lucide-react'
import { useCompany, useFeed } from '../api/hooks'
import type { FeedEntry, FeedFilters } from '../api/types'
import { useSession } from '../auth/session'
import { Dropdown, ErrorBanner, PageHeader, Pill, Skel, ringStyle, pct } from '../components/ui'
import { SOURCE_LABEL, fmtDate, fmtTime, plural, topicLabel } from '../dict/labels'

export default function FeedPage() {
  const s = useSession()
  const company = useCompany(s.companyId)
  const members = company.data?.level === 'group' ? company.data.member_ids : []
  const [scope, setScope] = useState('')
  const [kind, setKind] = useState<'all' | 'news' | 'bank_analytics'>('all')
  const [topic, setTopic] = useState('all')
  const [period, setPeriod] = useState('0')
  const [hideStale, setHideStale] = useState(false)
  const [maybeOpen, setMaybeOpen] = useState(false)

  // Разрез для группы: вся группа или отдельная компания-участник (К10)
  const feedId = scope && members.includes(scope) ? scope : s.companyId
  const all = useFeed(feedId)
  const asOf = all.data?.as_of
  const dateFrom = asOf && Number(period) ? shiftDate(asOf, -Number(period)) : undefined
  const filters: FeedFilters = { source_kind: kind === 'all' ? undefined : kind, topic: topic === 'all' ? undefined : topic, date_from: dateFrom, hide_stale: hideStale || undefined }
  const feed = useFeed(feedId, filters)
  const topics = useMemo(() => [...new Set(all.data?.entries.flatMap(e => e.topics))].sort(), [all.data])
  const main = feed.data?.entries.filter(e => e.tier === 'main') ?? []
  const maybe = feed.data?.entries.filter(e => e.tier === 'maybe') ?? []
  const reset = () => { setKind('all'); setTopic('all'); setPeriod('0'); setHideStale(false) }

  return <div className="page feed-page">
    <PageHeader eyebrow="Персональная лента" title="События для вашего бизнеса" subtitle="Не пересказ новостей — только то, что влияет на вашу компанию." action={<div className="updated"><span /> Обновлено сегодня, {feed.data ? fmtTime(feed.data.fetched_at ?? new Date().toISOString()) : '…'}</div>} />
    <div className="feed-layout">
      <aside className="filters surface">
        <div className="filter-title"><SlidersHorizontal size={18} /><b>Фильтры</b><button onClick={reset}>Сбросить</button></div>
        {members.length > 0 && <div className="filter-field"><span>Разрез</span><Dropdown ariaLabel="Разрез" value={feedId} onChange={setScope} options={[{ value: s.companyId, label: 'Вся группа' }, ...members.map(m => ({ value: m, label: `Компания ${m}` }))]} /></div>}
        <div className="filter-field"><span>Источник</span><Dropdown ariaLabel="Источник" value={kind} onChange={v => setKind(v as typeof kind)} options={[{ value: 'all', label: 'Все источники' }, { value: 'news', label: 'Новости' }, { value: 'bank_analytics', label: 'Аналитика банка' }]} /></div>
        <div className="filter-field"><span>Тема</span><Dropdown ariaLabel="Тема" value={topic} onChange={setTopic} options={[{ value: 'all', label: 'Все темы' }, ...topics.map(t => ({ value: t, label: topicLabel(t) }))]} /></div>
        <div className="filter-field"><span>Период</span><Dropdown ariaLabel="Период" value={period} onChange={setPeriod} options={[{ value: '0', label: 'Всё время' }, { value: '30', label: '30 дней' }, { value: '7', label: '7 дней' }]} /></div>
        <label className="switch-row"><input type="checkbox" checked={hideStale} onChange={e => setHideStale(e.target.checked)} /><span className="switch" />Скрыть устаревшие</label>
        <div className="filter-note"><ShieldCheck size={18} /><p><b>Лента персональна</b><br />{feed.data ? `${feed.data.filtered_out} ${plural(feed.data.filtered_out, 'материал отсеян', 'материала отсеяно', 'материалов отсеяно')} как не связанные с компанией` : '…'}</p></div>
      </aside>
      <div className="feed-content">
        {feed.error && <ErrorBanner error={feed.error as never} onRetry={() => feed.refetch()} />}
        <div className="feed-summary"><b>{main.length} {plural(main.length, 'важное событие', 'важных события', 'важных событий')}</b><span>Отсортированы по влиянию на {company.data?.level === 'group' && feedId === s.companyId ? 'группу' : 'компанию'}{asOf ? ` · данные на ${fmtDate(asOf)}` : ''}</span></div>
        {feed.isLoading && [0, 1].map(i => <div key={i} className="feed-card surface"><Skel w={78} h={78} /><div className="skel-stack"><Skel w="30%" /><Skel w="85%" h={20} /><Skel w="60%" /></div></div>)}
        {feed.data && main.length === 0 && <div className="surface empty-card"><h3>Подходящих событий нет</h3><p>По выбранным фильтрам нет материалов, уверенно связанных с вашей компанией. Снимите фильтры или посмотрите блок «Может быть интересно».</p><button className="secondary-button" onClick={reset}>Сбросить фильтры</button></div>}
        {main.map(e => <FeedCard e={e} key={e.item_id} />)}
        {!!maybe.length && <section className="maybe-section">
          <button className="maybe-toggle" onClick={() => setMaybeOpen(!maybeOpen)} aria-expanded={maybeOpen}><span><Lightbulb size={20} /><b>Может быть интересно</b><small>Связь с профилем слабее</small></span><span>{maybe.length}<ChevronDown className={maybeOpen ? 'rotated' : ''} /></span></button>
          {maybeOpen && maybe.map(e => <FeedCard e={e} key={e.item_id} compact />)}
        </section>}
      </div>
    </div>
  </div>
}

function FeedCard({ e, compact = false }: { e: FeedEntry; compact?: boolean }) {
  return <Link to={`/insight/${e.item_id}`} className={`feed-card surface interactive-card ${compact ? 'compact' : ''}`}>
    <div className="feed-card-side"><div className="score-box" style={ringStyle(e.score)}><div><b>{pct(e.score)}</b><small>релевантность</small></div></div></div>
    <div className="feed-card-body">
      <div className="meta-row">
        <Pill tone={e.source_kind === 'bank_analytics' ? 'bank' : 'neutral'}>{e.source_kind === 'bank_analytics' ? <Landmark size={13} /> : <BookOpen size={13} />} {SOURCE_LABEL[e.source_kind]}</Pill>
        {e.is_new && <Pill tone="red">Новое</Pill>}
        {e.stale && <Pill tone="warning">Устарело</Pill>}
        <span>{fmtDate(e.published_at)}</span>{e.topics.slice(0, 1).map(t => <span key={t}>· {topicLabel(t)}</span>)}{e.related.length > 0 && <span>· +{e.related.length} {plural(e.related.length, 'похожий материал', 'похожих материала', 'похожих материалов')}</span>}
      </div>
      <h2>{e.title}</h2>
      <div className="reason-row"><span>Почему вам:</span>{e.reasons.slice(0, 2).map(r => <Pill key={r}>{r}</Pill>)}</div>
      {!compact && <div className="card-state">Откройте, чтобы разобрать последствия для компании</div>}
    </div>
    <span className="round-link" aria-hidden="true"><ArrowRight /></span>
  </Link>
}

function shiftDate(iso: string, days: number) {
  const d = new Date(`${iso}T12:00:00`)
  d.setDate(d.getDate() + days)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
