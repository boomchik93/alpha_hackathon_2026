import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Bot, BookOpen, CircleAlert, Clock3, ExternalLink, FileChartColumn, Landmark, Layers3, MessageSquareText, ShieldCheck, TrendingDown, TrendingUp } from 'lucide-react'
import { useCard, useCompany, useFeed } from '../../api/hooks'
import type { CompanyProfile, FeedEntry, InsightCard, SourceRef } from '../../api/types'
import { useRequestRole, useSession } from '../../auth/session'
import { ErrorBanner, Eyebrow, Pill, Skel, pct, ringStyle } from '../../components/ui'
import { ROLE_LABEL, SOURCE_LABEL, featureText, fmtDate, fmtTime, topicLabel } from '../../dict/labels'
import { Followup } from './Followup'
import { Claim, Feedback, Generating, SourceButtons, srcNum } from './parts'
import { Scenarios } from './Scenarios'

export default function InsightPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const s = useSession()
  const role = useRequestRole()
  const feed = useFeed(s.companyId)
  const company = useCompany(s.companyId)
  const entry = feed.data?.entries.find(e => e.item_id === id)
  const { card, job, pending, error, retry } = useCard(s.companyId, id, role)

  return <div className="page insight-page">
    <button className="back-button" onClick={() => navigate('/feed')}><ArrowLeft size={18} /> К ленте</button>
    <Header card={card} entry={entry} />
    {error && !pending && <div className="insight-error"><ErrorBanner error={error} onRetry={retry} /></div>}
    {pending && <Generating job={job} />}
    {card && <Body card={card} entry={entry} profile={company.data} />}
  </div>
}

// what_happened.facts — id фактов из разбора материала, а не источников; ссылаемся на сами материалы-новости
const factSources = (card: InsightCard) => { const news = card.sources.filter(x => x.kind !== 'bank_analytics'); return (news.length ? news : card.sources).map(x => x.id) }

export const cardTitle = (card?: InsightCard, entry?: FeedEntry) => entry?.title ?? card?.sources.find(x => x.kind !== 'bank_analytics')?.title ?? card?.sources[0]?.title

function Header({ card, entry }: { card?: InsightCard; entry?: FeedEntry }) {
  const s = useSession()
  const title = cardTitle(card, entry)
  const kind = entry?.source_kind ?? card?.sources[0]?.kind
  const date = entry?.published_at ?? card?.sources[0]?.published_at
  const score = card?.relevance.score ?? entry?.score
  const reasons = card?.relevance.reasons ?? entry?.reasons ?? []
  const role = card?.role ?? s.role
  return <div className="insight-header">
    <div>
      <div className="meta-row">
        {kind && <Pill tone={kind === 'bank_analytics' ? 'bank' : 'neutral'}>{kind === 'bank_analytics' ? <Landmark size={13} /> : <BookOpen size={13} />} {SOURCE_LABEL[kind]}</Pill>}
        {entry?.is_new && <Pill tone="red">Новое</Pill>}
        {entry?.stale && <Pill tone="warning">Устарело</Pill>}
        {date && <span>{fmtDate(date)}</span>}
        {entry?.topics.slice(0, 2).map(t => <span key={t}>· {topicLabel(t)}</span>)}
      </div>
      {title ? <h1>{title}</h1> : <div className="skel-stack"><Skel w="80%" h={40} /><Skel w="50%" h={40} /></div>}
      <p>{role && role !== 'other' ? `Персональная интерпретация для роли «${ROLE_LABEL[role].toLowerCase()}»` : 'Интерпретация без акцента на роль'}</p>
      <div className="feature-list">{reasons.map(r => <Pill key={r}>{r}</Pill>)}</div>
    </div>
    {score !== undefined && <div className="big-score"><b style={ringStyle(score)}><span>{pct(score)}</span></b><em>Релевантность</em></div>}
  </div>
}

const TOC = [['what', 'Что произошло'], ['why', 'Почему важно вам'], ['cons', 'Последствия'], ['act', 'Что можно сделать'], ['bank', 'Позиция банка'], ['ai', 'Вывод модели'], ['scen', 'Сценарии'], ['sources', 'Источники'], ['followup', 'Задать вопрос']] as const

function SourceCard({ src, card, active }: { src: SourceRef; card: InsightCard; active: string | null }) {
  const bank = src.kind === 'bank_analytics'
  const meta = [bank ? 'Аналитика Альфа-Банка' : SOURCE_LABEL[src.kind], src.page ? `страница ${src.page}` : null, fmtDate(src.published_at)].filter(Boolean).join(' · ')
  const inner = <>{bank ? <FileChartColumn /> : <BookOpen />}<div><b><em className="src-n">{srcNum(src.id, card.sources)}</em>{src.title}</b><span>{meta}{src.stale ? ' · устарело' : ''}</span>{src.quote && <q className="src-quote">{src.quote}</q>}</div><span className="icon-button" title="Открыть источник"><ExternalLink /></span></>
  const cls = `source-card interactive-card ${active === src.id ? 'flash' : ''}`
  if (src.doc_id) return <Link id={`src-${src.id}`} to={`/source/${card.item_ids[0]}/${src.id}`} className={cls}>{inner}</Link>
  if (src.url) return <a id={`src-${src.id}`} href={src.url} target="_blank" rel="noreferrer" className={cls}>{inner}</a>
  return <div id={`src-${src.id}`} className={`source-card ${active === src.id ? 'flash' : ''}`}>{inner}</div>
}

function Body({ card, entry, profile }: { card: InsightCard; entry?: FeedEntry; profile?: CompanyProfile }) {
  const navigate = useNavigate()
  const [active, setActive] = useState<string | null>(null)
  const [prefill, setPrefill] = useState<{ text: string; n: number }>()
  const actions = card.suggested_actions.filter(a => a.type === 'action')
  const questions = card.suggested_actions.filter(a => a.type === 'question_for_analysis')
  const n = (key: string) => String(TOC.findIndex(([k]) => k === key) + 1).padStart(2, '0')
  const itemId = card.item_ids[0]

  const onSource = (id: string) => {
    setActive(id)
    document.getElementById(`src-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => setActive(a => (a === id ? null : a)), 2200)
  }
  const go = (key: string) => document.getElementById(key)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  return <div className="insight-layout">
    <div className="story">
      {card.meta.verifier_issues.length > 0 && <div className="warning-banner"><CircleAlert /><div><b>Проверка оставила замечания</b>{card.meta.verifier_issues.map(v => <p key={v}>{v}</p>)}</div></div>}

      <section className="story-section" id="what"><div className="section-number">{n('what')}</div><div className="full">
        <Eyebrow>Что произошло</Eyebrow><h2>По фактам из источников</h2>
        <p className="what-text">{card.what_happened.text}</p>
        <div className="fact-refs"><SourceButtons ids={factSources(card)} sources={card.sources} onSource={onSource} active={active} /></div>
      </div></section>

      <section className="story-section why" id="why"><div className="section-number">{n('why')}</div><div>
        <Eyebrow>Почему это важно вам</Eyebrow><h2 className="why-text">{card.why_it_matters.text}</h2>
        <div className="feature-list">{card.why_it_matters.company_features_used.map(f => <Pill key={f}>{featureText(profile, f)}</Pill>)}</div>
      </div></section>

      <section className="story-section" id="cons"><div className="section-number">{n('cons')}</div><div className="full">
        <Eyebrow>Возможные последствия</Eyebrow><h2>Где риск, а где возможность</h2>
        <div className="consequence-grid">
          <div className="consequence risk"><div className="consequence-title"><TrendingDown /><span>Риски</span></div>{card.consequences.risks.map(c => <Claim key={c.text} c={c} sources={card.sources} onSource={onSource} active={active} />)}{!card.consequences.risks.length && <p className="empty-claim">Не выявлены</p>}</div>
          <div className="consequence opportunity"><div className="consequence-title"><TrendingUp /><span>Возможности</span></div>{card.consequences.opportunities.map(c => <Claim key={c.text} c={c} sources={card.sources} onSource={onSource} active={active} />)}{!card.consequences.opportunities.length && <p className="empty-claim">Не выявлены</p>}</div>
        </div>
      </div></section>

      <section className="story-section" id="act"><div className="section-number">{n('act')}</div><div className="full">
        <Eyebrow>Что можно сделать</Eyebrow><h2>Следующие шаги</h2>
        <ol className="action-list">
          {actions.map((a, i) => <li key={a.text}><span className="action-index">{i + 1}</span><div><b>{a.text}</b>{a.horizon && <small>{a.horizon}</small>}</div>{a.is_model_inference || !a.evidence.length ? <Pill tone="ai"><Bot size={13} /> Вывод модели</Pill> : <span className="claim-refs"><SourceButtons ids={a.evidence} sources={card.sources} onSource={onSource} active={active} /></span>}</li>)}
          {questions.map(q => <li key={q.text} className="question-item"><span className="action-index"><MessageSquareText size={13} /></span><div><b>Вопрос для уточнения</b><p>{q.text}</p></div><button className="secondary-button" onClick={() => setPrefill({ text: q.text, n: Date.now() })}>Спросить <ArrowRight size={15} /></button></li>)}
        </ol>
      </div></section>

      {/* Позиция банка — только дословные цитаты из аналитики (К03, О02). Отделена от вывода модели цветом, подписью, иконкой и рамкой. */}
      <section className="bank-view" id="bank"><div className="block-label"><Landmark /> Позиция Альфа-Банка · дословно</div>
        {card.bank_view ? (card.bank_view.quotes.length ? card.bank_view.quotes : [{ source_id: card.bank_view.source_ids[0], text: card.bank_view.text }]).map(q => {
          const src = card.sources.find(x => x.id === q.source_id)
          return <div key={q.text}>
            <blockquote>«{q.text}»</blockquote>
            {src && <Link to={`/source/${itemId}/${src.id}`} className="source-link light-source">{src.title}{src.page ? ` · стр. ${src.page}` : ''}{src.published_at ? ` · ${fmtDate(src.published_at)}` : ''} <ExternalLink size={15} /></Link>}
          </div>
        }) : <p className="bank-empty">У банка нет аналитики по этой теме. Всё остальное в карточке — вывод модели на основе материала и профиля компании.</p>}
      </section>
      <section className="ai-view" id="ai"><div className="block-label"><Bot /> {card.ai_inference.label || 'Вывод модели'}</div><p>{card.ai_inference.text}</p><div className="ai-note"><CircleAlert /> Это интерпретация AI, а не позиция банка.</div></section>

      <section className="story-section" id="scen"><div className="section-number">{n('scen')}</div><Scenarios card={card} onSource={onSource} active={active} /></section>

      <section className="story-section sources" id="sources"><div className="section-number">{n('sources')}</div><div className="full">
        <Eyebrow>Основания и прозрачность</Eyebrow><h2>Источники и ограничения</h2>
        {card.sources.map(src => <SourceCard key={src.id} src={src} card={card} active={active} />)}
        {entry && entry.related.length > 0 && <p className="muted small-note">Похожие материалы объединены с этим: {entry.related.length}.</p>}
        {card.limitations.map(l => <div key={l} className="limitation"><CircleAlert /><p><b>Ограничение:</b> {l}</p></div>)}
        <button className="trace-link" onClick={() => navigate(`/trace/${itemId}`)}><Layers3 /> Посмотреть, как получен вывод <ArrowRight /></button>
      </div></section>

      <Followup card={card} title={cardTitle(card, entry) ?? ''} onSource={onSource} prefill={prefill} />
      <Feedback card={card} />
    </div>
    <aside className="insight-toc surface">
      <b>В карточке</b>
      {TOC.map(([k, l], i) => <a href={`#${k}`} key={k} onClick={e => { e.preventDefault(); go(k) }}><span>{String(i + 1).padStart(2, '0')}</span>{l}</a>)}
      <hr />
      <div><ShieldCheck /> {card.meta.verifier_issues.length ? 'Проверка с замечаниями' : 'Цифры и цитаты проверены'}</div>
      <div><Bot /> Персональный вывод модели</div>
      <div><Clock3 /> Собрана {fmtDate(card.meta.generated_at, true)}, {fmtTime(card.meta.generated_at)}</div>
    </aside>
  </div>
}
