import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, BadgeCheck, Bot, Building2, Check, Landmark, Network, RotateCw, Search, ShieldCheck, Target, Wrench } from 'lucide-react'
import { api } from '../../api'
import { useCard, useCompany, useFeed } from '../../api/hooks'
import type { ApiError, CardTrace } from '../../api/types'
import { useRequestRole, useSession } from '../../auth/session'
import { ErrorBanner, Eyebrow, Logo, PageHeader, Pill, Skel } from '../../components/ui'
import { ROLE_LABEL, SOURCE_LABEL, featureText, fmtDate, plural } from '../../dict/labels'
import { cardTitle } from './InsightPage'

const TOOL_LABEL: Record<string, string> = {
  get_item: 'Чтение материала', get_company_profile: 'Профиль компании', search_bank_analytics: 'Поиск в аналитике банка',
  verify_card: 'Проверка цифр и цитат', 'llm.generate': 'Генерация текста', get_feed: 'Лента компании',
  compute_exposure: 'Оценка влияния на компанию', search_bank_analytics_multi: 'Поиск в аналитике банка по нескольким темам',
  get_counterparty_network: 'Связи с контрагентами', search_news: 'Поиск похожих новостей',
}
// Неизвестный инструмент не показываем сырым именем: подпись общая, имя остаётся в подсказке.
const toolLabel = (t: string) => TOOL_LABEL[t] ?? 'Служебный шаг'
const TOOL_ICON: Record<string, typeof Search> = {
  get_item: Search, get_company_profile: Building2, search_bank_analytics: Landmark, search_bank_analytics_multi: Landmark,
  compute_exposure: Target, get_counterparty_network: Network, search_news: Search, verify_card: ShieldCheck, 'llm.generate': Bot,
}
// Пункт плана без инструмента (tool: null в ml/agent/orchestrator.py, make_plan) — шаг рассуждения модели: только пояснение.
const planText = (p: CardTrace['plan'][number]) => {
  const why = p.why.charAt(0).toLowerCase() + p.why.slice(1)
  return p.tool ? `${toolLabel(p.tool)} — ${why}` : p.why
}

// Первоисточник (К10): PDF аналитики банка открывается на нужной странице, у новости — ссылка на сайт.
export function SourcePage() {
  const { itemId = '', sourceId = '' } = useParams()
  const navigate = useNavigate()
  const s = useSession()
  const { card, pending, error } = useCard(s.companyId, itemId, useRequestRole())
  const src = card?.sources.find(x => x.id === sourceId)
  const quote = card?.bank_view?.quotes.find(q => q.source_id === sourceId)?.text ?? src?.quote
  const bank = src?.kind === 'bank_analytics'
  const pdf = useQuery({ queryKey: ['pdf', src?.doc_id], queryFn: () => api.sourcePdf(src!.doc_id!), enabled: !!src?.doc_id, staleTime: Infinity, retry: false })
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!pdf.data) return
    const u = URL.createObjectURL(pdf.data)
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [pdf.data])

  return <div className="page source-page">
    <button className="back-button" onClick={() => navigate(`/insight/${itemId}`)}><ArrowLeft size={18} /> К карточке</button>
    {error && <ErrorBanner error={error} />}
    {pending && <div className="skel-stack"><Skel w="40%" h={36} /><Skel w="70%" /></div>}
    {card && !src && <ErrorBanner error={{ status: 404, code: 'not_found', message: 'Источник не найден в карточке' }} />}
    {card && src && <>
      <PageHeader eyebrow="Первоисточник" title={src.title} subtitle={`${bank ? 'Аналитика Альфа-Банка' : SOURCE_LABEL[src.kind]}${src.published_at ? ` · ${fmtDate(src.published_at)}` : ''}${src.page ? ` · страница ${src.page}` : ''}`} />
      <div className="source-layout">
        {url
          ? <iframe className="source-pdf" title={src.title} src={`${url}#page=${src.page ?? 1}`} />
          : <section className={`source-preview ${bank ? 'bank-document' : 'cbr-document'}`}>
            <div className="document-head"><Logo /><span>{fmtDate(src.published_at)}</span></div>
            <Eyebrow>{bank ? 'Аналитика банка' : SOURCE_LABEL[src.kind]}</Eyebrow>
            <h2>{src.title}</h2>
            {pdf.isLoading && <Skel w="80%" />}
            {quote ? <blockquote>«{quote}»</blockquote> : <p>{card.what_happened.text}</p>}
            <div className="document-footer"><span>{src.page ?? 1}</span><span>{src.doc_id ? 'PDF отчёта недоступен в демо-режиме: показан фрагмент, на который ссылается карточка' : 'Фрагмент материала, на который ссылается карточка'}</span></div>
          </section>}
        <aside className="surface source-context">
          <BadgeCheck />
          <Eyebrow>Использовано в выводе</Eyebrow>
          <h3>Почему этот фрагмент важен</h3>
          <p>{bank ? 'Это дословная позиция банка. В карточке она показана отдельно от интерпретации модели.' : 'Из этого источника взяты факты блока «Что произошло».'}</p>
          <hr />
          <div><span>Тип</span><b>{SOURCE_LABEL[src.kind]}</b></div>
          {src.published_at && <div><span>Дата</span><b>{fmtDate(src.published_at)}</b></div>}
          {src.page && <div><span>Страница</span><b>{src.page}</b></div>}
          {src.stale && <div><span>Актуальность</span><b>устарел</b></div>}
          {pdf.error && <p className="muted">PDF не открылся: {(pdf.error as unknown as ApiError).message}</p>}
          {url && <a className="secondary-button" href={`${url}#page=${src.page ?? 1}`} target="_blank" rel="noreferrer">Открыть PDF в новой вкладке</a>}
          {src.url && <a className="secondary-button" href={src.url} target="_blank" rel="noreferrer">Открыть на сайте</a>}
        </aside>
      </div>
    </>}
  </div>
}

// Трасса (К11, К12): /cards/{id}/trace. Пользователь видит упрощённый вид, эксперт и админ — ещё и стоимость.
export function TracePage() {
  const { itemId = '' } = useParams()
  const navigate = useNavigate()
  const s = useSession()
  const { card, pending, error } = useCard(s.companyId, itemId, useRequestRole())
  const feed = useFeed(s.companyId)
  const company = useCompany(s.companyId)
  const trace = useQuery({ queryKey: ['trace', card?.card_id], queryFn: () => api.cardTrace(card!.card_id), enabled: !!card, retry: false })
  const full = s.user?.access !== 'user'

  if (!card) return <div className="page">{error ? <ErrorBanner error={error} /> : pending ? <div className="skel-stack"><Skel w="40%" h={36} /><Skel w="70%" /></div> : null}</div>
  const t = trace.data
  const title = cardTitle(card, feed.data?.entries.find(e => e.item_id === itemId))
  const ms = (v: number | null) => (v === null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1).replace('.', ',')} с` : `${v} мс`)
  const lastCheck = t?.checks.at(-1)
  // Число цитат банка — итог по карточке: показываем его у первого поиска в аналитике, а не у каждого
  const firstBankSearch = t?.tool_calls.findIndex(c => c.tool.startsWith('search_bank_analytics')) ?? -1

  return <div className="page">
    <button className="back-button" onClick={() => navigate(`/insight/${itemId}`)}><ArrowLeft size={18} /> К карточке</button>
    <PageHeader eyebrow="Трассируемость" title="Как получен вывод" subtitle={title ? `Карточка «${title}»` : undefined} action={lastCheck && (lastCheck.issues ? <Pill tone="warning">Замечаний проверки: {lastCheck.issues}</Pill> : <Pill tone="success"><Check size={13} /> Проверка пройдена</Pill>)} />
    {trace.error && <ErrorBanner error={trace.error as unknown as ApiError} onRetry={() => trace.refetch()} />}
    {trace.isLoading && <div className="skel-stack"><Skel w="60%" /><Skel w="80%" /></div>}
    {t && <div className="trace-layout">
      <section className="surface trace-timeline">
        <div className="trace-step"><div className="trace-line"><span><Search /></span></div><div><small>План</small><h3>Что модель решила сделать</h3><p>{t.plan.map(planText).join('; ')}.</p></div><b>{t.plan.length} {plural(t.plan.length, 'пункт', 'пункта', 'пунктов')}</b></div>
        {t.tool_calls.map((c, i) => { const Icon = TOOL_ICON[c.tool] ?? Wrench; return <div key={i} className="trace-step"><div className="trace-line"><span><Icon /></span></div><div><small>Шаг {String(i + 1).padStart(2, '0')}</small><h3 title={c.tool}>{toolLabel(c.tool)}</h3>{c.tool === 'get_company_profile' && <p>Использованы признаки: {t.company_features_used.map(f => featureText(company.data, f).toLowerCase()).join(', ')}.</p>}{i === firstBankSearch && <p>{card.bank_view ? `Найдено цитат: ${card.bank_view.quotes.length}.` : 'Материалов банка по теме не найдено, поэтому блок позиции банка пуст.'}</p>}</div><b>{ms(c.ms)}</b></div> })}
        {t.replans.map((r, i) => <div key={`r${i}`} className="trace-step warn"><div className="trace-line"><span><RotateCw /></span></div><div><small>Повтор</small><h3>Перепланирование</h3><p>{r.why}</p></div><b>{r.trigger}</b></div>)}
        {t.checks.map(c => <div key={`c${c.attempt}`} className={`trace-step ${c.issues ? 'warn' : ''}`}><div className="trace-line"><span><ShieldCheck /></span></div><div><small>Проверка, попытка {c.attempt + 1}</small><h3>{c.issues ? 'Найдены несоответствия' : 'Цифры и цитаты совпали с источниками'}</h3><p>{c.issues ? `Замечаний: ${c.issues}. Ответ отправлен на доработку.` : 'Каждое утверждение со ссылкой сверено с текстом источника.'}</p></div><b>{c.issues ? `${c.issues} замеч.` : 'ок'}</b></div>)}
        <div className="trace-step last"><div className="trace-line"><span><BadgeCheck /></span></div><div><small>Источники</small><h3>На что опирается карточка</h3><p>{t.sources.map(x => `${x.title}${x.page ? `, стр. ${x.page}` : ''}`).join('; ')}</p></div><b>{t.sources.length}</b></div>
      </section>
      <aside className="surface trace-meta">
        <h3>Параметры запуска</h3>
        {[[ 'Роль', card.role ? ROLE_LABEL[card.role] : 'Без ролевого акцента'], ['Длительность', t.duration_s === null ? '—' : `${t.duration_s.toFixed(1).replace('.', ',')} с`], ['Повторов', String(t.retries)], ['Источников', String(t.sources.length)], ...(full && t.cost_rub !== undefined ? [['Стоимость', `${t.cost_rub.toFixed(2).replace('.', ',')} ₽`]] : [])].map(([a, b]) => <div key={a}><span>{a}</span><b>{b}</b></div>)}
        {!full && <p className="trace-hint">Полная трасса с расходом доступна эксперту и администратору.</p>}
      </aside>
    </div>}
  </div>
}
