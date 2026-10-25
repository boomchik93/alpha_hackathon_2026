import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, CircleAlert, MessageSquareText, Search } from 'lucide-react'
import { readHistory } from '../lib/history'
import type { HistoryItem } from '../api/types'
import { useSession } from '../auth/session'
import { AiPill, ErrorBanner, Eyebrow, PageHeader, Skel } from '../components/ui'
import { ROLE_LABEL, fmtDate, fmtTime } from '../dict/labels'

export default function HistoryPage() {
  const s = useSession()
  const nav = useNavigate()
  const q = useQuery({ queryKey: ['history', s.user?.login], queryFn: async () => readHistory(s.user!.login), enabled: !!s.user, staleTime: 0 })
  const [search, setSearch] = useState('')
  const [onlyCompany, setOnlyCompany] = useState(true)

  const list = useMemo(() => {
    const t = search.trim().toLowerCase()
    return (q.data ?? []).filter(h => !onlyCompany || h.company_id === s.companyId).filter(h => !t || (h.question + h.answer + h.card_title).toLowerCase().includes(t))
  }, [q.data, search, onlyCompany, s.companyId])
  const groups = useMemo(() => {
    const m = new Map<string, HistoryItem[]>()
    const today = fmtDate(new Date().toISOString())
    list.forEach(h => { const d = fmtDate(h.asked_at); const k = d === today ? 'Сегодня' : d; m.set(k, [...(m.get(k) ?? []), h]) })
    return [...m.entries()]
  }, [list])
  const open = (h: HistoryItem) => { if (h.company_id !== s.companyId) s.setCompanyId(h.company_id); nav(`/insight/${h.item_id}`) }

  return <div className="page history-page">
    <PageHeader eyebrow="Уточняющие вопросы" title="История вопросов" subtitle="Вопросы по карточкам событий и ответы. История хранится в этом браузере." />
    <div className="history-tools">
      <label className="search-field" htmlFor="history-search"><Search size={17} /><input id="history-search" name="history-search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Поиск по вопросам и ответам" autoComplete="off" /></label>
      <label className="switch-row inline"><input type="checkbox" checked={onlyCompany} onChange={e => setOnlyCompany(e.target.checked)} /><span className="switch" />Только текущая компания</label>
    </div>
    {q.error && <ErrorBanner error={q.error as never} onRetry={() => q.refetch()} />}
    {q.isLoading && <div className="surface history-item"><Skel w="50%" h={20} /><Skel w="90%" /></div>}
    {q.data && list.length === 0 && <div className="surface empty-card"><MessageSquareText /><h3>{search ? 'Ничего не нашлось' : 'Вопросов пока нет'}</h3><p>{search ? 'Попробуйте другое слово или снимите фильтр по компании.' : 'Откройте карточку события и задайте вопрос в блоке «Задайте вопрос по событию».'}</p>{!search && <button className="primary-button" onClick={() => nav('/feed')}>Перейти в ленту <ArrowRight size={17} /></button>}</div>}
    {groups.map(([day, items]) => <section key={day} className="history-group">
      <Eyebrow>{day}</Eyebrow>
      {items.map(h => <article key={h.id} className="surface history-item">
        <div className="history-q"><h3>{h.question}</h3><span>{fmtTime(h.asked_at)}</span></div>
        <div className="answer"><AiPill /><p>{h.answer}</p>{h.has_issues && <div className="limitation"><CircleAlert /><p>Проверка оставила предупреждение — подробности в карточке.</p></div>}</div>
        <div className="history-foot"><span>К событию: <b>{h.card_title}</b> {h.role ? ` · ${ROLE_LABEL[h.role]}` : ''}</span><button className="text-button link-red" onClick={() => open(h)}>Открыть карточку <ArrowRight size={15} /></button></div>
      </article>)}
    </section>)}
  </div>
}
