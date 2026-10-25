import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ArrowRight, CircleAlert } from 'lucide-react'
import { api } from '../../api'
import type { ApiError, FollowupAnswer, InsightCard } from '../../api/types'
import { useRequestRole, useSession } from '../../auth/session'
import { AiPill, ErrorBanner, Eyebrow, Skel } from '../../components/ui'
import { addHistory } from '../../lib/history'
import { SourceButtons } from './parts'

// Ответы живут, пока открыта вкладка: при возврате к карточке переписка сохраняется.
const threads = new Map<string, FollowupAnswer[]>()
const SUGGESTIONS = ['Что ожидает банк?', 'Насколько это изменит наши расходы?', 'С чего начать?']

// prefill: вопрос из блока «Вопросы аналитику банка» — подставляется в поле, а не отправляется сразу.
export function Followup({ card, title, onSource, prefill }: { card: InsightCard; title: string; onSource(id: string): void; prefill?: { text: string; n: number } }) {
  const qc = useQueryClient()
  const s = useSession()
  const role = useRequestRole()
  const [items, setItems] = useState<FollowupAnswer[]>(threads.get(card.card_id) ?? [])
  const [question, setQuestion] = useState('')
  const [pendingQ, setPendingQ] = useState<string | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    if (!prefill) return
    setQuestion(prefill.text)
    document.getElementById('followup')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setTimeout(() => inputRef.current?.focus(), 400)
  }, [prefill])

  async function ask(q: string) {
    if (!q.trim() || pendingQ) return
    setPendingQ(q); setErr(null); setQuestion('')
    try {
      const a = await api.followup(card.card_id, q.trim(), role)
      if (s.user) addHistory({ login: s.user.login, company_id: card.company_id, item_id: card.item_ids[0], card_title: title, role: card.role, question: a.question, answer: a.answer, has_issues: a.verifier_issues.length > 0 })
      setItems(prev => { const next = [...prev, a]; threads.set(card.card_id, next); return next })
      qc.invalidateQueries({ queryKey: ['history'] })
    } catch (e) { setErr(e as ApiError) } finally { setPendingQ(null) }
  }

  return <section className="followup" id="followup">
    <div><Eyebrow>Уточнить вывод</Eyebrow><h2>Задайте вопрос по событию</h2><p>Ответ проходит ту же проверку цифр, что и карточка, и опирается на её источники. До 500 символов.</p></div>
    {items.map((a, i) => <div key={i} className="qa">
      <div className="qa-question">{a.question}</div>
      <div className="answer">
        {a.is_model_inference && <AiPill />}
        <p>{a.answer}</p>
        {a.verifier_issues.map(v => <div key={v} className="limitation"><CircleAlert /><p>{v}</p></div>)}
        {a.source_ids.length > 0 && <div className="answer-sources"><SourceButtons ids={a.source_ids} sources={[...card.sources, ...a.sources]} onSource={onSource} active={null} /></div>}
      </div>
    </div>)}
    {pendingQ && <div className="qa"><div className="qa-question">{pendingQ}</div><div className="answer"><small className="thinking">Модель отвечает и сверяет цифры…</small><Skel w="92%" /><Skel w="78%" /></div></div>}
    {err && <ErrorBanner error={err} />}
    <form onSubmit={(e: FormEvent) => { e.preventDefault(); ask(question) }}>
      <div className="question-box">
        <textarea ref={inputRef} id="followup-question" maxLength={500} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Например: что это значит для бюджета на 2027 год?"
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(question) } }} />
        <button disabled={!question.trim() || !!pendingQ} aria-label="Спросить">{pendingQ ? <span className="spinner" /> : <ArrowRight />}</button>
      </div>
      <div className="suggestions">{SUGGESTIONS.map(q => <button type="button" key={q} onClick={() => ask(q)} disabled={!!pendingQ}>{q}</button>)}<span className="counter">{question.length}/500</span></div>
    </form>
  </section>
}
