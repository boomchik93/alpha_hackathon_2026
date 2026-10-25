import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, BadgeCheck, Bot, Check, CircleCheck, Landmark, Search, ShieldCheck, Building2, ThumbsDown, ThumbsUp } from 'lucide-react'
import { api } from '../../api'
import type { ApiError, Claim as ClaimT, FeedbackReason, InsightCard, JobView, SourceRef } from '../../api/types'
import { useSession } from '../../auth/session'
import { ErrorBanner, Eyebrow, Pill, Skel } from '../../components/ui'
import { GEN_STEPS, REASON_LABEL } from '../../dict/labels'

// Номер источника — его позиция в карточке: id у ML произвольные (src_..., s1)
export const srcNum = (id: string, sources: SourceRef[]) => { const i = sources.findIndex(x => x.id === id); return i >= 0 ? String(i + 1) : id }

export function SourceButtons({ ids, sources, onSource, active }: { ids: string[]; sources: SourceRef[]; onSource(id: string): void; active: string | null }) {
  return <>{ids.map(id => {
    if (id === 'profile') return <span key={id} className="source-chip" title="Данные профиля компании"><Building2 size={12} /> Профиль</span>
    const s = sources.find(x => x.id === id)
    return <button key={id} className={`source-chip ${active === id ? 'active' : ''}`} onClick={() => onSource(id)} title={s?.title}>
      {s?.kind === 'bank_analytics' ? <Landmark size={12} /> : <BadgeCheck size={12} />} Источник {srcNum(id, sources)}
    </button>
  })}</>
}

// Утверждение: либо со ссылками на источники, либо с явной пометкой «вывод модели» (К03, О02).
export function Claim({ c, sources, onSource, active }: { c: ClaimT; sources: SourceRef[]; onSource(id: string): void; active: string | null }) {
  return <div className="claim"><b>{c.text}</b>{c.horizon && <span>{c.horizon}</span>}
    <div className="claim-refs">{c.is_model_inference || !c.evidence.length ? <span className="model-chip"><Bot size={12} /> Вывод модели</span> : <SourceButtons ids={c.evidence} sources={sources} onSource={onSource} active={active} />}</div>
  </div>
}

// Фоновая генерация. Бэкенд отдаёт только статус задачи, поэтому текущий шаг оцениваем по времени.
export function Generating({ job, steps = GEN_STEPS, expected = 20, title = 'Собираем карточку для вас', inline = false }: { job?: JobView; steps?: string[]; expected?: number; title?: string; inline?: boolean }) {
  const [t, setT] = useState(0)
  useEffect(() => { const i = setInterval(() => setT(x => x + 1), 1000); return () => clearInterval(i) }, [])
  const step = Math.min(steps.length - 1, Math.floor((t / expected) * steps.length))
  const icons = [Search, Building2, Landmark, ShieldCheck, Bot]
  const box = <section className="generation surface" aria-live="polite">
    <div className="generation-head"><span className="spinner red" /><div><Eyebrow>{job?.status === 'queued' ? 'В очереди' : 'Генерация'}</Eyebrow><h2>{title}</h2><p>Обычно {expected <= 20 ? '10–30' : '20–60'} секунд. Можно вернуться в ленту — результат не пропадёт.</p></div><b className="generation-time">{t} с</b></div>
    <div className="tasks-progress"><i style={{ width: `${Math.min(95, ((step + 0.5) / steps.length) * 100)}%` }} /></div>
    <ol className="generation-steps">
      {steps.map((l, i) => {
        const st = i < step ? 'done' : i === step ? 'run' : 'wait'
        const Icon = icons[i] ?? Bot
        return <li key={l} className={st}><span>{st === 'done' ? <Check size={16} /> : <Icon size={16} />}</span><b>{l}</b><em>{st === 'done' ? 'готово' : st === 'run' ? 'идёт' : ''}</em></li>
      })}
    </ol>
    {!inline && <Link to="/feed" className="secondary-button"><ArrowLeft size={16} /> Вернуться в ленту</Link>}
  </section>
  if (inline) return box
  return <div className="insight-layout">
    <div className="story">
      {box}
      {['Что произошло', 'Почему это важно вам', 'Возможные последствия'].map((x, i) => <section key={x} className="story-section skeleton-section"><div className="section-number">0{i + 1}</div><div className="skel-stack"><Eyebrow>{x}</Eyebrow><Skel w="70%" h={22} /><Skel w="95%" /><Skel w="85%" /></div></section>)}
    </div>
    <aside className="insight-toc surface"><b>В карточке</b>{['Позиция банка', 'Вывод модели', 'Источники'].map(x => <div key={x} className="toc-wait"><Skel w="80%" h={9} /><span>{x}</span></div>)}</aside>
  </div>
}

// Оценка «полезно / не полезно» с причиной (К08). Оценка эксперта учитывается отдельно.
export function Feedback({ card }: { card: InsightCard }) {
  const s = useSession()
  const [vote, setVote] = useState<'up' | 'down' | null>(null)
  const [reason, setReason] = useState<FeedbackReason | null>(null)
  const [comment, setComment] = useState('')
  const [sent, setSent] = useState(false)
  const [err, setErr] = useState<ApiError | null>(null)
  const expert = s.user?.access !== 'user'
  async function send(rating: 'up' | 'down', r?: FeedbackReason) {
    setErr(null)
    try { await api.feedback(card.card_id, { rating, reason: r, comment: comment.trim() || undefined }); setSent(true) } catch (e) { setErr(e as ApiError) }
  }

  return <section className="feedback">
    <div><b>{sent ? 'Спасибо, оценка сохранена' : 'Была ли карточка полезна?'}</b><span>{sent ? `Оценка привязана к трассе ${card.trace_id}` : expert ? 'Вы оцениваете как сотрудник банка — оценка учтётся отдельно' : 'Это поможет улучшить следующие выводы'}</span></div>
    {!sent && <div>
      <button className={vote === 'up' ? 'selected' : ''} onClick={() => { setVote('up'); send('up') }}><ThumbsUp /> Да</button>
      <button className={vote === 'down' ? 'selected negative-select' : ''} onClick={() => setVote('down')}><ThumbsDown /> Нет</button>
    </div>}
    {sent && <small><CircleCheck size={14} /> Учтём в мониторинге качества</small>}
    {vote === 'down' && !sent && <div className="feedback-reasons">
      <span>Что не так?</span>
      <div>{(Object.keys(REASON_LABEL) as FeedbackReason[]).map(r => <button key={r} className={reason === r ? 'selected negative-select' : ''} onClick={() => setReason(r)}>{REASON_LABEL[r]}</button>)}</div>
      <textarea id="feedback-comment" maxLength={1000} value={comment} onChange={e => setComment(e.target.value)} placeholder="Комментарий, необязательно" rows={2} />
      <button className="send" disabled={!reason} onClick={() => send('down', reason!)}>Отправить</button>
    </div>}
    {err && <ErrorBanner error={err} />}
    {expert && !sent && <Pill tone="ai">Оценка эксперта</Pill>}
  </section>
}
