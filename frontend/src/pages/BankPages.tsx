import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft, Building2, Check, Clock3, Landmark, Lightbulb, ShieldCheck, TrendingUp, WalletCards } from 'lucide-react'
import { Eyebrow, PageHeader, Pill } from '../components/ui'
import { TASKS } from './HomePage'

// Антураж интернет-банка: счета и задачи. Данные выдуманы, в прототипе не связаны с API.
export function FinancesPage() {
  const navigate = useNavigate()
  const accounts = [
    ['Расчётный счёт · RUB', '•• 4417', '1 102 340 120 ₽', '+ 48 600 000 ₽'],
    ['Валютный счёт · USD', '•• 9031', '1 874 210,55 $', '+ 1 240 000 $'],
    ['Валютный счёт · TRY', '•• 9058', '42 118 900 ₺', '− 3 100 000 ₺'],
  ]
  return <div className="page finance-page">
    <button className="back-button" onClick={() => navigate('/')}><ArrowLeft size={18} /> На главную</button>
    <PageHeader eyebrow="Финансы" title="Счета и остатки" subtitle="Движение денег и доступная ликвидность компании." />
    <section className="finance-summary"><div><span>Доступно сейчас, в рублёвом эквиваленте</span><b>1 284 560 312 ₽</b><small><TrendingUp /> +3,1% за 30 дней</small></div><div className="finance-orbit"><WalletCards /><span>3 счёта</span></div></section>
    <div className="finance-grid">
      <section className="surface cashflow-card"><div className="section-title"><div><Eyebrow>Денежный поток</Eyebrow><h3>Поступления и списания</h3></div><Pill tone="success">+38,4 млн ₽</Pill></div><div className="cashflow-chart" aria-label="График денежного потока">{[32, 46, 39, 62, 55, 73, 49, 67, 82, 64, 91, 78].map((h, i) => <i key={i} style={{ height: `${h}%` }} className={i === 10 ? 'peak' : ''} />)}</div><div className="chart-legend"><span><i className="income" />Поступления · 412,6 млн ₽</span><span><i />Списания · 374,2 млн ₽</span></div></section>
      <section className="surface accounts-list"><div className="section-title"><div><Eyebrow>Счета</Eyebrow><h3>Текущие остатки</h3></div></div>{accounts.map(([name, num, balance, flow]) => <div className="account-row" key={num}><span className="account-icon"><Landmark /></span><div><b>{name}</b><small>{num}</small></div><strong>{balance}</strong><em className={flow.startsWith('+') ? 'positive' : ''}>{flow}</em></div>)}</section>
    </div>
  </div>
}

export function TasksPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const focus = Number(new URLSearchParams(location.search).get('focus') ?? -1)
  const [done, setDone] = useState<number[]>([])
  const [tab, setTab] = useState<'active' | 'done'>('active')
  const icons = [<WalletCards key="w" />, <ShieldCheck key="s" />, <TrendingUp key="t" />, <Building2 key="b" />]
  const toggle = (i: number) => setDone(c => (c.includes(i) ? c.filter(x => x !== i) : [...c, i]))
  const visible = TASKS.map((t, i) => ({ t, i })).filter(({ i }) => (tab === 'done' ? done.includes(i) : !done.includes(i)))
  return <div className="page tasks-page">
    <button className="back-button" onClick={() => navigate('/')}><ArrowLeft size={18} /> На главную</button>
    <PageHeader eyebrow="На контроле" title="Ближайшие действия" subtitle="Задачи, которые требуют внимания финансовой команды." action={<Pill tone="red">{TASKS.length - done.length} активных</Pill>} />
    <div className="tasks-layout">
      <section className="surface tasks-list">
        <div className="tasks-tabs"><button className={tab === 'active' ? 'active' : ''} onClick={() => setTab('active')}>Активные <span>{TASKS.length - done.length}</span></button><button className={tab === 'done' ? 'active' : ''} onClick={() => setTab('done')}>Завершённые <span>{done.length}</span></button></div>
        {visible.length === 0 && <p className="empty-note">{tab === 'done' ? 'Пока ничего не завершено.' : 'Все задачи выполнены.'}</p>}
        {visible.map(({ t, i }) => <article className={`task-detail ${focus === i ? 'focused' : ''} ${done.includes(i) ? 'completed' : ''}`} key={t.title}>
          <button className="task-check" onClick={() => toggle(i)} aria-label={done.includes(i) ? 'Вернуть задачу' : 'Завершить задачу'}>{done.includes(i) && <Check />}</button>
          <span className={`task-detail-icon ${t.tone}`}>{icons[i]}</span>
          <div><b>{t.title}</b><p>{t.note}</p><small><Clock3 /> {t.date}</small></div>
          <Pill tone={t.tone === 'urgent' ? 'red' : 'neutral'}>{t.tone === 'urgent' ? 'Сегодня' : 'Запланировано'}</Pill>
        </article>)}
      </section>
      <aside className="surface tasks-side"><Eyebrow>Прогресс</Eyebrow><h3>{done.length} из {TASKS.length}</h3><div className="tasks-progress"><i style={{ width: `${(done.length / TASKS.length) * 100}%` }} /></div><p>Завершите приоритетные задачи, чтобы снять риски по платёжному календарю.</p><div className="tasks-tip"><Lightbulb /><span>Начните с платежей со сроком сегодня.</span></div></aside>
    </div>
  </div>
}
