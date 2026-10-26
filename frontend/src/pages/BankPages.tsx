import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Landmark, TrendingUp, WalletCards } from 'lucide-react'
import { Eyebrow, PageHeader, Pill } from '../components/ui'

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
      <section className="surface cashflow-card"><div className="section-title"><div><Eyebrow>Денежный поток</Eyebrow><h3>Поступления и списания</h3></div><Pill tone="success">+38,4 млн ₽</Pill></div><div className="cashflow-chart" aria-label="Красные столбцы — поступления, серые — списания">{[32, 46, 39, 62, 55, 73, 49, 67, 82, 64, 91, 78].map((h, i) => { const type = i % 2 === 0 ? 'income' : 'expense'; return <i key={i} style={{ height: `${h}%` }} className={`${type}${i === 10 ? ' peak' : ''}`} title={type === 'income' ? 'Поступления' : 'Списания'} /> })}</div><div className="chart-legend"><span><i className="income" />Поступления · 412,6 млн ₽</span><span><i className="expense" />Списания · 374,2 млн ₽</span></div></section>
      <section className="surface accounts-list"><div className="section-title"><div><Eyebrow>Счета</Eyebrow><h3>Текущие остатки</h3></div></div>{accounts.map(([name, num, balance, flow]) => <div className="account-row" key={num}><span className="account-icon"><Landmark /></span><div><b>{name}</b><small>{num}</small></div><strong>{balance}</strong><em className={flow.startsWith('+') ? 'positive' : ''}>{flow}</em></div>)}</section>
    </div>
  </div>
}

