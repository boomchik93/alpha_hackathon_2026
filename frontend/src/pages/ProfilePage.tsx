import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, BadgeCheck, Building2, CircleAlert, Layers3, ListChecks, ShieldCheck } from 'lucide-react'
import { useCompanies, useCompany, useFeed } from '../api/hooks'
import { useSession } from '../auth/session'
import { ErrorBanner, PageHeader, Pill, Skel } from '../components/ui'
import type { BankProducts, CompanyProfile } from '../api/types'
import { companyLabel, featureLabel, featureValue, fmtRub } from '../dict/labels'

const PRODUCTS: (keyof BankProducts)[] = ['rko', 'loan', 'leasing', 'deposit', 'guarantee', 'payroll', 'factoring']
const FEATURES = ['industry.okved_code', 'size.revenue_band', 'size.workers_band', 'geo.foreign_trade', 'geo.fts_countries', 'network.counterparty_sectors', 'network.known_share']
const yn = (v: boolean | null) => (v === null ? 'нет данных' : v ? 'есть' : 'нет')
const geoText = (c: CompanyProfile) => [c.geo.city, c.geo.region].filter((x, i, a) => x && a.indexOf(x) === i).join(', ') || 'нет данных'

export default function ProfilePage() {
  const s = useSession()
  const company = useCompany(s.companyId)
  const companies = useCompanies(s.user?.login)
  const feed = useFeed(s.companyId)
  const c = company.data
  const isGroup = c?.level === 'group'
  const main = feed.data?.entries.filter(e => e.tier === 'main') ?? []
  const reasons = useMemo(() => {
    const m = new Map<string, number>()
    main.forEach(e => e.reasons.forEach(r => m.set(r, (m.get(r) ?? 0) + 1)))
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)
  }, [main])
  const gaps = c ? c.data_quality.missing_fields.length + c.data_quality.conflicts.length : 0
  const quality = Math.max(35, 96 - gaps * 15)

  return <div className="page">
    <PageHeader eyebrow="Профиль влияния" title={c ? companyLabel(c) : '…'} subtitle="По этим признакам мы отбираем события и объясняем их влияние именно на ваш бизнес." />
    {company.error && <ErrorBanner error={company.error as never} />}
    {c && gaps > 0 && <div className="warning-banner"><CircleAlert /><div><b>В профиле есть пробелы</b>{c.data_quality.notes.map(w => <p key={w}>{w}.</p>)}<p>Не заполнено: {c.data_quality.missing_fields.map(featureLabel).join(', ')}. Карточки покажут это в блоке «Ограничения» и не будут делать выводов по этим полям.</p></div></div>}
    {!c && <div className="surface profile-hero"><Skel w={68} h={68} /><div className="skel-stack"><Skel w="40%" h={24} /><Skel w="30%" /></div></div>}
    {c && <div className="profile-grid">
      <section className="profile-hero surface">
        <div className="company-monogram">{(c.industry.okved_name ?? 'К')[0]}</div>
        <div><Pill tone="success"><BadgeCheck size={13} /> {isGroup ? 'Группа компаний' : c.client_type === 'PROSPECT' ? 'Потенциальный клиент' : 'Клиент банка'}</Pill><h2>{c.industry.okved_name ?? 'Отрасль не указана'}</h2><p>ОКВЭД {c.industry.okved_code ?? '—'}</p></div>
        <div className="quality"><span>Полнота данных профиля</span><b>{gaps ? 'Есть пробелы' : 'Высокая'}</b><div><i style={{ width: `${quality}%`, background: gaps ? '#d79b13' : undefined }} /></div></div>
      </section>
      <div className="profile-col">
      <section className="surface info-card"><div className="info-title"><span><Building2 /></span><h3>{isGroup ? 'Группа' : 'Компания'}</h3></div>
        {[['Выручка', fmtRub(c.size.revenue_rub)], ['Размер', c.size.revenue_band ?? 'нет данных'], ['Сотрудники', c.size.workers_band ?? 'нет данных'], ['География', geoText(c)], ...(isGroup ? [['Выручка группы', fmtRub(c.group.group_revenue_rub)]] : []), ...(c.market_position ? [['Положение в отрасли', c.market_position.note]] : [])].map(([a, b]) => <div className="info-row" key={a}><span>{a}</span><b>{b}</b></div>)}
      </section>
      <section className="surface info-card"><div className="info-title"><span><Layers3 /></span><h3>Почему события попали в ленту</h3></div>
        {feed.isLoading && <Skel w="80%" />}
        {reasons.map(([r, n]) => <div className="bar-row" key={r}><span>{r}</span><div><i style={{ width: `${(n / Math.max(1, main.length)) * 100}%` }} /></div><b>{n} из {main.length}</b></div>)}
        <Link to="/feed" className="trace-link">Перейти в ленту <ArrowRight /></Link>
      </section>
      </div>
      <div className="profile-col">
      <section className="surface info-card"><div className="info-title"><span><ListChecks /></span><h3>Продукты банка</h3></div>
        {PRODUCTS.map(k => <div className="info-row" key={k}><span>{featureLabel(`bank_products.${k}`)}</span><b>{yn(c.bank_products[k])}</b></div>)}
      </section>
      <section className="surface info-card"><div className="info-title"><span><ListChecks /></span><h3>Признаки для ленты</h3></div>
        {FEATURES.map(f => <div className="info-row" key={f}><span>{featureLabel(f)}</span><b>{featureValue(c, f) ?? 'нет данных'}</b></div>)}
      </section>
      {isGroup && <section className="surface info-card"><div className="info-title"><span><Layers3 /></span><h3>Компании группы</h3></div>
        {c.member_ids.map((id, index) => { const member = companies.data?.find(x => x.company_id === id); return <div className="info-row" key={id}><span>{member ? companyLabel(member) : `Компания группы ${index + 1}`}</span>{member ? <button className="text-button link-red" onClick={() => s.setCompanyId(id)}>Открыть профиль</button> : <b>учтена в группе</b>}</div> })}
      </section>}
      <section className="surface data-use"><ShieldCheck /><div><h3>Как используются данные</h3><p>Модель получает профиль только этой компании, работает в режиме чтения и не видит реквизиты и персональные данные. Данные банка обезличены.</p></div></section>
      </div>
    </div>}
  </div>
}
