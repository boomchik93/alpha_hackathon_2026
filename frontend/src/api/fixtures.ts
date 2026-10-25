// Демо-данные в формате контракта (ml/schemas.py). Компании и пользователи совпадают с backend/seed_users.json
// и выборкой ml/eval/companies.json, новости — с ml/fixtures/news.jsonl. Тексты карточек и сценариев написаны для демо.
import type { BankProducts, Claim, CompanyProfile, FeedEntry, Me, Role, ScenarioBranch, ScenarioDecision, ScenarioFork, SourceRef, SuggestedAction } from './types'

// ---------- пользователи (backend/seed_users.json) ----------
export interface SeedUser extends Me { password: string }
export const USERS: SeedUser[] = [
  { login: 'retail.cfo', password: 'demo-cfo-1595', display_name: 'Финансовый директор, розница одеждой', role: 'cfo', access: 'user', companies: ['U_1595'] },
  { login: 'retail.owner', password: 'demo-owner-1595', display_name: 'Собственник, розница одеждой', role: 'owner', access: 'user', companies: ['U_1595'] },
  { login: 'retail.treasurer', password: 'demo-treasurer-1595', display_name: 'Казначей, розница одеждой', role: 'treasurer', access: 'user', companies: ['U_1595'] },
  { login: 'fuel.ceo', password: 'demo-ceo-26030', display_name: 'Генеральный директор, опт моторным топливом', role: 'ceo', access: 'user', companies: ['U_26030'] },
  { login: 'logistics.cfo', password: 'demo-cfo-3263', display_name: 'Финансовый директор, грузоперевозки', role: 'cfo', access: 'user', companies: ['U_3263'] },
  { login: 'build.accountant', password: 'demo-acc-183405', display_name: 'Главный бухгалтер, опт стройматериалами (данные с пропусками)', role: 'chief_accountant', access: 'user', companies: ['U_183405'] },
  { login: 'group.cfo', password: 'demo-cfo-group', display_name: 'Финансовый директор группы, бетонные смеси', role: 'cfo', access: 'user', companies: ['group_134463_sb'] },
  { login: 'expert', password: 'demo-expert', display_name: 'Эксперт банка', role: 'other', access: 'expert', companies: [] },
  { login: 'admin', password: 'demo-admin', display_name: 'Администратор', role: 'other', access: 'admin', companies: [] },
]

// ---------- профили ----------
const products = (p: Partial<BankProducts>): BankProducts => ({ rko: null, guarantee: null, loan: null, leasing: null, deposit: null, payroll: null, factoring: null, ...p })
const empty = { group: { group_id: null, group_size: null, group_revenue_rub: null, group_clients: null }, member_ids: [] as string[], org_type: 'ООО', ownership: 'частная', client_type: 'CLIENT', stage: 'Зрелая' }

export const PROFILES: Record<string, CompanyProfile> = {
  U_1595: {
    ...empty, company_id: 'U_1595', level: 'company', eligible_big: true,
    industry: { okved_code: '47.71', okved_name: 'Торговля розничная одеждой в специализированных магазинах', okved_section: '47', sector_tags: ['розница', 'одежда'] },
    size: { revenue_rub: 21.4e9, revenue_band: 'крупный', workers_band: '1001–5000', age_band: '10+ лет' },
    geo: { city: 'Москва', region: 'Москва', foreign_trade: true, fts_countries: ['Турция', 'Нидерланды'] },
    bank_products: products({ rko: true, loan: true, leasing: true, payroll: true, deposit: false, guarantee: false, factoring: false }),
    network: { top_counterparties: [{ id: 'U_88120', role: 'beneficiary', amt_rub: 2.1e9, cnt: 140, known: true, okved_name: 'Торговля оптовая одеждой' }, { id: 'U_40231', role: 'beneficiary', amt_rub: 0.9e9, cnt: 36, known: true, okved_name: 'Аренда и управление недвижимостью' }], counterparty_sectors: ['Торговля оптовая одеждой', 'Аренда недвижимости', 'Логистика'], known_share: 0.64, note: null },
    data_quality: { missing_fields: [], conflicts: [], notes: [] },
    market_position: { okved_prefix: '47.71', digits: 4, peers: 41, rank: 3, percentile: 0.95, revenue_share: 0.12, note: 'по выручке 3-я из 41 компании выборки банка с кодом ОКВЭД 47.71' },
  },
  U_26030: {
    ...empty, company_id: 'U_26030', level: 'company', eligible_big: true,
    industry: { okved_code: '46.71', okved_name: 'Торговля оптовая моторным топливом, включая авиационный бензин', okved_section: '46', sector_tags: ['опт', 'топливо'] },
    size: { revenue_rub: 64.2e9, revenue_band: 'крупный', workers_band: '251–1000', age_band: '10+ лет' },
    geo: { city: 'Екатеринбург', region: 'Свердловская область', foreign_trade: true, fts_countries: ['Казахстан', 'Беларусь'] },
    bank_products: products({ rko: true, loan: true, leasing: true, deposit: true, guarantee: true, payroll: false, factoring: false }),
    network: { top_counterparties: [{ id: 'U_51002', role: 'payer', amt_rub: 6.8e9, cnt: 220, known: true, okved_name: 'Деятельность автомобильного грузового транспорта' }], counterparty_sectors: ['Грузоперевозки', 'Добыча и переработка нефти', 'Сельское хозяйство'], known_share: 0.71, note: null },
    data_quality: { missing_fields: [], conflicts: [], notes: [] },
    market_position: { okved_prefix: '46.71', digits: 4, peers: 58, rank: 6, percentile: 0.91, revenue_share: 0.07, note: 'по выручке 6-я из 58 компаний выборки банка с кодом ОКВЭД 46.71' },
  },
  U_3263: {
    ...empty, company_id: 'U_3263', level: 'company', eligible_big: true,
    industry: { okved_code: '49.41', okved_name: 'Перевозка грузов неспециализированными автотранспортными средствами', okved_section: '49', sector_tags: ['логистика'] },
    size: { revenue_rub: 16.9e9, revenue_band: 'крупный', workers_band: '1001–5000', age_band: '10+ лет' },
    geo: { city: 'Пермь', region: 'Пермский край', foreign_trade: true, fts_countries: ['Казахстан', 'Китай'] },
    bank_products: products({ rko: true, leasing: true, loan: false, deposit: false, payroll: true, guarantee: false, factoring: false }),
    network: { top_counterparties: [{ id: 'U_26030', role: 'beneficiary', amt_rub: 1.4e9, cnt: 310, known: true, okved_name: 'Торговля оптовая моторным топливом' }], counterparty_sectors: ['Топливо', 'Розничная торговля', 'Производство стройматериалов'], known_share: 0.58, note: null },
    data_quality: { missing_fields: [], conflicts: [], notes: [] },
    market_position: { okved_prefix: '49.41', digits: 4, peers: 73, rank: 9, percentile: 0.88, revenue_share: 0.04, note: 'по выручке 9-я из 73 компаний выборки банка с кодом ОКВЭД 49.41' },
  },
  U_183405: {
    ...empty, company_id: 'U_183405', level: 'company', eligible_big: true, client_type: 'PROSPECT', stage: null,
    industry: { okved_code: '46.73', okved_name: 'Торговля оптовая лесоматериалами, строительными материалами и санитарно-техническим оборудованием', okved_section: '46', sector_tags: ['опт', 'стройматериалы'] },
    size: { revenue_rub: 15.6e9, revenue_band: 'крупный', workers_band: null, age_band: null },
    geo: { city: null, region: 'Краснодарский край', foreign_trade: null, fts_countries: [] },
    bank_products: products({}),
    network: { top_counterparties: [], counterparty_sectors: [], known_share: null, note: 'платежей через банк нет' },
    data_quality: { missing_fields: ['bank_products', 'network.top_counterparties', 'size.workers_band', 'geo.foreign_trade'], conflicts: [], notes: ['Нет данных о продуктах банка и платежах: выводы опираются только на отрасль, регион и размер'] },
    market_position: null,
  },
  group_134463_sb: {
    ...empty, company_id: 'group_134463_sb', level: 'group', eligible_big: true, member_ids: ['U_134463', 'U_134471'],
    group: { group_id: 'group_134463_sb', group_size: 2, group_revenue_rub: 38.5e9, group_clients: 2 },
    industry: { okved_code: '23.64', okved_name: 'Производство сухих бетонных смесей', okved_section: '23', sector_tags: ['стройматериалы', 'производство'] },
    size: { revenue_rub: 38.5e9, revenue_band: 'крупный', workers_band: '1001–5000', age_band: '10+ лет' },
    geo: { city: 'Новосибирск', region: 'Новосибирская область', foreign_trade: true, fts_countries: ['Узбекистан', 'Казахстан'] },
    bank_products: products({ rko: true, loan: true, leasing: true, deposit: true, guarantee: true, payroll: true, factoring: true }),
    network: { top_counterparties: [{ id: 'U_77310', role: 'payer', amt_rub: 4.2e9, cnt: 95, known: true, okved_name: 'Строительство жилых и нежилых зданий' }], counterparty_sectors: ['Строительство', 'Добыча песка и гравия', 'Грузоперевозки'], known_share: 0.69, note: null },
    data_quality: { missing_fields: [], conflicts: [], notes: [] },
    market_position: { okved_prefix: '23.6', digits: 3, peers: 27, rank: 2, percentile: 0.96, revenue_share: 0.21, note: 'по выручке 2-я из 27 компаний и групп выборки банка с кодом ОКВЭД 23.6' },
  },
  U_134463: {
    ...empty, company_id: 'U_134463', level: 'company', eligible_big: true, group: { group_id: 'group_134463_sb', group_size: 2, group_revenue_rub: 38.5e9, group_clients: 2 },
    industry: { okved_code: '23.64', okved_name: 'Производство сухих бетонных смесей', okved_section: '23', sector_tags: ['стройматериалы'] },
    size: { revenue_rub: 29.1e9, revenue_band: 'крупный', workers_band: '1001–5000', age_band: '10+ лет' },
    geo: { city: 'Новосибирск', region: 'Новосибирская область', foreign_trade: true, fts_countries: ['Узбекистан'] },
    bank_products: products({ rko: true, loan: true, leasing: true, deposit: true, guarantee: true, payroll: true, factoring: false }),
    network: { top_counterparties: [], counterparty_sectors: ['Строительство', 'Добыча песка и гравия'], known_share: 0.66, note: null },
    data_quality: { missing_fields: [], conflicts: [], notes: [] },
    market_position: null,
  },
  U_134471: {
    ...empty, company_id: 'U_134471', level: 'company', eligible_big: false, group: { group_id: 'group_134463_sb', group_size: 2, group_revenue_rub: 38.5e9, group_clients: 2 },
    industry: { okved_code: '49.41', okved_name: 'Перевозка грузов неспециализированными автотранспортными средствами', okved_section: '49', sector_tags: ['логистика'] },
    size: { revenue_rub: 9.4e9, revenue_band: 'средний', workers_band: '251–1000', age_band: '5–10 лет' },
    geo: { city: 'Новосибирск', region: 'Новосибирская область', foreign_trade: false, fts_countries: [] },
    bank_products: products({ rko: true, leasing: true, factoring: true }),
    network: { top_counterparties: [], counterparty_sectors: ['Производство стройматериалов'], known_share: 0.81, note: null },
    data_quality: { missing_fields: [], conflicts: [], notes: [] },
    market_position: null,
  },
}

// ---------- материалы ----------
interface Item { title: string; published_at: string; source_kind: 'news' | 'bank_analytics'; topics: string[]; publisher: string; url?: string; doc_id?: string; page?: number; related?: string[] }
export const ITEMS: Record<string, Item> = {
  n_2026_09_11_cbr_rate: { title: 'Банк России сохранил ключевую ставку на уровне 14%', published_at: '2026-09-11', source_kind: 'news', topics: ['key_rate'], publisher: 'Интерфакс', url: 'https://www.cbr.ru/press/keypr/', related: ['n_2026_09_11_cbr_rate_dup'] },
  a_macro_2026_09: { title: 'Макроэкономический обзор: ставка, инфляция и курс до конца 2026 года', published_at: '2026-09-15', source_kind: 'bank_analytics', topics: ['key_rate', 'inflation'], publisher: 'Альфа-Банк', doc_id: 'alfa_macro_2026_09', page: 1 },
  n_2026_09_14_ofz: { title: 'Доходности ОФЗ выросли после решения ЦБ', published_at: '2026-09-14', source_kind: 'news', topics: ['bonds', 'key_rate'], publisher: 'РБК', url: 'https://www.rbc.ru/finances/' },
  n_2026_09_08_retail: { title: 'Рост розничных продаж замедлился', published_at: '2026-09-08', source_kind: 'news', topics: ['consumption'], publisher: 'Ведомости', url: 'https://www.vedomosti.ru/economics' },
  n_2026_09_09_wages: { title: 'Дефицит кадров поддерживает рост зарплат', published_at: '2026-09-09', source_kind: 'news', topics: ['labor'], publisher: 'Коммерсантъ', url: 'https://www.kommersant.ru/rubric/3' },
  n_2026_09_03_petrol: { title: 'Оптовые цены на бензин обновили максимум', published_at: '2026-09-03', source_kind: 'news', topics: ['fuel'], publisher: 'Интерфакс', url: 'https://www.interfax.ru/business/' },
  n_2026_09_10_trucking: { title: 'Перевозчики сообщают о росте тарифов на грузоперевозки', published_at: '2026-09-10', source_kind: 'news', topics: ['fuel', 'consumption'], publisher: 'РБК', url: 'https://www.rbc.ru/business/' },
  n_2026_09_11_steel: { title: 'Спрос на стальной прокат снижается на фоне паузы в стройке', published_at: '2026-09-11', source_kind: 'news', topics: ['steel'], publisher: 'Интерфакс', url: 'https://www.interfax.ru/business/' },
  n_2026_09_12_gold: { title: 'Золото обновило рекорд на ожиданиях снижения ставки ФРС', published_at: '2026-09-12', source_kind: 'news', topics: ['fx_gold'], publisher: 'РБК', url: 'https://www.rbc.ru/finances/' },
  n_2026_09_13_rate_cut_rumor: { title: 'Источник: ЦБ может снизить ставку уже в октябре', published_at: '2026-09-13', source_kind: 'news', topics: ['key_rate'], publisher: 'Телеграм-канал', url: 'https://t.me/' },
  n_2026_09_06_noisy: { title: 'Промышленность: ВВП в I полугодии 2026 года вырос на 0,1%', published_at: '2026-09-06', source_kind: 'news', topics: ['gdp_industry'], publisher: 'Росстат', url: 'https://rosstat.gov.ru/' },
}

type Entry = [itemId: string, score: number, reasons: string[], tier?: 'main' | 'maybe', stale?: boolean]
const FEEDS_RAW: Record<string, { filtered: number; entries: Entry[] }> = {
  U_1595: { filtered: 47, entries: [
    ['n_2026_09_11_cbr_rate', 0.91, ['есть кредит и лизинг в банке', 'ставка задаёт стоимость долга']],
    ['n_2026_09_08_retail', 0.86, ['ваша отрасль: розничная торговля одеждой', 'спрос покупателей']],
    ['a_macro_2026_09', 0.78, ['аналитика банка по ставке и курсу', 'закупки в Турции и Нидерландах']],
    ['n_2026_09_09_wages', 0.71, ['более 1000 сотрудников', 'зарплатный проект в банке']],
    ['n_2026_09_14_ofz', 0.63, ['заёмное финансирование', 'ставки по новым кредитам']],
    ['n_2026_09_12_gold', 0.41, ['есть внешняя торговля: Турция, Нидерланды'], 'maybe'],
    ['n_2026_09_13_rate_cut_rumor', 0.38, ['есть кредит в банке', 'источник не подтверждён'], 'maybe'],
    ['n_2026_09_06_noisy', 0.33, ['общая экономическая ситуация'], 'maybe', true],
  ] },
  U_26030: { filtered: 52, entries: [
    ['n_2026_09_03_petrol', 0.93, ['ваша отрасль: опт моторным топливом', 'цены закупки и продажи']],
    ['n_2026_09_11_cbr_rate', 0.84, ['есть кредит и лизинг в банке', 'оборотное финансирование запасов']],
    ['a_macro_2026_09', 0.70, ['аналитика банка по ставке и курсу']],
    ['n_2026_09_10_trucking', 0.62, ['ключевые покупатели — перевозчики']],
    ['n_2026_09_14_ofz', 0.45, ['заёмное финансирование'], 'maybe'],
    ['n_2026_09_12_gold', 0.40, ['есть внешняя торговля: Казахстан, Беларусь'], 'maybe'],
  ] },
  U_3263: { filtered: 55, entries: [
    ['n_2026_09_10_trucking', 0.92, ['ваша отрасль: грузоперевозки', 'тарифы и спрос на перевозки']],
    ['n_2026_09_03_petrol', 0.88, ['топливо — главная статья затрат перевозчика']],
    ['n_2026_09_11_cbr_rate', 0.74, ['есть лизинг в банке', 'обновление автопарка']],
    ['n_2026_09_09_wages', 0.66, ['более 1000 сотрудников', 'дефицит водителей']],
    ['a_macro_2026_09', 0.44, ['аналитика банка: общий фон'], 'maybe'],
  ] },
  U_183405: { filtered: 61, entries: [
    ['n_2026_09_11_steel', 0.79, ['ваша отрасль: опт стройматериалами', 'спрос со стороны стройки']],
    ['n_2026_09_11_cbr_rate', 0.58, ['отрасль чувствительна к ставке через спрос застройщиков']],
    ['n_2026_09_08_retail', 0.35, ['косвенно: спрос на ремонт'], 'maybe'],
  ] },
  group_134463_sb: { filtered: 44, entries: [
    ['n_2026_09_11_steel', 0.88, ['ваша отрасль: производство бетонных смесей', 'покупатели — застройщики']],
    ['n_2026_09_11_cbr_rate', 0.87, ['все продукты банка, включая кредит и лизинг', 'ипотека и спрос на жильё']],
    ['a_macro_2026_09', 0.73, ['аналитика банка по ставке и стройке']],
    ['n_2026_09_09_wages', 0.61, ['более 1000 сотрудников в группе']],
    ['n_2026_09_14_ofz', 0.46, ['заёмное финансирование'], 'maybe'],
    ['n_2026_09_03_petrol', 0.42, ['доставка смесей собственным транспортом группы'], 'maybe'],
  ] },
}
FEEDS_RAW.U_134463 = FEEDS_RAW.group_134463_sb
FEEDS_RAW.U_134471 = { filtered: 58, entries: [['n_2026_09_03_petrol', 0.9, ['ваша отрасль: грузоперевозки', 'топливо — главная статья затрат']], ['n_2026_09_11_steel', 0.64, ['перевозите стройматериалы группы']]] }

export function feedEntries(companyId: string): { entries: FeedEntry[]; filtered: number } {
  const raw = FEEDS_RAW[companyId] ?? { filtered: 0, entries: [] }
  return {
    filtered: raw.filtered,
    entries: raw.entries.map(([id, score, reasons, tier = 'main', stale = false]) => {
      const it = ITEMS[id]
      return { item_id: id, title: it.title, published_at: it.published_at, source_kind: it.source_kind, score, reasons, topics: it.topics, stale, tier, related: it.related ?? [] }
    }),
  }
}

// ---------- содержимое карточек ----------
const claim = (text: string, horizon: string | null, evidence: string[] = []): Claim => ({ text, horizon, evidence, is_model_inference: evidence.length === 0 })
const act = (text: string, evidence: string[] = [], type: SuggestedAction['type'] = 'action'): SuggestedAction => ({ ...claim(text, null, evidence), type })
const src = (id: string, s: Partial<SourceRef> & Pick<SourceRef, 'kind' | 'title'>): SourceRef => ({ id, url: null, published_at: null, doc_id: null, page: null, quote: null, stale: false, ...s })
const MACRO = { kind: 'bank_analytics' as const, title: 'Макроэкономический обзор Альфа-Банка, сентябрь 2026', doc_id: 'alfa_macro_2026_09', published_at: '2026-09-15' }

export interface CardContent {
  what: string
  why: (p: CompanyProfile) => string
  features: (p: CompanyProfile) => string[]
  risks: Claim[]
  opps: Claim[]
  actions: SuggestedAction[]
  bank: { source_id: string; text: string } | null
  ai: string
  sources: SourceRef[]
  limitations: (p: CompanyProfile) => string[]
}
const has = (p: CompanyProfile, k: keyof BankProducts) => p.bank_products[k] === true
const debtFeatures = (p: CompanyProfile) => {
  const f = (['loan', 'leasing', 'deposit'] as const).filter(k => has(p, k)).map(k => `bank_products.${k}`)
  return f.length ? f : ['industry.okved_name', 'size.revenue_rub']
}

export const CARDS: Record<string, CardContent> = {
  n_2026_09_11_cbr_rate: {
    what: 'Банк России 11 сентября 2026 года сохранил ключевую ставку на уровне 14% годовых. Это первая пауза после серии снижений, начавшейся в июне 2025 года. Дальнейшие решения регулятор связал с динамикой инфляции и инфляционных ожиданий.',
    why: p => has(p, 'loan') || has(p, 'leasing')
      ? `У компании есть ${[has(p, 'loan') && 'кредит', has(p, 'leasing') && 'лизинг'].filter(Boolean).join(' и ')} в банке, поэтому ставка напрямую задаёт стоимость обслуживания долга. Пауза откладывает удешевление кредитов, на которое можно было рассчитывать при планировании 2027 года.`
      : 'Продуктов банка в профиле нет, поэтому прямого эффекта на стоимость долга не видно. Ставка влияет на компанию через покупателей: застройщики откладывают новые проекты, пока кредиты дорогие.',
    features: debtFeatures,
    risks: [claim('Рефинансирование долга в ближайшие месяцы останется дорогим', '3–6 месяцев', ['s1', 's2']), claim('Платежи по договорам с плавающей ставкой не снизятся до весны', 'до весны 2027')],
    opps: [claim('Свободные деньги можно разместить на депозите под высокую ставку', '1–3 месяца'), claim('Зафиксировать условия новых траншей до следующего заседания ЦБ', 'до 24 октября', ['s2'])],
    actions: [act('Заложить в бюджет 2027 года ставку 13–14% как базовый сценарий', ['s2']), act('Проверить запас по ковенантам при текущей ставке'), act('Когда банк ожидает первое снижение ставки и насколько?', [], 'question_for_analysis')],
    bank: { source_id: 's2', text: 'Мы ожидаем, что ключевая ставка останется на уровне 14% как минимум до конца 2026 года; снижение возможно не раньше первого квартала 2027 года' },
    ai: 'Пауза не меняет направление цикла, но сдвигает ожидаемое удешевление долга примерно на квартал. Для компании главный эффект — не текущая ставка сама по себе, а более позднее окно для рефинансирования.',
    sources: [src('s1', { kind: 'news', title: 'Банк России сохранил ключевую ставку на уровне 14%', url: 'https://www.cbr.ru/press/keypr/', published_at: '2026-09-11' }), src('s2', { ...MACRO, page: 4, quote: 'ключевая ставка останется на уровне 14% как минимум до конца 2026 года' })],
    limitations: p => [p.bank_products.loan === null ? 'В профиле нет данных о продуктах банка: эффект на долг оценить нельзя' : 'В профиле нет сроков погашения кредитов, поэтому эффект дан без суммы'],
  },
  a_macro_2026_09: {
    what: 'Аналитики Альфа-Банка ожидают, что ключевая ставка останется на уровне 14% до конца 2026 года, инфляция замедлится к 5–6% к концу года, а курс рубля к доллару будет в диапазоне 95–100.',
    why: p => p.geo.foreign_trade ? `Компания ведёт внешнюю торговлю (${p.geo.fts_countries.join(', ')}), поэтому прогноз курса важен для закупок и цен, а прогноз ставки — для стоимости долга.` : 'Прогноз ставки и инфляции задаёт фон для бюджета компании на 2027 год.',
    features: p => (p.geo.foreign_trade ? ['geo.fts_countries'] : []).concat(debtFeatures(p)).slice(0, 3),
    risks: [claim('Ослабление рубля к концу года удорожит импортные закупки', '3–6 месяцев', ['s1'])],
    opps: [claim('Медленное снижение ставки в 2027 году удешевит новые кредиты', '6–12 месяцев', ['s1'])],
    actions: [act('Обновить валютный прогноз в бюджете на I квартал 2027', ['s1']), act('Какие инструменты хеджирования валюты банк предлагает для импортёров?', [], 'question_for_analysis')],
    bank: { source_id: 's1', text: 'Мы сохраняем прогноз курса в диапазоне 95–100 рублей за доллар до конца года при постепенном замедлении инфляции к 5–6%' },
    ai: 'Прогноз банка не предполагает резких движений, поэтому главный риск — не базовый сценарий, а его нарушение. Имеет смысл заложить в бюджет запас на курс выше 100.',
    sources: [src('s1', { ...MACRO, page: 2, quote: 'курс в диапазоне 95–100 рублей за доллар' })],
    limitations: () => ['Доля валютных закупок в профиле не указана'],
  },
  n_2026_09_14_ofz: {
    what: 'После решения ЦБ доходности длинных ОФЗ выросли до 16,3%. Рынок пересматривает ожидания по снижению ставки.',
    why: () => 'Доходности ОФЗ задают уровень ставок по новым корпоративным кредитам и облигациям. Рост доходностей означает, что новое финансирование подешевеет позже.',
    features: debtFeatures,
    risks: [claim('Ставки по новым кредитам не будут снижаться ближайшие месяцы', '3–6 месяцев', ['s1'])],
    opps: [claim('Высокие доходности позволяют выгодно разместить свободную ликвидность', '1–3 месяца')],
    actions: [act('Сравнить доходность депозитов и коротких ОФЗ для свободных денег')],
    bank: null,
    ai: 'Это продолжение решения ЦБ, а не отдельное событие: эффект для компании тот же — более позднее удешевление долга.',
    sources: [src('s1', { kind: 'news', title: 'Доходности ОФЗ выросли после решения ЦБ', url: 'https://www.rbc.ru/finances/', published_at: '2026-09-14' })],
    limitations: () => ['У банка нет отдельной аналитики по этой теме'],
  },
  n_2026_09_08_retail: {
    what: 'Рост оборота розничной торговли в августе замедлился до 1,2% год к году против 2,5% месяцем ранее. Сильнее всего замедлились непродовольственные товары.',
    why: p => p.industry.okved_section === '47' ? 'Компания продаёт одежду в рознице: замедление продаж непродовольственных товаров прямо бьёт по её выручке во втором полугодии.' : 'Покупатели компании зависят от розничного спроса, поэтому замедление дойдёт и до неё, но с задержкой.',
    features: () => ['industry.okved_name', 'size.revenue_rub'],
    risks: [claim('Выручка осеннего сезона может оказаться ниже плана', '1–3 месяца', ['s1']), claim('Запасы осенней коллекции придётся распродавать со скидками', '3–6 месяцев')],
    opps: [claim('Перенести часть закупок весенней коллекции на более поздний срок', '3–6 месяцев')],
    actions: [act('Пересмотреть план закупок весенней коллекции с учётом спроса'), act('Какой спрос в рознице банк ожидает в IV квартале?', [], 'question_for_analysis')],
    bank: { source_id: 's2', text: 'Рост реальных зарплат замедляется, что будет сдерживать потребительский спрос до конца года' },
    ai: 'Замедление совпадает с высокой ставкой: покупатели меньше берут в кредит и экономят на одежде. Восстановления до снижения ставки ждать не стоит.',
    sources: [src('s1', { kind: 'news', title: 'Рост розничных продаж замедлился', url: 'https://www.vedomosti.ru/economics', published_at: '2026-09-08' }), src('s2', { ...MACRO, page: 5, quote: 'будет сдерживать потребительский спрос до конца года' })],
    limitations: () => ['Нет данных о доле онлайн-продаж компании'],
  },
  n_2026_09_09_wages: {
    what: 'Компании в торговле, логистике и строительстве продолжают повышать зарплаты, чтобы удержать персонал. Безработица остаётся на историческом минимуме.',
    why: p => `В компании ${p.size.workers_band ?? 'много'} сотрудников, поэтому рост зарплат заметно увеличит фонд оплаты труда в 2027 году.`,
    features: p => ['size.workers_band', ...(has(p, 'payroll') ? ['bank_products.payroll'] : [])],
    risks: [claim('Фонд оплаты труда в 2027 году вырастет быстрее выручки', '6–12 месяцев', ['s1'])],
    opps: [claim('Удержать ключевых сотрудников за счёт нематериальных условий', '3–6 месяцев')],
    actions: [act('Заложить в бюджет 2027 года рост ФОТ на уровне рынка'), act('Как банк оценивает рост зарплат в отрасли в 2027 году?', [], 'question_for_analysis')],
    bank: { source_id: 's2', text: 'Дефицит кадров остаётся главным ограничением для роста производства и торговли' },
    ai: 'Рост зарплат вместе с замедлением спроса сжимает маржу. Это повод пересмотреть производительность, а не только бюджет.',
    sources: [src('s1', { kind: 'news', title: 'Дефицит кадров поддерживает рост зарплат', url: 'https://www.kommersant.ru/rubric/3', published_at: '2026-09-09' }), src('s2', { ...MACRO, page: 6, quote: 'Дефицит кадров остаётся главным ограничением' })],
    limitations: () => ['Нет данных о текущем фонде оплаты труда компании'],
  },
  n_2026_09_03_petrol: {
    what: 'Биржевые цены на бензин АИ-92 и АИ-95 обновили исторический максимум. Участники рынка связывают рост с ремонтами на НПЗ и высоким сезонным спросом.',
    why: p => p.industry.okved_code?.startsWith('46.71') ? 'Компания торгует топливом оптом: рост биржевых цен увеличивает стоимость закупки и потребность в оборотных средствах, а покупатели болезненно реагируют на рост отпускных цен.' : 'Топливо — одна из главных статей затрат компании, поэтому рост цен сразу сократит маржу перевозок.',
    features: p => ['industry.okved_name', ...(has(p, 'loan') ? ['bank_products.loan'] : has(p, 'leasing') ? ['bank_products.leasing'] : [])],
    risks: [claim('Рост потребности в оборотных средствах на закупку топлива', '1–3 месяца', ['s1']), claim('Покупатели могут сократить объёмы или перейти к конкурентам', '3–6 месяцев')],
    opps: [claim('Поставщики с запасами по старым ценам получают временное преимущество', '1–2 месяца')],
    actions: [act('Пересчитать лимит оборотного кредита под новые цены закупки', ['s1']), act('Обсудить с ключевыми покупателями формулу цены с привязкой к бирже')],
    bank: null,
    ai: 'Аналитики банка по топливу нет, поэтому оценка последствий — вывод модели. Ключевой вопрос — продлится ли рост цен дольше сезона ремонтов.',
    sources: [src('s1', { kind: 'news', title: 'Оптовые цены на бензин обновили максимум', url: 'https://www.interfax.ru/business/', published_at: '2026-09-03' })],
    limitations: () => ['У банка нет материала по этой теме', 'Нет данных о доле биржевых закупок компании'],
  },
  n_2026_09_10_trucking: {
    what: 'Транспортные компании сообщают о росте тарифов на грузоперевозки на 8–12% с начала года. Главные причины — топливо, зарплаты водителей и лизинг техники.',
    why: p => p.industry.okved_section === '49' ? 'Компания сама перевозчик: рост тарифов поддерживает выручку, но он лишь частично компенсирует рост затрат на топливо и водителей.' : 'Перевозчики — ключевые покупатели компании: при росте их затрат они будут торговаться за цену топлива.',
    features: () => ['industry.okved_name', 'network.counterparty_sectors'],
    risks: [claim('Затраты растут быстрее тарифов', '3–6 месяцев', ['s1'])],
    opps: [claim('Пересмотреть тарифы с клиентами, у которых контракты без индексации', '1–3 месяца', ['s1'])],
    actions: [act('Проверить, в каких договорах нет индексации тарифа')],
    bank: null,
    ai: 'Рост тарифов говорит о том, что рынок перевозок пока выдерживает рост цен. Окно для пересмотра условий с клиентами открыто, но ненадолго.',
    sources: [src('s1', { kind: 'news', title: 'Перевозчики сообщают о росте тарифов на грузоперевозки', url: 'https://www.rbc.ru/business/', published_at: '2026-09-10' })],
    limitations: () => ['У банка нет материала по этой теме'],
  },
  n_2026_09_11_steel: {
    what: 'Цены на арматуру снизились на 5% за месяц. Застройщики откладывают запуск новых проектов из-за дорогого финансирования, спрос на прокат и стройматериалы падает.',
    why: p => p.industry.okved_section === '23' ? 'Покупатели группы — застройщики. Пауза в новых стройках снизит спрос на бетонные смеси в IV квартале и в начале 2027 года.' : 'Компания продаёт стройматериалы, а её спрос напрямую зависит от запуска новых строек.',
    features: p => ['industry.okved_name', ...(p.network.counterparty_sectors.length ? ['network.counterparty_sectors'] : ['geo.region'])],
    risks: [claim('Снижение объёмов продаж в IV квартале', '1–3 месяца', ['s1', 's2']), claim('Рост дебиторской задолженности застройщиков', '3–6 месяцев')],
    opps: [claim('Закупить сырьё по сниженным ценам на I квартал', '1–3 месяца', ['s1'])],
    actions: [act('Проверить лимиты по самым закредитованным покупателям'), act('Какой объём ввода жилья банк ожидает в 2027 году?', [], 'question_for_analysis')],
    bank: { source_id: 's2', text: 'Спрос со стороны строительной отрасли в IV квартале останется слабым на фоне сокращения запуска новых проектов' },
    ai: 'Слабый спрос стройки — следствие высокой ставки, поэтому восстановление начнётся не раньше её снижения. Главный риск — не цены, а платёжная дисциплина покупателей.',
    sources: [src('s1', { kind: 'news', title: 'Спрос на стальной прокат снижается на фоне паузы в стройке', url: 'https://www.interfax.ru/business/', published_at: '2026-09-11' }), src('s2', { ...MACRO, page: 8, quote: 'Спрос со стороны строительной отрасли в IV квартале останется слабым' })],
    limitations: p => (p.data_quality.notes.length ? p.data_quality.notes : ['Нет данных о доле застройщиков в выручке']),
  },
  n_2026_09_12_gold: {
    what: 'Золото обновило рекорд на ожиданиях снижения ставки ФРС. Доллар ослаб к основным валютам, курс рубля к юаню остался стабильным.',
    why: p => `Компания ведёт внешнюю торговлю (${p.geo.fts_countries.join(', ') || 'страны не указаны'}), но прямой связи с золотом нет: влияние только через курс валют.`,
    features: () => ['geo.fts_countries'],
    risks: [claim('Колебания курса при расчётах с зарубежными поставщиками', '1–3 месяца')],
    opps: [],
    actions: [act('Проверить валютную позицию по контрактам в долларах')],
    bank: null,
    ai: 'Слабая связь с компанией: материал полезен как фон, решений по нему не требуется.',
    sources: [src('s1', { kind: 'news', title: 'Золото обновило рекорд на ожиданиях снижения ставки ФРС', url: 'https://www.rbc.ru/finances/', published_at: '2026-09-12' })],
    limitations: () => ['Связь события с компанией слабая'],
  },
  n_2026_09_13_rate_cut_rumor: {
    what: 'Анонимный источник сообщил, что ЦБ может снизить ставку уже на октябрьском заседании. Официального подтверждения нет.',
    why: () => 'Если слух подтвердится, долг компании подешевеет раньше. Но источник не подтверждён и противоречит позиции банка.',
    features: debtFeatures,
    risks: [claim('Решения, принятые на основе слуха, могут оказаться ошибочными', null)],
    opps: [],
    actions: [act('Не менять бюджетные допущения до официального решения ЦБ', ['s2'])],
    bank: { source_id: 's2', text: 'Мы ожидаем, что ключевая ставка останется на уровне 14% как минимум до конца 2026 года' },
    ai: 'Слух противоречит прогнозу банка. Модель не учитывает его в выводах.',
    sources: [src('s1', { kind: 'news', title: 'Источник: ЦБ может снизить ставку уже в октябре', url: 'https://t.me/', published_at: '2026-09-13' }), src('s2', { ...MACRO, page: 4, quote: 'ключевая ставка останется на уровне 14%' })],
    limitations: () => ['Источник не подтверждён'],
  },
  n_2026_09_06_noisy: {
    what: 'По оценке Росстата, ВВП в I полугодии 2026 года вырос на 0,1% год к году. Текст исходного материала был зашумлён и очищен при разборе.',
    why: () => 'Общий фон экономики: прямой связи с компанией нет.',
    features: () => ['industry.okved_name'],
    risks: [], opps: [],
    actions: [act('Решений не требуется')],
    bank: null,
    ai: 'Материал устарел и слабо связан с компанией.',
    sources: [src('s1', { kind: 'news', title: 'Промышленность: ВВП в I полугодии 2026 года вырос на 0,1%', url: 'https://rosstat.gov.ru/', published_at: '2026-09-06', stale: true })],
    limitations: () => ['Материал устарел', 'Исходный текст был зашумлён'],
  },
}

export const ROLE_FOCUS: Record<Role, string> = {
  cfo: 'Для финансового директора главное — бюджет процентов на 2027 год и запас по ковенантам.',
  treasurer: 'Для казначейства главное — запас ликвидности и платёжный календарь на ближайшие месяцы.',
  ceo: 'Для генерального директора это вопрос спроса, издержек и того, на что опираться в планах на 2027 год.',
  owner: 'Для собственника это вопрос устойчивости бизнеса и стоимости долга в инвестиционной программе.',
  chief_accountant: 'Для бухгалтерии важно вовремя отразить изменения в условиях договоров и расчётах.',
  other: '',
}

// ---------- сценарии ----------
interface ScenarioContent { fork: ScenarioFork; branches: Omit<ScenarioBranch, 'verifier_issues'>[]; summary: string; key_uncertainty: string; decisions: ScenarioDecision[]; recommendation: string; switch_signals: string[]; limitations: string[] }
const br = (kind: ScenarioBranch['kind'], b: Omit<ScenarioBranch, 'kind' | 'verifier_issues' | 'company_features_used' | 'limitations'> & { limitations?: string[]; company_features_used?: string[] }): Omit<ScenarioBranch, 'verifier_issues'> => ({ kind, company_features_used: ['bank_products.loan', 'bank_products.leasing'], limitations: [], ...b })
const dec = (text: string, type: ScenarioDecision['type'], effects: ScenarioDecision['effects'], rationale: string, trigger: string | null = null): ScenarioDecision => ({ text, type, effects, rationale, trigger, from_branches: [], is_model_inference: true })

export const SCENARIOS: Record<string, ScenarioContent> = {
  n_2026_09_11_cbr_rate: {
    fork: { variable: 'Траектория ключевой ставки до конца I квартала 2027 года', outcomes: { good: 'ЦБ начинает снижать ставку в декабре и к марту доходит до 12–13%', base: 'Ставка остаётся около 14% до весны, как ожидает банк', bad: 'Инфляция ускоряется, и ЦБ повышает ставку до 15–16%' }, base_basis: 'bank_forecast', evidence: ['s2'] },
    branches: [
      br('good', { title: 'Раннее снижение ставки', assumption: 'Инфляция замедляется быстрее прогноза, и ЦБ снижает ставку уже в декабре', likelihood: 'low', likelihood_basis: 'Банк ожидает снижения не раньше I квартала 2027 года', impact: 'Долг дешевеет на квартал раньше плана, появляется окно для рефинансирования по ставке 12–13%', risks: [claim('Зафиксированные сейчас ставки окажутся дороже рыночных', '3–6 месяцев')], opportunities: [claim('Рефинансировать дорогие кредиты уже в I квартале', '3–6 месяцев')], signals: ['Недельная инфляция ниже 0,1% два месяца подряд', 'Мягкие формулировки ЦБ в октябре'], actions: [act('Подготовить пакет документов на рефинансирование заранее')] }),
      br('base', { title: 'Пауза до весны', assumption: 'Ставка держится на 14% до весны 2027 года', likelihood: 'high', likelihood_basis: 'Совпадает с прогнозом аналитиков банка', impact: 'Стоимость долга остаётся на текущем уровне, бюджет 2027 года надо считать от ставки 13–14%', risks: [claim('Проценты по долгу в I полугодии будут выше плана, если он считался от снижения', '3–6 месяцев', ['s2'])], opportunities: [claim('Держать свободные деньги на депозите под высокую ставку', '1–3 месяца')], signals: ['Инфляция около 6%', 'ЦБ сохраняет ставку в октябре и декабре'], actions: [act('Заложить ставку 13–14% в бюджет 2027 года', ['s2'])] }),
      br('bad', { title: 'Повышение ставки', assumption: 'Инфляция ускоряется из-за курса и зарплат, ЦБ повышает ставку до 15–16%', likelihood: 'medium', likelihood_basis: 'Риски по курсу и зарплатам названы в материале, но банк их не ожидает', impact: 'Обслуживание долга дорожает, растёт риск нарушить ковенанты и сократить инвестиционную программу', risks: [claim('Нарушение ковенант по долговой нагрузке', '6–12 месяцев'), claim('Сдвиг инвестиционных проектов', '6–12 месяцев')], opportunities: [claim('Конкуренты с большим долгом ослабнут', '6–12 месяцев')], signals: ['Курс выше 100 рублей за доллар', 'Жёсткий сигнал ЦБ в октябре'], actions: [act('Рассчитать запас по ковенантам при ставке 16%')] }),
    ],
    summary: 'Во всех ветках ставка до весны не опускается ниже 13%, поэтому планировать удешевление долга в I полугодии рискованно. Разница между ветками — во втором полугодии.',
    key_uncertainty: 'Ускорится ли инфляция из-за курса и зарплат',
    decisions: [
      dec('Заложить в бюджет 2027 года ставку 14%', 'robust', { good: 'neutral', base: 'helps', bad: 'helps' }, 'В хорошей ветке бюджет окажется с запасом, в остальных он реалистичен'),
      dec('Зафиксировать ставку по новому траншу сейчас', 'hedge', { good: 'hurts', base: 'neutral', bad: 'helps' }, 'Защищает от повышения, но в хорошей ветке компания переплатит'),
      dec('Подготовить рефинансирование и запустить его после первого снижения', 'conditional', { good: 'helps', base: 'neutral', bad: 'neutral' }, 'Ничего не стоит, пока ставка не снизится', 'ЦБ снижает ставку хотя бы на 0,5 п. п.'),
      dec('Держать свободную ликвидность на коротких депозитах', 'robust', { good: 'neutral', base: 'helps', bad: 'helps' }, 'Высокая доходность сохраняется в двух ветках из трёх'),
    ],
    recommendation: 'Строить бюджет от ставки 14%, не фиксировать весь долг, а держать готовое решение по рефинансированию на случай снижения.',
    switch_signals: ['Недельная инфляция ниже 0,1%', 'Курс выше 100 рублей за доллар', 'Формулировки пресс-релиза ЦБ'],
    limitations: ['Сроки погашения кредитов в профиле не указаны'],
  },
  n_2026_09_11_steel: {
    fork: { variable: 'Запуск новых строек до весны 2027 года', outcomes: { good: 'Стройка оживает после первых снижений ставки в начале 2027 года', base: 'Запуск новых проектов остаётся низким до весны, как ожидает банк', bad: 'Застройщики замораживают уже начатые проекты' }, base_basis: 'bank_forecast', evidence: ['s2'] },
    branches: [
      br('good', { title: 'Оживление стройки', assumption: 'Ставка снижается, ипотека дешевеет, застройщики возвращаются к запускам', likelihood: 'low', likelihood_basis: 'Банк ожидает слабый спрос стройки в IV квартале', impact: 'Объёмы продаж восстанавливаются ко II кварталу 2027 года', company_features_used: ['industry.okved_name'], risks: [claim('Нехватка мощностей при резком росте заказов', '6–12 месяцев')], opportunities: [claim('Нарастить долю рынка за счёт запасов, купленных дёшево', '3–6 месяцев', ['s1'])], signals: ['Рост выдач ипотеки', 'Новые разрешения на строительство'], actions: [act('Держать мощности в рабочем состоянии без сокращения персонала')] }),
      br('base', { title: 'Слабый спрос до весны', assumption: 'Запусков мало, текущие стройки идут по графику', likelihood: 'high', likelihood_basis: 'Совпадает с прогнозом банка', impact: 'Продажи в IV квартале ниже плана на величину отложенных проектов', company_features_used: ['industry.okved_name'], risks: [claim('Снижение объёмов продаж', '1–3 месяца', ['s1', 's2'])], opportunities: [claim('Закупить сырьё дешевле на I квартал', '1–3 месяца', ['s1'])], signals: ['Цены на арматуру продолжают снижаться', 'Застройщики переносят сроки ввода'], actions: [act('Сократить план продаж IV квартала')] }),
      br('bad', { title: 'Заморозка проектов', assumption: 'Застройщики останавливают начатые стройки из-за дорогого финансирования', likelihood: 'medium', likelihood_basis: 'Признаки есть в материале, но массовых заморозок пока нет', impact: 'Резкое падение продаж и рост просроченной дебиторской задолженности', company_features_used: ['industry.okved_name'], risks: [claim('Невозврат дебиторской задолженности застройщиков', '3–6 месяцев'), claim('Простой мощностей', '3–6 месяцев')], opportunities: [], signals: ['Задержки платежей крупных покупателей', 'Новости о заморозке проектов в регионе'], actions: [act('Перейти на предоплату с самыми рискованными покупателями')] }),
    ],
    summary: 'В двух ветках из трёх продажи до весны будут ниже плана. Главный риск — не объёмы, а платежи покупателей.',
    key_uncertainty: 'Начнут ли застройщики замораживать уже начатые проекты',
    decisions: [
      dec('Проверить лимиты и условия оплаты по крупным покупателям', 'robust', { good: 'neutral', base: 'helps', bad: 'helps' }, 'Снижает потери в плохой ветке и почти ничего не стоит'),
      dec('Закупить сырьё на I квартал по текущим ценам', 'conditional', { good: 'helps', base: 'helps', bad: 'hurts' }, 'Выгодно, если спрос не рухнет', 'Платежи покупателей идут без задержек'),
      dec('Перейти на предоплату с рискованными покупателями', 'hedge', { good: 'hurts', base: 'neutral', bad: 'helps' }, 'Страхует от неплатежей, но может стоить части клиентов'),
    ],
    recommendation: 'Начать с проверки лимитов покупателей, а закупку сырья делать только при стабильных платежах.',
    switch_signals: ['Задержки платежей крупных покупателей', 'Динамика выдач ипотеки'],
    limitations: ['Нет данных о доле застройщиков в выручке'],
  },
  n_2026_09_03_petrol: {
    fork: { variable: 'Оптовые цены на бензин в IV квартале 2026 года', outcomes: { good: 'Цены возвращаются к уровню лета после окончания ремонтов на НПЗ', base: 'Цены держатся у максимума до конца года', bad: 'Дефицит усиливается, цены растут ещё на 10–15%' }, base_basis: 'item_expectation', evidence: ['s1'] },
    branches: [
      br('good', { title: 'Цены возвращаются', assumption: 'НПЗ выходят из ремонтов, предложение растёт', likelihood: 'medium', likelihood_basis: 'Сезон ремонтов заканчивается в октябре', impact: 'Потребность в оборотных средствах снижается, маржа восстанавливается', company_features_used: ['industry.okved_name'], risks: [claim('Запасы, купленные дорого, придётся продавать дешевле', '1–3 месяца')], opportunities: [claim('Нарастить объёмы продаж при снижении цен', '1–3 месяца')], signals: ['Снижение биржевых цен две недели подряд'], actions: [act('Не создавать избыточных запасов по пиковым ценам')] }),
      br('base', { title: 'Цены у максимума', assumption: 'Цены держатся до конца года', likelihood: 'high', likelihood_basis: 'Ожидание участников рынка из материала', impact: 'Растёт потребность в оборотном финансировании, маржа под давлением', company_features_used: ['industry.okved_name', 'bank_products.loan'], risks: [claim('Нехватка лимита оборотного кредита', '1–3 месяца', ['s1'])], opportunities: [], signals: ['Цены стоят на месте', 'Нет новостей о мерах правительства'], actions: [act('Увеличить лимит оборотного кредита', ['s1'])] }),
      br('bad', { title: 'Дефицит топлива', assumption: 'Дефицит усиливается, вводятся ограничения экспорта', likelihood: 'low', likelihood_basis: 'Признаков дефицита пока нет', impact: 'Перебои с поставками, покупатели уходят к вертикально интегрированным компаниям', company_features_used: ['industry.okved_name'], risks: [claim('Срыв поставок ключевым покупателям', '1–3 месяца')], opportunities: [claim('Рост цен на остатки, если они есть', '1 месяц')], signals: ['Ограничения экспорта топлива', 'Очереди на АЗС в регионах'], actions: [act('Заранее договориться о резервных поставщиках')] }),
    ],
    summary: 'Во всех ветках, кроме хорошей, компании нужно больше оборотных денег. Главное решение — лимит финансирования, а не цены.',
    key_uncertainty: 'Закончится ли рост цен вместе с сезоном ремонтов НПЗ',
    decisions: [
      dec('Согласовать с банком резервный лимит оборотного кредита', 'robust', { good: 'neutral', base: 'helps', bad: 'helps' }, 'Лимит без выборки почти ничего не стоит'),
      dec('Закупить топливо впрок по текущим ценам', 'hedge', { good: 'hurts', base: 'neutral', bad: 'helps' }, 'Защищает от дефицита, но в хорошей ветке приведёт к убытку'),
      dec('Перейти на формулу цены с привязкой к бирже', 'robust', { good: 'helps', base: 'helps', bad: 'helps' }, 'Переносит ценовой риск на покупателя в любой ветке'),
    ],
    recommendation: 'Согласовать резервный лимит и перевести ключевые договоры на биржевую формулу цены. Запасы впрок — только при признаках дефицита.',
    switch_signals: ['Динамика биржевых цен', 'Новости об ограничении экспорта'],
    limitations: ['У банка нет аналитики по топливному рынку'],
  },
}
SCENARIOS.a_macro_2026_09 = SCENARIOS.n_2026_09_11_cbr_rate

// ---------- история и мониторинг для демо ----------
export const HISTORY_SEED = [
  { login: 'retail.cfo', company_id: 'U_1595', item_id: 'n_2026_09_11_cbr_rate', card_title: 'Банк России сохранил ключевую ставку на уровне 14%', role: 'cfo' as Role, question: 'Стоит ли сейчас фиксировать ставку по новому траншу?', answer: 'Банк ожидает снижения ставки не раньше I квартала 2027 года. Если транш нужен в ближайшие месяцы, плавающая ставка с возможностью рефинансирования гибче фиксированной. Точнее оценить нельзя: в профиле нет сроков погашения кредитов.', asked_at: '2026-10-22T16:40:00', has_issues: true },
  { login: 'retail.cfo', company_id: 'U_1595', item_id: 'n_2026_09_08_retail', card_title: 'Рост розничных продаж замедлился', role: 'cfo' as Role, question: 'Насколько упадёт выручка осеннего сезона?', answer: 'Точной оценки нет: данных о продажах компании по сезонам в профиле нет. По рынку замедление непродовольственной розницы — примерно вдвое к июлю.', asked_at: '2026-10-22T11:05:00', has_issues: false },
]
