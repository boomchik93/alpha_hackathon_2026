import type { CompanyBrief, CompanyProfile, FeedbackReason, Role, ScenarioDecision, ScenarioKind, SourceKind } from '../api/types'

export const ROLE_LABEL: Record<Role, string> = {
  owner: 'Собственник', ceo: 'Генеральный директор', cfo: 'Финансовый директор', treasurer: 'Казначей',
  chief_accountant: 'Главный бухгалтер', other: 'Без роли',
}
export const SOURCE_LABEL: Record<SourceKind, string> = { news: 'Новость', bank_analytics: 'Аналитика банка', synthetic_test: 'Тестовая новость' }
export const REASON_LABEL: Record<FeedbackReason, string> = { wrong_fact: 'Неверный факт', not_relevant: 'Не про мою компанию', obvious: 'Очевидно, ничего нового', other: 'Другое' }

// Коды ошибок бэкенда (backend/app/errors.py)
export const ERROR_TEXT: Record<string, string> = {
  unauthorized: 'Нужно войти заново',
  forbidden: 'Нет доступа',
  not_found: 'Не найдено',
  validation_error: 'Неверный запрос',
  scenarios_not_applicable: 'Для этого события сценарии не строятся',
  rate_limited: 'Слишком много запросов, подождите немного',
  queue_full: 'Сейчас генерируется слишком много карточек, повторите через минуту',
  budget_exceeded: 'Дневной лимит расходов на модели исчерпан, доступны сохранённые карточки',
  model_failed: 'Модель не дала годный ответ',
  ml_error: 'ML-сервис ответил неожиданно',
  ml_unavailable: 'ML-сервис недоступен',
  ml_timeout: 'Сервис отвечает слишком долго',
  internal_error: 'Ошибка сервера',
  network: 'Нет связи с сервером',
}

// Темы ленты (ml/topics.py, TOPICS)
export const TOPIC_LABEL: Record<string, string> = {
  key_rate: 'Ключевая ставка', bonds: 'Долговой рынок, ОФЗ', inflation: 'Инфляция', fuel: 'Топливо',
  consumption: 'Потребление и розница', labor: 'Рынок труда', fx_gold: 'Валюта и внешние рынки',
  gdp_industry: 'ВВП и промышленность', steel: 'Металлургия и сталь',
}
export const topicLabel = (k: string) => TOPIC_LABEL[k] ?? k

// Пути полей профиля (why_it_matters.company_features_used) → понятная подпись и значение из профиля.
const FEATURE_LABEL: Record<string, string> = {
  'bank_products': 'Продукты банка', 'network.top_counterparties.': 'Контрагенты',
  'bank_products.loan': 'Кредит в банке', 'bank_products.leasing': 'Лизинг', 'bank_products.deposit': 'Депозиты',
  'bank_products.rko': 'Расчётный счёт', 'bank_products.guarantee': 'Банковские гарантии', 'bank_products.payroll': 'Зарплатный проект',
  'bank_products.factoring': 'Факторинг', 'industry.okved_name': 'Отрасль', 'industry.okved_code': 'Код ОКВЭД',
  'industry.sector_tags': 'Сектор', 'size.revenue_rub': 'Выручка', 'size.revenue_band': 'Размер', 'size.workers_band': 'Сотрудники',
  'geo.region': 'Регион', 'geo.city': 'Город', 'geo.foreign_trade': 'Внешняя торговля', 'geo.fts_countries': 'Страны ВЭД',
  'network.counterparty_sectors': 'Отрасли контрагентов', 'network.top_counterparties': 'Ключевые контрагенты', 'network.known_share': 'Известные контрагенты',
  'group.group_id': 'Группа компаний', 'group.group_revenue_rub': 'Выручка группы', 'market_position': 'Положение в отрасли',
}
export function featureLabel(path: string): string {
  return FEATURE_LABEL[path] ?? path
}
export function featureValue(profile: CompanyProfile | undefined, path: string): string | null {
  if (!profile) return null
  const v = path.split('.').reduce<unknown>((o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), profile)
  if (v === null || v === undefined) return null
  if (typeof v === 'boolean') return v ? 'есть' : 'нет'
  if (typeof v === 'number') return path.includes('revenue') ? fmtRub(v) : path.includes('share') ? `${Math.round(v * 100)}%` : String(v)
  if (Array.isArray(v)) return v.length ? v.map(x => (typeof x === 'object' ? (x as { okved_name?: string }).okved_name ?? '' : String(x))).filter(Boolean).slice(0, 3).join(', ') : null
  if (typeof v === 'object') return (v as { note?: string }).note ?? null
  return String(v)
}
export const featureText = (profile: CompanyProfile | undefined, path: string) => {
  const v = featureValue(profile, path)
  return v ? `${featureLabel(path)}: ${v}` : featureLabel(path)
}

// Компании обезличены: имени нет, показываем отрасль и идентификатор.
export function shortIndustry(name: string | null | undefined): string {
  if (!name) return 'Компания'
  return name.replace(/^Торговля (оптовая|розничная) /, (_, t) => (t === 'оптовая' ? 'Опт: ' : 'Розница: ')).replace(/, включая.*$/, '').replace(/ в специализированных магазинах/, '').split(',')[0]
}
export function companyLabel(c: CompanyBrief | CompanyProfile | undefined): string {
  if (!c) return '…'
  const name = 'okved_name' in c ? c.okved_name : c.industry.okved_name
  return `${shortIndustry(name)}${c.level === 'group' ? ' · группа' : ''}`
}

export function fmtRub(v: number | null | undefined): string {
  if (v === null || v === undefined) return 'нет данных'
  if (v >= 1e9) return `${(v / 1e9).toFixed(1).replace('.', ',')} млрд ₽`
  if (v >= 1e6) return `${Math.round(v / 1e6)} млн ₽`
  return `${Math.round(v).toLocaleString('ru-RU')} ₽`
}

export const LIKELIHOOD: Record<'low' | 'medium' | 'high', string> = { low: 'Маловероятно', medium: 'Возможно', high: 'Вероятно' }
export const KIND_LABEL: Record<ScenarioKind, string> = { good: 'Хороший', base: 'Базовый', bad: 'Плохой' }
export const DECISION_TYPE: Record<ScenarioDecision['type'], string> = { robust: 'Делать в любом случае', conditional: 'Запускать по сигналу', hedge: 'Страховка от плохого' }
export const EFFECT_LABEL: Record<'helps' | 'neutral' | 'hurts', string> = { helps: 'помогает', neutral: 'без эффекта', hurts: 'вредит' }
export const BASE_BASIS: Record<string, string> = { bank_forecast: 'прогноз банка', item_expectation: 'ожидание из материала', assumption: 'допущение модели' }

const MONTHS = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
const MONTHS_SHORT = ['янв', 'фев', 'мар', 'апр', 'мая', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
// ISO-дата без времени (YYYY-MM-DD) разбирается как локальная, чтобы не съезжать на день из-за часового пояса
const parse = (iso: string) => (/^\d{4}-\d{2}-\d{2}$/.test(iso) ? new Date(`${iso}T12:00:00`) : new Date(iso))
export function fmtDate(iso: string | null | undefined, short = false): string {
  if (!iso) return ''
  const d = parse(iso)
  return `${d.getDate()} ${(short ? MONTHS_SHORT : MONTHS)[d.getMonth()]}${short ? '' : ' ' + d.getFullYear()}`
}
export function fmtTime(iso: string): string {
  const d = parse(iso)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}
export const fmtScore = (v: number) => v.toFixed(2).replace('.', ',')
export function plural(n: number, one: string, few: string, many: string) {
  const m10 = n % 10, m100 = n % 100
  if (m10 === 1 && m100 !== 11) return one
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few
  return many
}

// Шаги генерации для индикатора. Бэкенд отдаёт только статус задачи, поэтому шаг оцениваем по прошедшему времени.
export const GEN_STEPS = ['Читаем материал и проверяем источник', 'Сопоставляем с профилем компании', 'Ищем аналитику банка по теме', 'Проверяем цифры и цитаты', 'Формулируем последствия и действия']
export const SCEN_STEPS = ['Находим главную неопределённость', 'Пишем хороший исход', 'Пишем базовый исход', 'Пишем плохой исход', 'Разбираем решения по всем веткам']
