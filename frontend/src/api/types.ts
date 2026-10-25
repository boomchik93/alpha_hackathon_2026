// Типы данных фронта. Повторяют контракт ml/schemas.py и ответы backend/app/routes.py.
// Меняются только вместе с ними (CONTRIBUTING.md: контракт между слоями).

export type Role = 'owner' | 'ceo' | 'cfo' | 'treasurer' | 'chief_accountant' | 'other'
export type Access = 'user' | 'expert' | 'admin'
export type SourceKind = 'news' | 'bank_analytics' | 'synthetic_test'
export type ISODate = string // YYYY-MM-DD
export type ISODateTime = string

// ---------- вход ----------
export interface Me { login: string; display_name: string; role: Role; access: Access; companies: string[] }
export interface LoginOut { token: string; expires_at: ISODateTime; user: Me }

// ---------- профиль компании (CompanyProfile) ----------
export interface CompanyBrief { company_id: string; level: 'company' | 'group'; okved_name: string | null; revenue_rub: number | null; region: string | null; client_type: string | null }
export interface Industry { okved_code: string | null; okved_raw?: string | null; okved_name: string | null; okved_section?: string | null; sector_tags: string[] }
export interface Size { revenue_rub: number | null; revenue_band: string | null; workers_band: string | null; age_band: string | null }
export interface Geo { city: string | null; region: string | null; foreign_trade: boolean | null; fts_countries: string[] }
export interface BankProducts { rko: boolean | null; guarantee: boolean | null; loan: boolean | null; leasing: boolean | null; deposit: boolean | null; payroll: boolean | null; factoring: boolean | null }
export interface GroupInfo { group_id: string | null; group_size: number | null; group_revenue_rub: number | null; group_clients: number | null }
export interface Counterparty { id: string; role: 'payer' | 'beneficiary'; amt_rub: number; cnt: number; known: boolean; okved_name: string | null }
export interface Network { top_counterparties: Counterparty[]; counterparty_sectors: string[]; known_share: number | null; note: string | null }
export interface DataQuality { missing_fields: string[]; conflicts: string[]; notes: string[] }
export interface MarketPosition { okved_prefix: string; digits: number; peers: number; rank: number; percentile: number; revenue_share: number | null; note: string }
export interface CompanyProfile {
  company_id: string
  level: 'company' | 'group'
  org_type: string | null
  ownership: string | null
  client_type: string | null
  stage: string | null
  eligible_big: boolean
  industry: Industry
  size: Size
  geo: Geo
  bank_products: BankProducts
  group: GroupInfo
  network: Network
  data_quality: DataQuality
  member_ids: string[]
  market_position: MarketPosition | null
}

// ---------- лента ----------
export interface FeedEntry {
  item_id: string
  title: string
  published_at: ISODate
  source_kind: SourceKind
  score: number
  reasons: string[]
  topics: string[]
  stale: boolean
  tier: 'main' | 'maybe'
  related: string[]
  is_new?: boolean // добавляет бэкенд
}
export interface Feed { company_id: string; as_of: ISODate; entries: FeedEntry[]; filtered_out: number; maybe_count: number; new_count?: number; fetched_at?: ISODateTime }
export interface FeedFilters { tier?: 'main' | 'maybe' | 'all'; source_kind?: 'news' | 'bank_analytics'; topic?: string; date_from?: ISODate; date_to?: ISODate; hide_stale?: boolean }
export interface Summary { company_id: string; new_count: number; top: { item_id: string; title: string; score: number; published_at: ISODate; is_new: boolean } | null; as_of: ISODate; updated_at: ISODateTime }

// ---------- карточка (InsightCard) ----------
export interface Claim { text: string; horizon: string | null; evidence: string[]; is_model_inference: boolean }
export interface SuggestedAction extends Claim { type: 'action' | 'question_for_analysis' }
export interface BankQuote { source_id: string; text: string }
export interface BankView { text: string; source_ids: string[]; quotes: BankQuote[] }
export interface AIInference { text: string; source_ids: string[]; label: string }
export interface SourceRef { id: string; kind: SourceKind; title: string; url: string | null; published_at: ISODate | null; doc_id: string | null; page: number | null; quote: string | null; stale: boolean }
export interface CardMeta { model: string; prompt_version: string; generated_at: ISODateTime; retries: number; cost_rub: number; verifier_issues: string[] }
export interface ImpactChannel { channel: string; strength: 'strong' | 'weak' | 'none'; direction: 'negative' | 'positive' | 'mixed' | 'unclear'; mechanism: string; profile_features: string[]; evidence: string[] }
export interface InsightCard {
  card_id: string
  company_id: string
  item_ids: string[]
  role: Role | null
  relevance: { score: number; reasons: string[] }
  what_happened: { text: string; facts: string[] }
  why_it_matters: { text: string; company_features_used: string[] }
  consequences: { risks: Claim[]; opportunities: Claim[] }
  bank_view: BankView | null
  ai_inference: AIInference
  suggested_actions: SuggestedAction[]
  limitations: string[]
  sources: SourceRef[]
  trace_id: string
  meta: CardMeta
  channels: ImpactChannel[]
}

// ---------- сценарии (ScenarioSet) ----------
export type ScenarioKind = 'good' | 'base' | 'bad'
export interface ScenarioFork { variable: string; outcomes: Record<ScenarioKind, string>; base_basis: 'bank_forecast' | 'item_expectation' | 'assumption'; evidence: string[] }
export interface ScenarioBranch {
  kind: ScenarioKind
  title: string
  assumption: string
  likelihood: 'low' | 'medium' | 'high'
  likelihood_basis: string
  impact: string
  company_features_used: string[]
  risks: Claim[]
  opportunities: Claim[]
  signals: string[]
  actions: SuggestedAction[]
  limitations: string[]
  verifier_issues: string[]
}
export interface ScenarioDecision { text: string; type: 'robust' | 'conditional' | 'hedge'; effects: Record<ScenarioKind, 'helps' | 'neutral' | 'hurts'>; rationale: string; trigger: string | null; from_branches: ScenarioKind[]; is_model_inference: boolean }
export interface ScenarioAnalysis { summary: string; key_uncertainty: string; decisions: ScenarioDecision[]; recommendation: string; switch_signals: string[]; limitations: string[] }
export interface ScenarioSet {
  set_id: string
  card_id: string
  company_id: string
  item_ids: string[]
  role: Role | null
  fork: ScenarioFork
  branches: ScenarioBranch[]
  analysis: ScenarioAnalysis
  sources: SourceRef[]
  trace_id: string
  meta: { model: string; prompt_version: string; generated_at: ISODateTime; card_generated_at: ISODateTime; retries: number; cost_rub: number; verifier_issues: string[] }
}

// ---------- фоновая генерация ----------
export interface JobView {
  job_id: string
  kind: 'insight' | 'scenarios'
  status: 'queued' | 'running' | 'done' | 'failed'
  poll_url: string
  created_at: ISODateTime
  updated_at: ISODateTime
  card: InsightCard | null
  scenarios: ScenarioSet | null
  error: { code: string; message: string; retry_after?: number } | null
}
export type Ready<T> = { status: 'ready'; data: T } | { status: 'pending'; job: JobView }

// ---------- вопросы, оценки, трасса ----------
export interface FollowupAnswer { card_id: string; question: string; answer: string; source_ids: string[]; sources: SourceRef[]; is_model_inference: boolean; verifier_issues: string[]; trace_id: string }
export type FeedbackReason = 'wrong_fact' | 'not_relevant' | 'obvious' | 'other'
export interface FeedbackIn { rating: 'up' | 'down'; reason?: FeedbackReason; comment?: string }
export interface CardTrace {
  trace_id: string
  kind?: string
  duration_s: number | null
  retries: number
  plan: { tool: string | null; why: string }[] // null: шаг рассуждения без инструмента
  tool_calls: { tool: string; ms: number | null }[]
  replans: { trigger: string; why: string }[]
  checks: { attempt: number; issues: number }[]
  sources: Pick<SourceRef, 'id' | 'kind' | 'title' | 'published_at' | 'doc_id' | 'page' | 'url'>[]
  company_features_used: string[]
  model: string | null
  prompt_version: string | null
  cost_rub?: number // эксперт и админ
}

// ---------- мониторинг ----------
export interface Health { ok: boolean; ml_ok: boolean; ml_error?: string; llm_mock?: boolean; spent_today_rub?: number; budget_rub?: number }
export interface VoteStats { total: number; down: number; down_share: number | null; reasons: Record<string, number> }
export interface FeedbackSummary {
  user?: VoteStats & { by_role: Record<string, VoteStats> }
  expert?: VoteStats & { by_role: Record<string, VoteStats> }
  ml: Record<string, VoteStats> | null // по версии промпта, ключ «все» — по всем
  // Вовлечённость за `days` дней из журнала: действие -> число обращений и разных пользователей
  usage?: { days: number; actions: Record<string, { count: number; users: number }>; actions_per_card: number | null }
  items_by_day?: Record<string, number> | null // дата публикации -> число материалов; null, если ML недоступен
}
export interface AuditRow { id: number; ts: ISODateTime; user_id: number | null; action: string; company_id: string | null; item_id: string | null; card_id: string | null; trace_id: string | null; detail: string | null }

// ---------- история вопросов (хранится в браузере: ручки в бэкенде нет) ----------
export interface HistoryItem { id: string; login: string; company_id: string; item_id: string; card_title: string; role: Role | null; question: string; answer: string; asked_at: ISODateTime; has_issues: boolean }

export interface ApiError { status: number; code: string; message: string; retry_after?: number }
