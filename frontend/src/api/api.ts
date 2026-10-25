import type {
  AuditRow, CardTrace, CompanyBrief, CompanyProfile, Feed, FeedFilters, FeedbackIn, FeedbackSummary, FollowupAnswer, Health,
  InsightCard, JobView, LoginOut, Me, Ready, Role, ScenarioSet, Summary,
} from './types'

// Один интерфейс для мока и настоящего бэкенда (backend/README.md, «Ручки»): экраны не знают, откуда данные.
// role передаётся только для эксперта и админа: обычному пользователю роль задаёт профиль на бэкенде.
export interface Api {
  login(login: string, password: string): Promise<LoginOut>
  logout(): Promise<void>
  me(): Promise<Me>
  companies(): Promise<CompanyBrief[]>
  company(id: string): Promise<CompanyProfile>
  feed(companyId: string, filters?: FeedFilters): Promise<Feed>
  summary(companyId: string): Promise<Summary>
  card(companyId: string, itemId: string, role?: Role): Promise<Ready<InsightCard>>
  scenarios(companyId: string, itemId: string, role?: Role): Promise<Ready<ScenarioSet>>
  job(jobId: string): Promise<JobView>
  followup(cardId: string, question: string, role?: Role): Promise<FollowupAnswer>
  feedback(cardId: string, body: FeedbackIn): Promise<{ ok: boolean; synced_to_ml: boolean }>
  cardTrace(cardId: string): Promise<CardTrace>
  sourcePdf(docId: string): Promise<Blob | null> // null — PDF недоступен (мок)
  health(): Promise<Health>
  feedbackSummary(): Promise<FeedbackSummary>
  audit(limit?: number): Promise<AuditRow[]>
}
