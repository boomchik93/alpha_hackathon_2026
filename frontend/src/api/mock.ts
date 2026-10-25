// Мок в формате настоящего бэкенда: те же ручки, коды ошибок, задачи 202 → опрос /jobs/{id}.
// Нужен для демо без бэкенда и ML (npm run dev, npm run build:demo).
import type { Api } from './api'
import type {
  ApiError, AuditRow, CardTrace, CompanyBrief, Feed, FeedEntry, FeedbackIn, FollowupAnswer, InsightCard, JobView, Me, Role, ScenarioSet, SourceRef, VoteStats,
} from './types'
import { CARDS, ITEMS, PROFILES, ROLE_FOCUS, SCENARIOS, USERS, feedEntries } from './fixtures'
import { token } from './token'

const AS_OF = '2026-09-15'
const CARD_MS = 6500
const SCEN_MS = 9000
const wait = (ms = 250) => new Promise(r => setTimeout(r, ms))
const now = () => new Date().toISOString()
const fail = (status: number, code: string, message: string): never => { throw { status, code, message } satisfies ApiError }

// ---------- сессия ----------
function me(): Me {
  const t = token.get()
  const u = t?.startsWith('mock:') ? USERS.find(x => x.login === t.slice(5)) : undefined
  if (!u) { token.clear(); return fail(401, 'unauthorized', 'Нужно войти') }
  const { password: _p, ...rest } = u
  return rest
}
const visible = (u: Me) => (u.access === 'user' ? u.companies : ['U_1595', 'U_26030', 'U_3263', 'U_183405', 'group_134463_sb'])
function checkCompany(u: Me, id: string) {
  if (!PROFILES[id]) fail(404, 'not_found', 'Компания не найдена')
  const allowed = u.access !== 'user' || u.companies.some(c => c === id || PROFILES[c]?.member_ids.includes(id))
  if (!allowed) fail(403, 'forbidden', 'Нет доступа к этой компании')
}
const effRole = (u: Me, role?: Role): Role => (u.access === 'user' ? u.role : role ?? 'cfo')

// ---------- журнал ----------
const audit: AuditRow[] = []
let auditId = 1
function log(u: Me | null, action: string, extra: Partial<AuditRow> = {}) {
  audit.unshift({ id: auditId++, ts: now(), user_id: u ? USERS.findIndex(x => x.login === u.login) + 1 : null, action, company_id: null, item_id: null, card_id: null, trace_id: null, detail: null, ...extra })
}

// ---------- карточки ----------
const cardId = (company: string, item: string, role: Role) => `c_${company}_${item}_${role}`
const cards = new Map<string, InsightCard>()
const sets = new Map<string, ScenarioSet>()
const failedOnce = new Set<string>()

function entry(companyId: string, itemId: string): FeedEntry | undefined {
  return feedEntries(companyId).entries.find(e => e.item_id === itemId)
}

function buildCard(companyId: string, itemId: string, role: Role): InsightCard {
  const p = PROFILES[companyId]
  const c = CARDS[itemId]
  const e = entry(companyId, itemId)
  const focus = ROLE_FOCUS[role]
  const newsIds = c.sources.filter(s => s.kind !== 'bank_analytics').map(s => s.id)
  const bankIds = c.sources.filter(s => s.kind === 'bank_analytics').map(s => s.id)
  const id = cardId(companyId, itemId, role)
  return {
    card_id: id, company_id: companyId, item_ids: [itemId], role,
    relevance: { score: e?.score ?? 0.5, reasons: e?.reasons ?? [] },
    what_happened: { text: c.what, facts: (newsIds.length ? newsIds : bankIds).map((_, i) => `f${i + 1}`) },
    why_it_matters: { text: focus ? `${c.why(p)} ${focus}` : c.why(p), company_features_used: c.features(p) },
    consequences: { risks: c.risks, opportunities: c.opps },
    bank_view: c.bank ? { text: c.bank.text, source_ids: [c.bank.source_id], quotes: [c.bank] } : null,
    ai_inference: { text: c.ai, source_ids: [], label: 'Вывод модели' },
    suggested_actions: c.actions,
    limitations: c.limitations(p),
    sources: c.sources,
    trace_id: `t_${id}`,
    meta: { model: 'demo-mock', prompt_version: 'insight-v3', generated_at: now(), retries: itemId === 'n_2026_09_13_rate_cut_rumor' ? 1 : 0, cost_rub: 0.42, verifier_issues: [] },
    channels: [],
  }
}

function buildSet(card: InsightCard): ScenarioSet {
  const s = SCENARIOS[card.item_ids[0]]
  return {
    set_id: `s_${card.card_id}`, card_id: card.card_id, company_id: card.company_id, item_ids: card.item_ids, role: card.role,
    fork: s.fork,
    branches: s.branches.map(b => ({ ...b, verifier_issues: [] })),
    analysis: { summary: s.summary, key_uncertainty: s.key_uncertainty, decisions: s.decisions, recommendation: s.recommendation, switch_signals: s.switch_signals, limitations: s.limitations },
    sources: card.sources, trace_id: `t_s_${card.card_id}`,
    meta: { model: 'demo-mock', prompt_version: 'scenarios-v2', generated_at: now(), card_generated_at: card.meta.generated_at, retries: 0, cost_rub: 0.95, verifier_issues: [] },
  }
}

// ---------- задачи ----------
interface Job { view: JobView; ready: number; make: () => Partial<JobView> }
const jobs = new Map<string, Job>()
let jobSeq = 1
function startJob(kind: JobView['kind'], ms: number, make: () => Partial<JobView>): JobView {
  const id = `j_${jobSeq++}`
  const view: JobView = { job_id: id, kind, status: 'queued', poll_url: `/api/jobs/${id}`, created_at: now(), updated_at: now(), card: null, scenarios: null, error: null }
  jobs.set(id, { view, ready: Date.now() + ms, make })
  return view
}

// ---------- голоса ----------
const votes: { role: Role; access: Me['access']; rating: 'up' | 'down'; reason?: string }[] = [
  ...Array.from({ length: 17 }, (_, i) => ({ role: (['cfo', 'ceo', 'owner', 'treasurer'] as Role[])[i % 4], access: 'user' as const, rating: 'up' as const })),
  { role: 'cfo', access: 'user', rating: 'down', reason: 'obvious' },
  { role: 'ceo', access: 'user', rating: 'down', reason: 'not_relevant' },
  { role: 'cfo', access: 'user', rating: 'down', reason: 'wrong_fact' },
  ...Array.from({ length: 6 }, () => ({ role: 'cfo' as Role, access: 'expert' as const, rating: 'up' as const })),
  { role: 'cfo', access: 'expert', rating: 'down', reason: 'obvious' },
]
function stats(list: typeof votes): VoteStats {
  const down = list.filter(v => v.rating === 'down')
  const reasons: Record<string, number> = {}
  down.forEach(v => { if (v.reason) reasons[v.reason] = (reasons[v.reason] ?? 0) + 1 })
  return { total: list.length, down: down.length, down_share: list.length ? down.length / list.length : null, reasons }
}
function grouped(list: typeof votes) {
  const by_role: Record<string, VoteStats> = {}
  new Set(list.map(v => v.role)).forEach(r => { by_role[r] = stats(list.filter(v => v.role === r)) })
  return { ...stats(list), by_role }
}

let spent = 37.8

// ---------- API ----------
export const mockApi: Api = {
  async login(login, password) {
    await wait(400)
    const u = USERS.find(x => x.login === login.trim())
    if (!u || u.password !== password) { log(null, 'login_failed', { detail: login }); fail(401, 'unauthorized', 'Неверный логин или пароль') }
    token.set(`mock:${u!.login}`)
    const user = me()
    log(user, 'login')
    return { token: `mock:${u!.login}`, expires_at: new Date(Date.now() + 12 * 3600e3).toISOString(), user }
  },
  async logout() { try { log(me(), 'logout') } catch { /* уже вышли */ } token.clear() },
  async me() { await wait(100); return me() },

  async companies() {
    await wait()
    const u = me()
    return visible(u).map((id): CompanyBrief => {
      const p = PROFILES[id]
      return { company_id: id, level: p.level, okved_name: p.industry.okved_name, revenue_rub: p.size.revenue_rub, region: p.geo.region, client_type: p.client_type }
    })
  },
  async company(id) { await wait(); checkCompany(me(), id); return PROFILES[id] },

  async feed(companyId, f = {}) {
    await wait(350)
    checkCompany(me(), companyId)
    const { entries, filtered } = feedEntries(companyId)
    const withNew = entries.map(e => ({ ...e, is_new: e.published_at >= '2026-09-11' }))
    const tier = f.tier ?? 'all'
    const out = withNew.filter(e => (tier === 'all' || e.tier === tier)
      && (!f.source_kind || e.source_kind === f.source_kind)
      && (!f.topic || e.topics.includes(f.topic))
      && (!f.date_from || e.published_at >= f.date_from)
      && (!f.date_to || e.published_at <= f.date_to)
      && !(f.hide_stale && e.stale))
    const feed: Feed = {
      company_id: companyId, as_of: AS_OF, entries: out, filtered_out: filtered,
      maybe_count: out.filter(e => e.tier === 'maybe').length,
      new_count: withNew.filter(e => e.is_new && e.tier === 'main').length, fetched_at: now(),
    }
    return feed
  },
  async summary(companyId) {
    await wait()
    checkCompany(me(), companyId)
    const main = feedEntries(companyId).entries.filter(e => e.tier === 'main')
    const fresh = main.filter(e => e.published_at >= '2026-09-11')
    const top = [...main].sort((a, b) => b.score - a.score)[0]
    return { company_id: companyId, new_count: fresh.length, top: top ? { item_id: top.item_id, title: top.title, score: top.score, published_at: top.published_at, is_new: top.published_at >= '2026-09-11' } : null, as_of: AS_OF, updated_at: now() }
  },

  async card(companyId, itemId, role) {
    await wait(300)
    const u = me()
    checkCompany(u, companyId)
    if (!CARDS[itemId] || !ITEMS[itemId]) fail(404, 'not_found', 'Материал не найден')
    const r = effRole(u, role)
    const id = cardId(companyId, itemId, r)
    const ready = cards.get(id)
    if (ready) { log(u, 'insight', { company_id: companyId, item_id: itemId, detail: r }); return { status: 'ready', data: ready } }
    if (spent >= 500) fail(429, 'budget_exceeded', 'Дневной лимит исчерпан')
    log(u, 'insight', { company_id: companyId, item_id: itemId, detail: r })
    // Одна генерация в демо падает: показать повтор и понятную ошибку.
    const shouldFail = itemId === 'n_2026_09_12_gold' && !failedOnce.has(id)
    if (shouldFail) failedOnce.add(id)
    return {
      status: 'pending',
      job: startJob('insight', shouldFail ? 3000 : CARD_MS, () => {
        if (shouldFail) return { status: 'failed', error: { code: 'model_failed', message: 'Проверка не пропустила ответ модели: цитата не найдена в источнике' } }
        const card = buildCard(companyId, itemId, r)
        cards.set(id, card); spent += card.meta.cost_rub
        return { status: 'done', card }
      }),
    }
  },

  async scenarios(companyId, itemId, role) {
    await wait(300)
    const u = me()
    checkCompany(u, companyId)
    const id = cardId(companyId, itemId, effRole(u, role))
    const card = cards.get(id)
    if (!card) fail(404, 'not_found', 'Сначала откройте карточку')
    if (!SCENARIOS[itemId]) fail(422, 'scenarios_not_applicable', 'Событие не задаёт развилку для компании')
    const ready = sets.get(id)
    if (ready) return { status: 'ready', data: ready }
    log(u, 'scenarios', { company_id: companyId, item_id: itemId, detail: effRole(u, role) })
    return {
      status: 'pending',
      job: startJob('scenarios', SCEN_MS, () => {
        const set = buildSet(card!)
        sets.set(id, set); spent += set.meta.cost_rub
        return { status: 'done', scenarios: set }
      }),
    }
  },

  async job(jobId) {
    await wait(120)
    me()
    const j = jobs.get(jobId)
    if (!j) return fail(404, 'not_found', 'Задача не найдена')
    if (j.view.status === 'done' || j.view.status === 'failed') return j.view
    if (Date.now() >= j.ready) j.view = { ...j.view, ...j.make(), updated_at: now() }
    else j.view = { ...j.view, status: 'running', updated_at: now() }
    return j.view
  },

  async followup(cardIdArg, question) {
    await wait(1600)
    const u = me()
    const card = [...cards.values()].find(c => c.card_id === cardIdArg)
    if (!card) return fail(404, 'not_found', 'Карточка не найдена')
    const q = question.trim()
    if (!q || q.length > 500) fail(422, 'validation_error', 'Вопрос должен быть от 1 до 500 символов')
    log(u, 'followup', { company_id: card.company_id, item_id: card.item_ids[0], card_id: card.card_id })
    spent += 0.18
    const bankSrc = card.sources.find(s => s.kind === 'bank_analytics')
    const lower = q.toLowerCase()
    let answer: string, used: SourceRef[], inference = false, issues: string[] = []
    if (/процент|сколько|насколько|сумм/.test(lower)) {
      answer = `Точную сумму назвать нельзя: в профиле компании нет нужных цифр (${card.limitations[0]?.toLowerCase() ?? 'данных недостаточно'}). По материалу: ${card.what_happened.text.split('. ')[0]}.`
      used = card.sources.slice(0, 1); issues = ['В ответе нет расчёта: не хватает данных профиля']
    } else if (bankSrc && /банк|прогноз|ожида/.test(lower)) {
      answer = card.bank_view?.text ? `Позиция аналитиков банка: «${card.bank_view.text}». Это прогноз, а не гарантия.` : 'У банка нет материала по этой теме.'
      used = [bankSrc]
    } else {
      answer = `${card.ai_inference.text} Первый шаг: ${card.suggested_actions[0]?.text.toLowerCase() ?? 'дождаться новых данных'}.`
      used = []; inference = true
    }
    const out: FollowupAnswer = { card_id: card.card_id, question: q, answer, source_ids: used.map(s => s.id), sources: used, is_model_inference: inference, verifier_issues: issues, trace_id: `t_f_${Date.now()}` }
    return out
  },

  async feedback(cId, body: FeedbackIn) {
    await wait(300)
    const u = me()
    if (body.rating === 'down' && !body.reason) fail(422, 'validation_error', 'Укажите причину')
    const card = [...cards.values()].find(c => c.card_id === cId)
    votes.push({ role: card?.role ?? u.role, access: u.access, rating: body.rating, reason: body.reason })
    log(u, 'feedback', { company_id: card?.company_id ?? null, card_id: cId, trace_id: card?.trace_id ?? null, detail: body.rating })
    return { ok: true, synced_to_ml: true }
  },

  async cardTrace(cId) {
    await wait(300)
    const u = me()
    const card = [...cards.values()].find(c => c.card_id === cId)
    if (!card) return fail(404, 'not_found', 'Карточка не найдена')
    const hasBank = card.sources.some(s => s.kind === 'bank_analytics')
    const trace: CardTrace = {
      trace_id: card.trace_id, kind: 'insight', duration_s: 6.4, retries: card.meta.retries,
      plan: [
        { tool: 'get_item', why: 'Прочитать материал и даты' },
        { tool: 'get_company_profile', why: 'Взять отрасль, размер и продукты банка' },
        { tool: 'search_bank_analytics', why: 'Найти позицию аналитиков банка по теме' },
        { tool: 'verify_card', why: 'Сверить цифры и цитаты с источниками' },
        { tool: null, why: 'Интерпретация, затем самопроверка' },
      ],
      tool_calls: [{ tool: 'get_item', ms: 12 }, { tool: 'get_company_profile', ms: 8 }, { tool: 'search_bank_analytics', ms: hasBank ? 340 : 290 }, { tool: 'llm.generate', ms: 4800 }, { tool: 'verify_card', ms: 610 }],
      replans: card.meta.retries ? [{ trigger: 'verifier', why: 'Цитата банка не совпала с источником, переписали блок' }] : [],
      checks: card.meta.retries ? [{ attempt: 0, issues: 1 }, { attempt: 1, issues: 0 }] : [{ attempt: 0, issues: 0 }],
      sources: card.sources.map(({ id, kind, title, published_at, doc_id, page, url }) => ({ id, kind, title, published_at, doc_id, page, url })),
      company_features_used: card.why_it_matters.company_features_used,
      model: card.meta.model, prompt_version: card.meta.prompt_version,
    }
    log(u, 'trace', { company_id: card.company_id, card_id: card.card_id, trace_id: card.trace_id })
    if (u.access !== 'user') trace.cost_rub = card.meta.cost_rub
    return trace
  },

  async sourcePdf() { await wait(150); return null },

  async health() { await wait(150); return { ok: true, ml_ok: true, llm_mock: true, spent_today_rub: Math.round(spent * 100) / 100, budget_rub: 500 } },
  async feedbackSummary() {
    await wait()
    const u = me()
    if (u.access === 'user') fail(403, 'forbidden', 'Только для эксперта и администратора')
    return {
      user: grouped(votes.filter(v => v.access === 'user')),
      expert: grouped(votes.filter(v => v.access !== 'user')),
      ml: { 'все': stats(votes), 'insight-v3': stats(votes.slice(4)), 'insight-v2': stats(votes.slice(0, 8)) },
      usage: (() => {
        const actions: Record<string, { count: number; users: number }> = {}
        for (const a of audit) {
          const s = (actions[a.action] ??= { count: 0, users: 0 })
          s.count++
        }
        for (const [k, s] of Object.entries(actions)) s.users = new Set(audit.filter(a => a.action === k).map(a => a.user_id)).size
        const opened = actions.insight?.count ?? 0
        const after = ['followup', 'source', 'trace', 'feedback'].reduce((n, k) => n + (actions[k]?.count ?? 0), 0)
        return { days: 7, actions, actions_per_card: opened ? Math.round((after / opened) * 100) / 100 : null }
      })(),
      items_by_day: Object.fromEntries(Array.from({ length: 14 }, (_, i) => {
        const d = new Date(Date.parse('2026-09-15') - (13 - i) * 864e5).toISOString().slice(0, 10)
        return [d, [3, 5, 4, 0, 6, 7, 5, 4, 8, 6, 5, 9, 7, 4][i]]
      })),
    }
  },
  async audit(limit = 50) {
    await wait()
    const u = me()
    if (u.access !== 'admin') fail(403, 'forbidden', 'Журнал доступен только администратору')
    return audit.slice(0, limit)
  },
}

// Немного истории в журнале, чтобы мониторинг не был пустым при первом открытии.
;(() => {
  const base = Date.now() - 3 * 3600e3
  const rows: [string, string, string | null, string | null][] = [
    ['retail.cfo', 'login', null, null], ['retail.cfo', 'insight', 'U_1595', 'n_2026_09_11_cbr_rate'],
    ['fuel.ceo', 'login', null, null], ['fuel.ceo', 'insight', 'U_26030', 'n_2026_09_03_petrol'],
    ['group.cfo', 'scenarios', 'group_134463_sb', 'n_2026_09_11_steel'], ['retail.cfo', 'feedback', 'U_1595', 'n_2026_09_08_retail'],
  ]
  rows.forEach(([login, action, company_id, item_id], i) => {
    audit.unshift({ id: auditId++, ts: new Date(base + i * 25 * 60e3).toISOString(), user_id: USERS.findIndex(x => x.login === login) + 1, action, company_id, item_id, card_id: null, trace_id: null, detail: null })
  })
})()
