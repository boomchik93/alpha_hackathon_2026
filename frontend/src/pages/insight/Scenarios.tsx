import { Fragment, useState } from 'react'
import { ChevronDown, Compass, GitFork, Minus, ThumbsDown, ThumbsUp, Waypoints } from 'lucide-react'
import { useScenarios } from '../../api/hooks'
import type { InsightCard, ScenarioBranch, ScenarioKind, ScenarioSet } from '../../api/types'
import { useRequestRole } from '../../auth/session'
import { AiPill, ErrorBanner, Eyebrow, Pill } from '../../components/ui'
import { BASE_BASIS, DECISION_TYPE, EFFECT_LABEL, KIND_LABEL, LIKELIHOOD, SCEN_STEPS } from '../../dict/labels'
import { Claim, Generating, SourceButtons } from './parts'

const KINDS: ScenarioKind[] = ['good', 'base', 'bad']
const LIKELY_TONE = { low: 'neutral', medium: 'warning', high: 'red' } as const
const EFFECT_ICON = { helps: ThumbsUp, neutral: Minus, hurts: ThumbsDown }

// Сценарии развития событий (идея Сони): одна развилка → три исхода → какие решения работают в каждом.
// Строит ML по запросу (ScenarioSet). Вероятности качественные, процентов нет — так честнее.
export function Scenarios({ card, onSource, active }: { card: InsightCard; onSource(id: string): void; active: string | null }) {
  const [requested, setRequested] = useState(false)
  const role = useRequestRole()
  const { set, job, pending, error, retry } = useScenarios(card.company_id, card.item_ids[0], role, requested)
  const notApplicable = error?.code === 'scenarios_not_applicable'

  return <div className="full scen">
    <div className="scenario-head"><div><Eyebrow>Сценарии</Eyebrow><h2>Что будет дальше и что делать в каждом случае</h2></div>{set && <AiPill label="Анализ модели" />}</div>
    {!requested && <div className="scen-start surface">
      <GitFork />
      <div><b>Хороший, базовый и плохой исход</b><p>Модель найдёт главную неопределённость события, распишет три ветки для вашей компании и покажет, какие решения помогают в каждой из них.</p></div>
      <button className="primary-button" onClick={() => setRequested(true)}>Построить сценарии</button>
    </div>}
    {requested && pending && <Generating job={job} steps={SCEN_STEPS} expected={30} title="Строим сценарии" inline />}
    {notApplicable && <div className="limitation"><Compass /><p><b>Для этого события сценарии не строятся.</b> Оно не задаёт развилки, которая заметно меняет положение компании: решения не зависят от исхода.</p></div>}
    {error && !notApplicable && <ErrorBanner error={error} onRetry={retry} />}
    {set && <ScenarioView set={set} onSource={onSource} active={active} />}
  </div>
}

function ScenarioView({ set, onSource, active }: { set: ScenarioSet; onSource(id: string): void; active: string | null }) {
  const { fork, analysis } = set
  const branches = KINDS.map(k => set.branches.find(b => b.kind === k)).filter(Boolean) as ScenarioBranch[]
  const [openKind, setOpenKind] = useState<ScenarioKind | null>('base')
  const [closing, setClosing] = useState(false)
  const [nextKind, setNextKind] = useState<ScenarioKind | null>(null)

  const toggleBranch = (kind: ScenarioKind) => {
    if (closing) return
    if (openKind === kind) {
      setNextKind(null)
      setClosing(true)
    } else if (openKind) {
      setNextKind(kind)
      setClosing(true)
    } else {
      setOpenKind(kind)
    }
  }

  const finishClosing = () => {
    setClosing(false)
    setOpenKind(nextKind)
    setNextKind(null)
  }
  return <>
    <div className="scen-fork">
      <span className="scen-fork-icon"><Waypoints /></span>
      <div>
        <small>Главная неопределённость</small>
        <b>{fork.variable}</b>
        <span className="scen-basis">Базовый исход: {BASE_BASIS[fork.base_basis] ?? fork.base_basis}{fork.evidence.length > 0 && <> · <SourceButtons ids={fork.evidence} sources={set.sources} onSource={onSource} active={active} /></>}</span>
      </div>
    </div>

    <div className="scen-branches">
      {branches.map(b => <Fragment key={b.kind}>
        <Branch b={b} outcome={fork.outcomes[b.kind]} open={openKind === b.kind} onToggle={() => toggleBranch(b.kind)} />
        {openKind === b.kind && <BranchDetails b={b} set={set} onSource={onSource} active={active} closing={closing} onClose={() => toggleBranch(b.kind)} onClosed={finishClosing} />}
      </Fragment>)}
    </div>

    <div className="scen-summary">
      <p>{analysis.summary}</p>
      {analysis.key_uncertainty && <p className="muted">От чего зависит больше всего: {analysis.key_uncertainty.charAt(0).toLowerCase() + analysis.key_uncertainty.slice(1)}.</p>}
    </div>

    {analysis.decisions.length > 0 && <div className="scen-matrix surface">
      <div className="scen-matrix-row head"><span>Решение</span>{KINDS.map(k => <span key={k} className={`k-${k}`}>{KIND_LABEL[k]}</span>)}</div>
      {analysis.decisions.map(d => <div className="scen-matrix-row" key={d.text}>
        <div className="scen-decision"><b>{d.text}</b><Pill tone={d.type === 'robust' ? 'success' : d.type === 'hedge' ? 'warning' : 'neutral'}>{DECISION_TYPE[d.type]}</Pill><p>{d.rationale}</p>{d.trigger && <small>Сигнал к запуску: {d.trigger}</small>}</div>
        {KINDS.map(k => { const e = d.effects[k]; const Icon = EFFECT_ICON[e]; return <span key={k} className={`scen-effect ${e}`} data-kind={KIND_LABEL[k]}><Icon size={14} /> {EFFECT_LABEL[e]}</span> })}
      </div>)}
    </div>}

    <div className="scen-reco">
      <small>Что из этого следует</small>
      <b>{analysis.recommendation}</b>
      {analysis.switch_signals.length > 0 && <div className="scen-signals"><span>Следить за сигналами:</span>{analysis.switch_signals.map(x => <Pill key={x}>{x}</Pill>)}</div>}
    </div>
    {analysis.limitations.map(l => <div key={l} className="limitation"><Compass /><p><b>Ограничение:</b> {l}</p></div>)}
    <p className="scenario-note">Это анализ модели на основе карточки, профиля компании и аналитики банка, а не рекомендация банка. Вероятность оценена словами, без процентов: точных цифр для неё нет.</p>
  </>
}

function Branch({ b, outcome, open, onToggle }: { b: ScenarioBranch; outcome: string; open: boolean; onToggle(): void }) {
  return <article className={`scen-branch ${b.kind}${open ? ' is-open' : ''}`}>
    <div className="scen-branch-top"><span className="scen-kind">{KIND_LABEL[b.kind]}</span><Pill tone={LIKELY_TONE[b.likelihood]} title={b.likelihood_basis}>{LIKELIHOOD[b.likelihood]}</Pill></div>
    <h3>{b.title}</h3>
    <p className="scen-outcome">{outcome}</p>
    <p className="scen-impact"><b>Для компании:</b> {b.impact}</p>
    <button className="scen-more-btn" onClick={onToggle} aria-expanded={open} aria-controls={`scenario-details-${b.kind}`}>{open ? 'Свернуть' : 'Подробнее'} <ChevronDown size={15} className={open ? 'rotated' : ''} /></button>
  </article>
}

function BranchDetails({ b, set, onSource, active, closing, onClose, onClosed }: { b: ScenarioBranch; set: ScenarioSet; onSource(id: string): void; active: string | null; closing: boolean; onClose(): void; onClosed(): void }) {
  return <div
    id={`scenario-details-${b.kind}`}
    className={`scen-details-shell ${b.kind}${closing ? ' is-closing' : ''}`}
    onAnimationEnd={event => { if (closing && event.target === event.currentTarget) onClosed() }}
  >
  <section className={`scen-details surface ${b.kind}`}>
    <div className="scen-details-head">
      <div><span className="scen-kind">{KIND_LABEL[b.kind]}</span><h3>{b.title}</h3></div>
      <button className="scen-more-btn" onClick={onClose}>Свернуть <ChevronDown size={15} className="rotated" /></button>
    </div>
    <div className="scen-details-intro">
      <p><b>Допущение:</b> {b.assumption}</p>
      <p><b>Почему такая оценка:</b> {b.likelihood_basis}</p>
    </div>
    <div className="scen-details-grid">
      {b.risks.length > 0 && <div><span className="scen-label">Риски</span>{b.risks.map(c => <Claim key={c.text} c={c} sources={set.sources} onSource={onSource} active={active} />)}</div>}
      {b.opportunities.length > 0 && <div><span className="scen-label">Возможности</span>{b.opportunities.map(c => <Claim key={c.text} c={c} sources={set.sources} onSource={onSource} active={active} />)}</div>}
      {b.actions.length > 0 && <div><span className="scen-label">Что делать</span><ul>{b.actions.map(a => <li key={a.text}>{a.text}</li>)}</ul></div>}
      {b.signals.length > 0 && <div><span className="scen-label">Признаки, что идём по этой ветке</span><ul>{b.signals.map(x => <li key={x}>{x}</li>)}</ul></div>}
    </div>
    {b.limitations.map(l => <p key={l} className="scen-detail-limitation">Ограничение: {l}</p>)}
  </section>
  </div>
}
