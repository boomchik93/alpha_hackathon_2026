import { useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import { Bot, Check, ChevronDown, CircleAlert, RotateCw, X } from 'lucide-react'
import type { ApiError } from '../api/types'
import { ERROR_TEXT } from '../dict/labels'

export const pct = (score: number) => `${Math.round(score * 100)}%`
export const ringStyle = (score: number) => ({ '--score': Math.round(score * 100) } as CSSProperties)

export function Logo({ compact = false }: { compact?: boolean }) {
  return <div className="logo" aria-label="Альфа Бизнес"><span className="logo-mark">A</span>{!compact && <span>Альфа <b>Бизнес</b></span>}</div>
}

export function Eyebrow({ children }: { children: ReactNode }) { return <div className="eyebrow">{children}</div> }

export function Pill({ children, tone = 'neutral', title }: { children: ReactNode; tone?: string; title?: string }) {
  return <span className={`pill ${tone}`} title={title}>{children}</span>
}

export function AiPill({ label = 'Вывод модели' }: { label?: string }) {
  return <Pill tone="ai"><Bot size={13} /> {label}</Pill>
}

export function PageHeader({ eyebrow, title, subtitle, action }: { eyebrow?: string; title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return <div className="page-header"><div>{eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}<h1>{title}</h1>{subtitle && <p>{subtitle}</p>}</div>{action}</div>
}

export function ScoreRing({ score, dark = false }: { score: number; dark?: boolean }) {
  return <span className="score-ring" style={{ ...ringStyle(score), ...(dark ? { '--ring-bg': '#171719' } as CSSProperties : {}) }} title={`Релевантность ${pct(score)}`}><b>{pct(score)}</b></span>
}

export function Skel({ w = '100%', h = 12 }: { w?: number | string; h?: number }) {
  return <span className="skel" style={{ width: w, height: h }} />
}

export function ErrorBanner({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const soft = error.status === 429
  return (
    <div className={`error-banner ${soft ? 'soft' : ''}`} role="alert">
      <CircleAlert />
      <div>
        <b>{ERROR_TEXT[error.code] ?? 'Что-то пошло не так'}</b>
        {error.message && error.message !== ERROR_TEXT[error.code] && <p>{error.message}</p>}
      </div>
      {onRetry && !soft && error.status !== 403 && error.status !== 404 && error.code !== 'scenarios_not_applicable' && <button className="secondary-button" onClick={onRetry}><RotateCw size={16} /> Повторить</button>}
    </div>
  )
}

export function Modal({ title, eyebrow, onClose, children, wide = false }: { title: string; eyebrow?: string; onClose(): void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const h = (e: globalThis.KeyboardEvent) => e.key === 'Escape' && onClose()
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', h)
    return () => {
      window.removeEventListener('keydown', h)
      document.body.style.overflow = previousOverflow
    }
  }, [onClose])
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className={`modal surface ${wide ? 'wide' : ''}`} role="dialog" aria-modal="true" aria-label={title} onClick={e => e.stopPropagation()}>
        <div className="modal-head"><div>{eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}<h2>{title}</h2></div><button className="icon-button" onClick={onClose} aria-label="Закрыть"><X size={20} /></button></div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}

type DropdownOption = { value: string; label: string }

export function Dropdown({ value, onChange, options, ariaLabel, compact = false }: {
  value: string; onChange: (value: string) => void; options: DropdownOption[]; ariaLabel: string; compact?: boolean
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const listId = useId()
  const current = options.find(option => option.value === value) ?? options[0]

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: MouseEvent) => { if (!rootRef.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
  }, [open])

  useEffect(() => {
    if (!open) return
    requestAnimationFrame(() => { rootRef.current?.querySelector<HTMLButtonElement>('[aria-selected="true"]')?.focus() })
  }, [open])

  const onOptionKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const items = Array.from(rootRef.current?.querySelectorAll<HTMLButtonElement>('.dropdown-option') ?? [])
    const index = items.indexOf(event.currentTarget)
    if (event.key === 'ArrowDown') { event.preventDefault(); items[(index + 1) % items.length]?.focus() }
    if (event.key === 'ArrowUp') { event.preventDefault(); items[(index - 1 + items.length) % items.length]?.focus() }
    if (event.key === 'Home') { event.preventDefault(); items[0]?.focus() }
    if (event.key === 'End') { event.preventDefault(); items.at(-1)?.focus() }
    if (event.key === 'Escape' || event.key === 'Tab') {
      setOpen(false)
      if (event.key === 'Escape') { event.preventDefault(); buttonRef.current?.focus() }
    }
  }

  if (!current) return null
  return <div className={`dropdown ${open ? 'is-open' : ''} ${compact ? 'is-compact' : ''}`} ref={rootRef}>
    <button ref={buttonRef} type="button" className="dropdown-trigger" aria-label={ariaLabel} aria-expanded={open} aria-controls={listId} aria-haspopup="listbox"
      onClick={() => setOpen(!open)}
      onKeyDown={event => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setOpen(true) }
        if (event.key === 'Escape') setOpen(false)
      }}>
      <span>{current.label}</span><ChevronDown size={15} />
    </button>
    {open && <div className="dropdown-menu" id={listId} role="listbox" aria-label={ariaLabel}>
      {options.map(option => <button type="button" role="option" aria-selected={option.value === value} className="dropdown-option" key={option.value}
        onKeyDown={onOptionKeyDown}
        onClick={() => { onChange(option.value); setOpen(false); buttonRef.current?.focus() }}>
        <span>{option.label}</span>{option.value === value && <Check size={15} />}
      </button>)}
    </div>}
  </div>
}

// localStorage может быть недоступен (приватный режим, превью) — тогда тема просто не запоминается.
export const store = {
  get(k: string) { try { return localStorage.getItem(k) } catch { return null } },
  set(k: string, v: string) { try { localStorage.setItem(k, v) } catch { /* ignore */ } },
}
