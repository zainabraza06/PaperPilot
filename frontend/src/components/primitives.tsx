/**
 * Small shared pieces.
 *
 * Kept together because each is a handful of lines and splitting them
 * across files would make the visual language harder to keep consistent,
 * not easier — the point of these is that every badge in the app looks
 * like every other badge.
 */

import type { ReactNode } from 'react'

import type { SourceName, Tone } from '@/types/domain'
import { SOURCE_LABELS } from '@/types/domain'

export function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ')
}

const TONE_CLASSES: Record<Tone, string> = {
  ok: 'text-positive bg-positive/[0.08]',
  neutral: 'text-muted bg-sunken',
  warn: 'text-caution bg-caution/[0.08]',
  error: 'text-critical bg-critical/[0.08]',
}

export function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode
  tone?: Tone
  className?: string
  title?: string
}) {
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-2xs font-medium',
        TONE_CLASSES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

const SOURCE_COLOURS: Record<SourceName, string> = {
  pubmed: 'text-source-pubmed',
  arxiv: 'text-source-arxiv',
  crossref: 'text-source-crossref',
}

/**
 * Source provenance, as a coloured dot and a name.
 *
 * Previously a filled pill per source. On a card that also carries an
 * origin chip, a grounding chip and entity chips, three more pills made
 * the row read as a toolbar rather than as metadata — so provenance is
 * now the quietest possible thing that still encodes the source in colour
 * *and* in text, which is what keeps it legible to colour-blind readers
 * (WCAG 1.4.1: colour is never the only channel).
 *
 * A merged record shows every source that corroborated it, which is the
 * visible payoff of cross-source deduplication.
 */
export function SourceBadges({ sources }: { sources: SourceName[] }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {sources.map((source, index) => (
        <span key={source} className="inline-flex items-center gap-1.5">
          {index > 0 ? <span className="text-faint">+</span> : null}
          <span className={cx('inline-flex items-center gap-1 font-medium', SOURCE_COLOURS[source])}>
            <svg className="h-1.5 w-1.5" viewBox="0 0 6 6" aria-hidden="true">
              <circle cx="3" cy="3" r="3" fill="currentColor" />
            </svg>
            {SOURCE_LABELS[source]}
          </span>
        </span>
      ))}
    </span>
  )
}

/**
 * Relevance as a bar rather than a number.
 *
 * A raw 0.87 means nothing without the rest of the list to compare it to;
 * a bar is comparable at a glance down a column. The numeric score and its
 * two components stay in the tooltip for anyone who wants them.
 *
 * `role="img"` with a full label rather than a progressbar: this is not a
 * task completing, and a screen reader announcing "progress bar, 87%" on
 * every one of fifty rows would be actively misleading.
 */
export function ScoreBar({
  value,
  semantic,
  lexical,
}: {
  value: number
  semantic?: number
  lexical?: number
}) {
  const percent = Math.round(value * 100)
  const detail =
    semantic !== undefined && lexical !== undefined
      ? `Relevance ${percent}% · semantic ${semantic.toFixed(3)} · BM25 ${lexical.toFixed(2)}`
      : `Relevance ${percent}%`

  return (
    <span
      className="inline-flex items-center gap-2"
      title={detail}
      role="img"
      aria-label={detail}
    >
      <span className="h-1 w-12 overflow-hidden rounded-full bg-line-strong">
        <span
          className="block h-full rounded-full bg-accent-500 transition-[width] duration-500 ease-out"
          style={{ width: `${Math.max(percent, 3)}%` }}
        />
      </span>
      <span className="w-6 text-right text-2xs font-medium tabular-nums text-muted">
        {percent}
      </span>
    </span>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cx('animate-spin', className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M12 2a10 10 0 0 1 10 10h-3a7 7 0 0 0-7-7V2Z"
      />
    </svg>
  )
}

/**
 * An icon-only control.
 *
 * 36px square: WCAG 2.2 asks for a 24px minimum target (2.5.8) and the
 * pointer-comfort guidance for a primary control is closer to 40, so this
 * sits deliberately above the floor rather than at it.
 */
export function IconButton({
  onClick,
  label,
  children,
  className,
  active = false,
}: {
  onClick: () => void
  label: string
  children: ReactNode
  className?: string
  active?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cx(
        'inline-flex h-9 w-9 items-center justify-center rounded-lg transition duration-150',
        active
          ? 'bg-sunken text-strong'
          : 'text-muted hover:bg-sunken hover:text-strong active:scale-[0.96]',
        className,
      )}
    >
      {children}
    </button>
  )
}

/** The one button shape, so a primary action looks the same everywhere. */
export function Button({
  children,
  onClick,
  variant = 'secondary',
  size = 'md',
  type = 'button',
  className,
  disabled,
  ...rest
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md'
  type?: 'button' | 'submit'
  className?: string
  disabled?: boolean
} & Record<string, unknown>) {
  const variants = {
    primary:
      'bg-accent-600 text-accent-fg shadow-subtle hover:bg-accent-500 active:scale-[0.98]',
    secondary:
      'bg-surface text-body ring-1 ring-inset ring-control hover:bg-sunken hover:text-strong active:scale-[0.98]',
    ghost: 'text-muted hover:bg-sunken hover:text-strong',
  }
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cx(
        'inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition duration-150 disabled:pointer-events-none disabled:opacity-50',
        size === 'sm' ? 'h-8 px-2.5 text-xs' : 'h-9 px-3.5 text-sm',
        variants[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return 'no date'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return 'no date'
  return parsed.toLocaleDateString(undefined, { year: 'numeric', month: 'short' })
}

export function formatAuthors(names: string[], limit = 3): string {
  if (names.length === 0) return 'No authors listed'
  if (names.length <= limit) return names.join(', ')
  return `${names.slice(0, limit).join(', ')} +${names.length - limit} more`
}
