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
  ok: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-400/20',
  neutral:
    'bg-slate-100 text-slate-600 ring-slate-500/20 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-400/20',
  warn: 'bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-400/20',
  error:
    'bg-rose-50 text-rose-700 ring-rose-600/20 dark:bg-rose-500/10 dark:text-rose-300 dark:ring-rose-400/20',
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
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset',
        TONE_CLASSES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

const SOURCE_CLASSES: Record<SourceName, string> = {
  pubmed:
    'bg-blue-50 text-blue-700 ring-blue-600/20 dark:bg-blue-500/10 dark:text-blue-300 dark:ring-blue-400/20',
  arxiv:
    'bg-red-50 text-red-700 ring-red-600/20 dark:bg-red-500/10 dark:text-red-300 dark:ring-red-400/20',
  crossref:
    'bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-400/20',
}

/**
 * Source badges, colour-coded so provenance is recognisable before it is
 * read. A merged record shows every source that corroborated it, which is
 * the visible payoff of cross-source deduplication.
 */
export function SourceBadges({ sources }: { sources: SourceName[] }) {
  return (
    <span className="inline-flex items-center gap-1">
      {sources.map((source) => (
        <span
          key={source}
          className={cx(
            'inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-semibold ring-1 ring-inset',
            SOURCE_CLASSES[source],
          )}
        >
          {SOURCE_LABELS[source]}
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
    <div
      className="flex items-center gap-2"
      title={detail}
      role="img"
      aria-label={detail}
    >
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        <div
          className="h-full rounded-full bg-accent-500 transition-[width] duration-500 ease-out"
          style={{ width: `${Math.max(percent, 2)}%` }}
        />
      </div>
      <span className="tabular-nums text-xs text-slate-500 dark:text-slate-400">
        {percent}
      </span>
    </div>
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

export function IconButton({
  onClick,
  label,
  children,
  className,
}: {
  onClick: () => void
  label: string
  children: ReactNode
  className?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cx(
        'inline-flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100',
        className,
      )}
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
