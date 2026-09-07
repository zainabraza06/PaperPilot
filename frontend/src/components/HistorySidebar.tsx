import type { HistoryEntry, QueryIntent } from '@/types/domain'

import { cx } from './primitives'

const INTENT_SHORT: Record<QueryIntent, string> = {
  keyword: 'keyword',
  topic: 'topic',
  identifier: 'id',
  abstract_snippet: 'abstract',
}

/**
 * Recent queries.
 *
 * The brief frames this as a tool used daily, and the thing a daily user
 * reaches for most is the search they ran yesterday. Each entry records
 * what it was interpreted as and how many papers came back, so the list
 * reads as a record of work rather than a log of keystrokes.
 */
export function HistorySidebar({
  history,
  activeQuery,
  onSelect,
  onRemove,
  onClear,
}: {
  history: HistoryEntry[]
  activeQuery: string | null
  onSelect: (query: string) => void
  onRemove: (query: string) => void
  onClear: () => void
}) {
  return (
    <aside className="flex h-full flex-col">
      <div className="flex items-center justify-between px-1 pb-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
          Recent searches
        </h2>
        {history.length > 0 ? (
          <button
            type="button"
            onClick={onClear}
            className="rounded px-1 text-xs text-slate-400 transition hover:text-slate-700 dark:hover:text-slate-200"
          >
            Clear
          </button>
        ) : null}
      </div>

      {history.length === 0 ? (
        <p className="px-1 text-xs leading-relaxed text-slate-400 dark:text-slate-500">
          Searches you run are kept here, in this browser only.
        </p>
      ) : (
        <ul className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
          {history.map((entry) => {
            const active = entry.query === activeQuery
            return (
              <li key={entry.query} className="group relative">
                <button
                  type="button"
                  onClick={() => onSelect(entry.query)}
                  className={cx(
                    'w-full rounded-lg px-2.5 py-2 pr-7 text-left transition',
                    active
                      ? 'bg-accent-50 dark:bg-accent-500/10'
                      : 'hover:bg-slate-100 dark:hover:bg-slate-800',
                  )}
                >
                  <span
                    className={cx(
                      'line-clamp-2 text-sm leading-snug',
                      active
                        ? 'font-medium text-accent-800 dark:text-accent-200'
                        : 'text-slate-700 dark:text-slate-300',
                    )}
                  >
                    {entry.query}
                  </span>
                  <span className="mt-1 flex items-center gap-1.5 text-[11px] text-slate-400 dark:text-slate-500">
                    <span className="rounded bg-slate-100 px-1 dark:bg-slate-800">
                      {INTENT_SHORT[entry.intent]}
                    </span>
                    <span>{entry.resultCount} results</span>
                    <span aria-hidden="true">·</span>
                    <span>{relativeTime(entry.ranAt)}</span>
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => onRemove(entry.query)}
                  aria-label={`Remove ${entry.query} from history`}
                  className="absolute right-1 top-1.5 rounded p-1 text-slate-300 opacity-0 transition hover:bg-slate-200 hover:text-slate-600 focus-visible:opacity-100 group-hover:opacity-100 dark:text-slate-600 dark:hover:bg-slate-700 dark:hover:text-slate-200"
                >
                  <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                  </svg>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </aside>
  )
}

/** Coarse relative time — precision past "yesterday" is not useful here. */
function relativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days === 1) return 'yesterday'
  if (days < 30) return `${days}d ago`
  return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
