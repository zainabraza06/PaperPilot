import type { Summary, SummaryOrigin } from '@/types/domain'

import { cx } from './primitives'

/**
 * A summary, always shown with where it came from.
 *
 * The backend goes to real trouble to distinguish generated text that
 * passed a grounding check, text that needed a correcting retry, and
 * extractive sentences lifted verbatim from the abstract. Rendering all
 * three identically would throw that away — and would let a reader assume
 * an AI wrote something the abstract actually said, or the reverse.
 *
 * Extractive summaries are labelled "from abstract" rather than being
 * dressed up as AI output. They are the honest fallback, and saying so is
 * the whole point of having tracked the origin.
 */
const ORIGIN_COPY: Record<
  SummaryOrigin,
  { label: string; title: string; className: string }
> = {
  generated: {
    label: 'AI',
    title: 'Generated and verified against the abstract',
    className:
      'bg-accent-50 text-accent-700 ring-accent-600/20 dark:bg-accent-500/10 dark:text-accent-300 dark:ring-accent-400/20',
  },
  regenerated: {
    label: 'AI · corrected',
    title:
      'The first attempt failed the grounding check and was regenerated with the specific problems fed back',
    className:
      'bg-violet-50 text-violet-700 ring-violet-600/20 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-400/20',
  },
  extractive: {
    label: 'From abstract',
    title:
      'Sentences taken verbatim from the abstract — either no model was configured, or generated text could not be grounded',
    className:
      'bg-slate-100 text-slate-600 ring-slate-500/20 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-400/20',
  },
}

export function SummaryBlock({
  summary,
  className,
  full = false,
}: {
  summary: Summary
  className?: string
  full?: boolean
}) {
  const origin = ORIGIN_COPY[summary.origin]
  const unverifiable = summary.grounding.status === 'unverifiable'

  return (
    <div className={cx('rounded-lg bg-slate-50 p-3 dark:bg-slate-800/50', className)}>
      <div className="mb-1.5 flex flex-wrap items-center gap-2">
        <span
          title={origin.title}
          className={cx(
            'inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-semibold ring-1 ring-inset',
            origin.className,
          )}
        >
          {origin.label}
        </span>

        {unverifiable ? (
          <span
            title="This paper has no abstract, so nothing could be checked against"
            className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 dark:text-amber-400"
          >
            <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495ZM10 5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
                clipRule="evenodd"
              />
            </svg>
            unverifiable
          </span>
        ) : (
          <span
            title={`${Math.round(summary.grounding.overlap * 100)}% of the summary's content words appear in the abstract`}
            className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 dark:text-emerald-400"
          >
            <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z"
                clipRule="evenodd"
              />
            </svg>
            grounded
          </span>
        )}

        {summary.cached ? (
          <span className="text-[11px] text-slate-400 dark:text-slate-500">cached</span>
        ) : null}
      </div>

      <p
        className={cx(
          'text-sm leading-relaxed text-slate-700 dark:text-slate-300',
          !full && 'line-clamp-3',
        )}
      >
        {summary.text}
      </p>

      {/* What it took to get here, shown only in the detail view: it is
          useful provenance but would be noise in a scannable list. */}
      {full && summary.rejected_for.length > 0 ? (
        <div className="mt-2 border-t border-slate-200 pt-2 dark:border-slate-700">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
            First attempt rejected for
          </p>
          <ul className="mt-1 space-y-0.5">
            {summary.rejected_for.map((issue, index) => (
              <li
                key={`${issue.kind}-${index}`}
                className="text-xs text-slate-500 dark:text-slate-400"
              >
                {issue.detail}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}
