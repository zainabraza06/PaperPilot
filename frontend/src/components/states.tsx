import type { ApiError } from '@/lib/api'

import { cx } from './primitives'

/**
 * Loading, empty and error states, written for this pipeline rather than
 * borrowed from a generic template.
 *
 * A multi-source fan-out takes seconds, so the loading state names what is
 * happening instead of spinning anonymously. "No results" for a DOI lookup
 * means something different from "no results" for a topic, so the empty
 * state says which. And an unreachable API is a development-time mistake
 * with a specific fix, so it prints the command rather than apologising.
 */

export function ResultSkeleton() {
  return (
    <div className="space-y-3" aria-hidden="true">
      {[0, 1, 2, 3].map((index) => (
        <div key={index} className="surface space-y-3 p-4">
          <div className="skeleton h-4 w-3/4" />
          <div className="skeleton h-3 w-1/3" />
          <div className="space-y-2 pt-1">
            <div className="skeleton h-3 w-full" />
            <div className="skeleton h-3 w-11/12" />
          </div>
        </div>
      ))}
    </div>
  )
}

export function SearchingBanner({ query }: { query: string }) {
  return (
    <div className="surface flex items-center gap-3 px-4 py-3 text-sm">
      <span className="flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            className="h-2 w-2 animate-pulse rounded-full bg-accent-500"
            style={{ animationDelay: `${index * 160}ms` }}
          />
        ))}
      </span>
      <span className="min-w-0 flex-1 truncate text-slate-600 dark:text-slate-300">
        Searching PubMed, arXiv and Crossref for{' '}
        <span className="font-medium text-slate-900 dark:text-slate-100">
          {query.length > 60 ? `${query.slice(0, 60)}…` : query}
        </span>
      </span>
    </div>
  )
}

export function WelcomeState() {
  return (
    <div className="surface px-6 py-14 text-center">
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-50 text-accent-600 dark:bg-accent-500/10 dark:text-accent-400">
        <svg className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <path
            fillRule="evenodd"
            d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
            clipRule="evenodd"
          />
        </svg>
      </div>
      <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
        Search three databases at once
      </h2>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-slate-500 dark:text-slate-400">
        PubMed, arXiv and Crossref, queried together and merged into one ranked list —
        deduplicated across sources, grouped by sub-topic, and summarized against the
        abstract rather than around it.
      </p>
    </div>
  )
}

export function EmptyResults({
  query,
  isIdentifier,
}: {
  query: string
  isIdentifier: boolean
}) {
  return (
    <div className="surface px-6 py-12 text-center">
      <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
        {isIdentifier ? 'No paper with that identifier' : 'No papers matched'}
      </h2>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-slate-500 dark:text-slate-400">
        {isIdentifier ? (
          <>
            None of the three sources hold a record for{' '}
            <span className="font-mono text-slate-700 dark:text-slate-300">{query}</span>.
            Check the identifier, or search for the title instead.
          </>
        ) : (
          <>
            All three sources answered, and none had a match. Try broader terms — ranking
            is what narrows results down, so a wide search usually reads better here than
            a precise one.
          </>
        )}
      </p>
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <div className={cx('surface px-6 py-10 text-center', 'border-rose-200 dark:border-rose-500/30')}>
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400">
        <svg className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
            clipRule="evenodd"
          />
        </svg>
      </div>
      <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
        {error.offline ? 'Cannot reach the API' : 'That search failed'}
      </h2>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-slate-500 dark:text-slate-400">
        {error.message}
      </p>
      {error.offline ? (
        <pre className="mx-auto mt-4 w-fit rounded-lg bg-slate-100 px-3 py-2 text-left text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          cd backend{'\n'}
          uvicorn app.main:app --reload
        </pre>
      ) : null}
      <button
        type="button"
        onClick={onRetry}
        className="mt-5 rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
      >
        Try again
      </button>
    </div>
  )
}
