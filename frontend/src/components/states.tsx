import type { ApiError } from '@/lib/api'

import { Button } from './primitives'

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

/**
 * The skeleton mirrors the real card's geometry — same paddings, same rank
 * gutter, same line widths. A skeleton whose shape does not match what
 * replaces it produces a visible jolt on load, which is worse than no
 * skeleton at all.
 */
export function ResultSkeleton() {
  const widths = ['w-[92%]', 'w-[76%]', 'w-[88%]', 'w-[68%]']
  return (
    <div className="results-panel" aria-hidden="true">
      {widths.map((width, index) => (
        <div key={index} className="row px-4 py-4 sm:px-6 sm:py-5">
          <div className="flex gap-3 sm:gap-4">
            <div className="hidden w-7 shrink-0 sm:block">
              <div className="skeleton ms-auto h-3.5 w-3" />
            </div>
            <div className="min-w-0 flex-1 space-y-2.5">
              <div className={`skeleton h-4 ${width}`} />
              <div className="skeleton h-3 w-1/3" />
              <div className="flex gap-2 pt-0.5">
                <div className="skeleton h-2.5 w-16" />
                <div className="skeleton h-2.5 w-14" />
                <div className="skeleton h-2.5 w-24" />
              </div>
              <div className="space-y-1.5 pt-1.5">
                <div className="skeleton h-3 w-full" />
                <div className="skeleton h-3 w-[94%]" />
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export function SearchingBanner({ query }: { query: string }) {
  return (
    <div
      className="flex items-center gap-3 rounded-xl border border-line bg-surface px-4 py-3 text-base"
      role="status"
      aria-live="polite"
    >
      <span className="flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent-500"
            style={{ animationDelay: `${index * 160}ms` }}
          />
        ))}
      </span>
      <span className="min-w-0 flex-1 truncate text-muted">
        Searching PubMed, arXiv and Crossref for{' '}
        <span className="font-medium text-strong">
          {query.length > 60 ? `${query.slice(0, 60)}…` : query}
        </span>
      </span>
    </div>
  )
}

/**
 * The first screen. It has one job beyond looking finished: say what makes
 * this different from a single-database search box, because a user who
 * types one word and gets fifty results will never otherwise learn that
 * three sources were merged, deduplicated and checked.
 */
export function WelcomeState() {
  const capabilities = [
    {
      title: 'Three sources, one list',
      body: 'PubMed, arXiv and Crossref queried in parallel, then deduplicated by DOI and by title so a preprint and its published version arrive as one result.',
    },
    {
      title: 'Ranked, then grouped',
      body: 'Hybrid semantic and BM25 scoring, measured against a hand-judged set of 411 papers. Sub-topics appear only when the results genuinely split.',
    },
    {
      title: 'Summaries you can check',
      body: 'Every AI summary is verified against its own abstract. Anything that fails is corrected, or replaced with the abstract’s own sentences and labelled as such.',
    },
  ]

  return (
    <div className="animate-fade-in py-6">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-2xl font-semibold text-strong">Search three databases at once</h2>
        <p className="mx-auto mt-2.5 max-w-lg text-md text-muted">
          A topic, a keyword, a DOI or arXiv id — or paste a whole abstract and find work
          like it.
        </p>
      </div>

      <ul className="mx-auto mt-8 grid max-w-4xl gap-3 sm:grid-cols-3">
        {capabilities.map((item) => (
          <li key={item.title} className="rounded-xl border border-line bg-surface p-4">
            <h3 className="text-base font-semibold text-strong">{item.title}</h3>
            <p className="mt-1.5 text-sm text-muted">{item.body}</p>
          </li>
        ))}
      </ul>
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
    <div className="rounded-xl border border-line bg-surface px-6 py-14 text-center">
      <h2 className="text-lg font-semibold text-strong">
        {isIdentifier ? 'No paper with that identifier' : 'No papers matched'}
      </h2>
      <p className="mx-auto mt-2 max-w-md text-base text-muted">
        {isIdentifier ? (
          <>
            None of the three sources hold a record for{' '}
            <span className="font-mono text-body">{query}</span>. Check the identifier, or
            search for the title instead.
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
    <div
      className="rounded-xl border border-critical/30 bg-surface px-6 py-12 text-center"
      role="alert"
    >
      <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-critical/[0.08] text-critical">
        <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
            clipRule="evenodd"
          />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-strong">
        {error.offline ? 'Cannot reach the API' : 'That search failed'}
      </h2>
      <p className="mx-auto mt-2 max-w-md text-base text-muted">{error.message}</p>
      {error.offline ? (
        <pre className="mx-auto mt-4 w-fit rounded-lg border border-line bg-sunken px-3 py-2 text-left text-xs text-muted">
          cd backend{'\n'}
          uvicorn app.main:app --reload
        </pre>
      ) : null}
      <Button variant="primary" onClick={onRetry} className="mt-6">
        Try again
      </Button>
    </div>
  )
}
