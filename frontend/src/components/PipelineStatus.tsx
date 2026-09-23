import { useState } from 'react'

import type { SearchResponse } from '@/types/domain'
import { SOURCE_LABELS, SOURCE_STATUS_COPY } from '@/types/domain'

import { Badge, cx } from './primitives'

/**
 * What each stage of the pipeline did, shown rather than hidden.
 *
 * This is the component the backend's report objects were designed for.
 * Every stage reports its own outcome — sources, ranking, clustering,
 * entities, summaries — and each can degrade independently. A tool that
 * quietly returned two sources' worth of results while a third was down
 * would be lying by omission; "no results from PubMed, showing arXiv and
 * Crossref only" is a real state a researcher needs to see, because it
 * changes how much they trust the completeness of what they are reading.
 *
 * The collapsed row is a summary; the expanded panel is the detail. Most
 * searches succeed entirely and deserve one quiet line.
 */
export function PipelineStatus({ response }: { response: SearchResponse }) {
  const [expanded, setExpanded] = useState(false)

  const failed = response.sources.filter(
    (report) => !['ok', 'empty', 'skipped'].includes(report.status),
  )
  const empty = response.sources.filter((report) => report.status === 'empty')
  const degraded = failed.length > 0

  return (
    <div
      className={cx(
        'surface-panel overflow-hidden text-sm',
        degraded && 'border-amber-300 dark:border-amber-500/40',
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded((open) => !open)}
        aria-expanded={expanded}
        className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition hover:bg-sunken/50"
      >
        <span className="flex items-center gap-1.5">
          {response.sources.map((report) => {
            const copy = SOURCE_STATUS_COPY[report.status]
            return (
              <span
                key={report.source}
                title={`${SOURCE_LABELS[report.source]}: ${copy.label}`}
                className={cx(
                  'h-2 w-2 rounded-full',
                  copy.tone === 'ok' && 'bg-emerald-500',
                  copy.tone === 'neutral' && 'bg-line-strong',
                  copy.tone === 'warn' && 'bg-amber-500',
                  copy.tone === 'error' && 'bg-rose-500',
                )}
              />
            )
          })}
        </span>

        <span className="min-w-0 flex-1 truncate text-body">
          <strong className="font-semibold text-strong">
            {response.total}
          </strong>{' '}
          {response.total === 1 ? 'paper' : 'papers'}
          {response.duplicates_merged > 0 ? (
            <>
              {' '}
              · {response.duplicates_merged} duplicate
              {response.duplicates_merged === 1 ? '' : 's'} merged
            </>
          ) : null}{' '}
          · {(response.elapsed_ms / 1000).toFixed(1)}s
          {degraded ? (
            <>
              {' '}
              ·{' '}
              <span className="font-medium text-amber-700 dark:text-amber-400">
                {failed.length === 1
                  ? `${SOURCE_LABELS[failed[0]!.source]} unavailable`
                  : `${failed.length} sources unavailable`}
              </span>
            </>
          ) : empty.length > 0 ? (
            <>
              {' '}
              · no matches in{' '}
              {empty.map((report) => SOURCE_LABELS[report.source]).join(' or ')}
            </>
          ) : null}
        </span>

        <svg
          className={cx(
            'h-4 w-4 shrink-0 text-faint transition-transform',
            expanded && 'rotate-180',
          )}
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {expanded ? (
        <div className="animate-fade-in space-y-3 border-t border-line px-4 py-3">
          <Section title="Sources">
            {response.sources.map((report) => {
              const copy = SOURCE_STATUS_COPY[report.status]
              return (
                <Row
                  key={report.source}
                  label={SOURCE_LABELS[report.source]}
                  tone={copy.tone}
                  value={
                    report.status === 'ok'
                      ? `${report.returned} results in ${report.elapsed_ms} ms`
                      : copy.label
                  }
                  note={report.message}
                />
              )
            })}
          </Section>

          <Section title="Processing">
            <Row
              label="Ranking"
              tone={response.ranking.applied ? 'ok' : 'warn'}
              value={
                response.ranking.applied
                  ? `${response.ranking.strategy} on ${response.ranking.model} · ${response.ranking.elapsed_ms} ms`
                  : 'not applied — results are in retrieval order'
              }
              note={response.ranking.reason}
            />
            <Row
              label="Clusters"
              tone={response.clustering.applied ? 'ok' : 'neutral'}
              value={
                response.clustering.applied
                  ? `${response.clustering.clusters} sub-topics · silhouette ${response.clustering.silhouette ?? '—'}`
                  : 'not split'
              }
              note={response.clustering.reason}
            />
            <Row
              label="Entities"
              tone={response.entities.applied ? 'ok' : 'neutral'}
              value={
                response.entities.applied
                  ? `${response.entities.entities_found} found via ${response.entities.model}`
                  : 'not extracted'
              }
              note={response.entities.reason}
            />
            <Row
              label="Summaries"
              tone={response.summaries.applied ? 'ok' : 'neutral'}
              value={
                response.summaries.applied
                  ? [
                      `${response.summaries.summarized} via ${response.summaries.model}`,
                      response.summaries.from_cache > 0 &&
                        `${response.summaries.from_cache} cached`,
                      response.summaries.regenerated > 0 &&
                        `${response.summaries.regenerated} regenerated`,
                      response.summaries.fell_back > 0 &&
                        `${response.summaries.fell_back} fell back`,
                    ]
                      .filter(Boolean)
                      .join(' · ')
                  : 'not generated'
              }
              note={response.summaries.reason}
            />
          </Section>
        </div>
      ) : null}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-faint">
        {title}
      </h3>
      <dl className="space-y-1">{children}</dl>
    </div>
  )
}

function Row({
  label,
  value,
  tone,
  note,
}: {
  label: string
  value: string
  tone: 'ok' | 'neutral' | 'warn' | 'error'
  note?: string | null
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
      <dt className="w-20 shrink-0 text-muted">{label}</dt>
      <dd className="flex flex-wrap items-baseline gap-2">
        <Badge tone={tone}>{value}</Badge>
        {note ? (
          <span className="text-xs text-muted">{note}</span>
        ) : null}
      </dd>
    </div>
  )
}
