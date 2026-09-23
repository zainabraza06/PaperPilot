import { useEffect, useMemo, useRef, useState } from 'react'

import { exportPapers, downloadBlob } from '@/lib/api'
import type { Entity, EntityLabel, ExportFormat, Paper } from '@/types/domain'
import { ENTITY_LABEL_COPY, EXPORT_FORMAT_LABELS } from '@/types/domain'

import { SourceBadges, Spinner, cx, formatDate } from './primitives'
import { SummaryBlock } from './SummaryBlock'

/** Colour per entity category, so a highlighted abstract is readable. */
const ENTITY_CLASSES: Record<EntityLabel, string> = {
  gene_or_protein: 'bg-emerald-100 text-emerald-900 dark:bg-emerald-500/25 dark:text-emerald-100',
  disease: 'bg-rose-100 text-rose-900 dark:bg-rose-500/25 dark:text-rose-100',
  chemical: 'bg-amber-100 text-amber-900 dark:bg-amber-500/25 dark:text-amber-100',
  organism: 'bg-lime-100 text-lime-900 dark:bg-lime-500/25 dark:text-lime-100',
  anatomy: 'bg-pink-100 text-pink-900 dark:bg-pink-500/25 dark:text-pink-100',
  cell_type: 'bg-teal-100 text-teal-900 dark:bg-teal-500/25 dark:text-teal-100',
  technical_term: 'bg-indigo-100 text-indigo-900 dark:bg-indigo-500/25 dark:text-indigo-100',
  organization: 'bg-sky-100 text-sky-900 dark:bg-sky-500/25 dark:text-sky-100',
  person: 'bg-violet-100 text-violet-900 dark:bg-violet-500/25 dark:text-violet-100',
  location: 'bg-orange-100 text-orange-900 dark:bg-orange-500/25 dark:text-orange-100',
  other: 'bg-line-strong text-body',
}

interface Segment {
  text: string
  entity: Entity | null
}

/**
 * Split text into plain and highlighted runs using the entity spans.
 *
 * The backend guarantees non-overlapping spans — its pattern pass only
 * fills gaps the model pass did not claim — which is exactly what makes a
 * single left-to-right walk correct here. Spans are still sorted and
 * bounds-checked rather than trusted, because a rendering bug from bad
 * offsets would corrupt the text a researcher is reading.
 */
function segment(text: string, entities: Entity[], field: 'title' | 'abstract'): Segment[] {
  const spans = entities
    .flatMap((entity) =>
      entity.spans
        .filter((span) => span.field === field)
        .map((span) => ({ start: span.start, end: span.end, entity })),
    )
    .filter((span) => span.start >= 0 && span.end <= text.length && span.end > span.start)
    .sort((a, b) => a.start - b.start)

  const segments: Segment[] = []
  let cursor = 0
  for (const span of spans) {
    if (span.start < cursor) continue // defensive: skip any overlap
    if (span.start > cursor) {
      segments.push({ text: text.slice(cursor, span.start), entity: null })
    }
    segments.push({ text: text.slice(span.start, span.end), entity: span.entity })
    cursor = span.end
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), entity: null })
  }
  return segments
}

function Highlighted({
  text,
  entities,
  field,
  enabled,
}: {
  text: string
  entities: Entity[]
  field: 'title' | 'abstract'
  enabled: boolean
}) {
  const segments = useMemo(
    () => (enabled ? segment(text, entities, field) : [{ text, entity: null }]),
    [text, entities, field, enabled],
  )

  return (
    <>
      {segments.map((part, index) =>
        part.entity ? (
          <mark
            key={index}
            title={ENTITY_LABEL_COPY[part.entity.label]}
            className={cx('rounded px-0.5', ENTITY_CLASSES[part.entity.label])}
          >
            {part.text}
          </mark>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  )
}

export function PaperDetail({
  paper,
  onClose,
  formats,
}: {
  paper: Paper
  onClose: () => void
  formats: ExportFormat[]
}) {
  const [highlight, setHighlight] = useState(true)
  const [exporting, setExporting] = useState<ExportFormat | null>(null)
  const [error, setError] = useState<string | null>(null)
  const closeRef = useRef<HTMLButtonElement>(null)

  // Escape closes, and focus starts inside the dialog rather than
  // wherever it happened to be behind it.
  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [onClose])

  const labelsPresent = useMemo(
    () => [...new Set(paper.entities.map((entity) => entity.label))],
    [paper.entities],
  )

  async function handleExport(format: ExportFormat) {
    setExporting(format)
    setError(null)
    try {
      const result = await exportPapers([paper.id], format)
      downloadBlob(result.blob, result.filename)
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : 'Export failed.')
    } finally {
      setExporting(null)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-strong/50 p-4 backdrop-blur-sm sm:p-8"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={paper.title}
        onClick={(event) => event.stopPropagation()}
        className="surface-panel animate-scale-in my-auto w-full max-w-3xl shadow-dialog"
      >
        <header className="flex items-start gap-4 border-b border-line p-5">
          <div className="min-w-0 flex-1">
            <h2 className="text-lg font-semibold leading-snug text-strong">
              <Highlighted
                text={paper.title}
                entities={paper.entities}
                field="title"
                enabled={highlight}
              />
            </h2>
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted">
              <SourceBadges
                sources={
                  paper.also_found_in.length > 0
                    ? [paper.source, ...paper.also_found_in]
                    : [paper.source]
                }
              />
              <span>{formatDate(paper.published_date)}</span>
              {paper.journal ? <span className="italic">{paper.journal}</span> : null}
              {paper.doi ? (
                <a
                  href={`https://doi.org/${paper.doi}`}
                  target="_blank"
                  rel="noreferrer"
                  className="font-mono text-accent-600 hover:underline"
                >
                  {paper.doi}
                </a>
              ) : null}
            </div>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-faint transition hover:bg-sunken hover:text-body"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
            </svg>
          </button>
        </header>

        <div className="max-h-[60vh] space-y-5 overflow-y-auto p-5">
          <section>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">
              Authors
            </p>
            <p className="text-sm text-body">
              {paper.authors.length > 0
                ? paper.authors.map((author) => author.name).join(', ')
                : 'No authors listed'}
            </p>
          </section>

          {/* Same reasoning as the card: with no abstract the summary is
              the title, which is already at the top of this dialog. */}
          {paper.summary && paper.summary.grounding.status !== 'unverifiable' ? (
            <section>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">
                Summary
              </p>
              <SummaryBlock summary={paper.summary} full />
            </section>
          ) : null}

          <section>
            <div className="mb-2 flex items-center justify-between gap-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-faint">
                Abstract
              </p>
              {paper.entities.length > 0 ? (
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted">
                  <input
                    type="checkbox"
                    checked={highlight}
                    onChange={(event) => setHighlight(event.target.checked)}
                    className="h-3.5 w-3.5 rounded border-control text-accent-600 focus:ring-accent-500"
                  />
                  Highlight entities
                </label>
              ) : null}
            </div>
            {paper.abstract ? (
              <p className="text-sm leading-relaxed text-body">
                <Highlighted
                  text={paper.abstract}
                  entities={paper.entities}
                  field="abstract"
                  enabled={highlight}
                />
              </p>
            ) : (
              <p className="text-sm italic text-faint">
                No abstract was deposited for this record. It can still be ranked and
                cited, but not summarized or verified.
              </p>
            )}
          </section>

          {labelsPresent.length > 0 && highlight ? (
            <section>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">
                Entity types
              </p>
              <div className="flex flex-wrap gap-1.5">
                {labelsPresent.map((label) => (
                  <span
                    key={label}
                    className={cx('rounded px-1.5 py-0.5 text-xs', ENTITY_CLASSES[label])}
                  >
                    {ENTITY_LABEL_COPY[label]}
                  </span>
                ))}
              </div>
            </section>
          ) : null}

          {paper.keywords.length > 0 ? (
            <section>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">
                Keywords
              </p>
              <div className="flex flex-wrap gap-1.5">
                {paper.keywords.map((keyword) => (
                  <span
                    key={keyword}
                    className="rounded bg-sunken px-1.5 py-0.5 text-xs text-muted"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            </section>
          ) : null}
        </div>

        <footer className="flex flex-wrap items-center gap-2 border-t border-line p-4">
          <span className="mr-1 text-xs font-medium text-muted">
            Cite as
          </span>
          {formats.map((format) => (
            <button
              key={format}
              type="button"
              onClick={() => void handleExport(format)}
              disabled={exporting !== null}
              className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium text-body transition hover:border-accent-400 hover:text-accent-700 disabled:opacity-50 dark:hover:border-accent-500 dark:hover:text-accent-300"
            >
              {exporting === format ? <Spinner className="h-3 w-3" /> : null}
              {EXPORT_FORMAT_LABELS[format]}
            </button>
          ))}
          {paper.pdf_url ? (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="ml-auto inline-flex items-center gap-1.5 rounded-lg bg-accent-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-accent-500"
            >
              Open PDF
            </a>
          ) : null}
          {error ? (
            <p className="w-full text-xs text-rose-600 dark:text-rose-400">{error}</p>
          ) : null}
        </footer>
      </div>
    </div>
  )
}
