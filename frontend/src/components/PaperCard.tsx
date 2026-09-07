import type { Paper } from '@/types/domain'

import { ScoreBar, SourceBadges, cx, formatAuthors, formatDate } from './primitives'
import { SummaryBlock } from './SummaryBlock'

/**
 * One result row.
 *
 * Ordered by what a researcher scans for: title, then who and where, then
 * the summary, then the machinery. The relevance bar sits at the far right
 * of the metadata line so the whole column can be read vertically without
 * the eye crossing the card.
 *
 * The whole card is clickable to open the detail view, but the title is
 * also a real link to the paper, and the checkbox and DOI link stop
 * propagation — so "select this" and "go to the publisher" never trigger
 * "open the modal".
 */
export function PaperCard({
  paper,
  selected,
  selectionMode,
  onToggleSelect,
  onOpen,
}: {
  paper: Paper
  selected: boolean
  selectionMode: boolean
  onToggleSelect: (id: string) => void
  onOpen: (paper: Paper) => void
}) {
  const authorNames = paper.authors.map((author) => author.name)

  return (
    <article
      onClick={() => onOpen(paper)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onOpen(paper)
        }
      }}
      tabIndex={0}
      role="button"
      aria-label={`Open details for ${paper.title}`}
      className={cx(
        'surface group animate-fade-in cursor-pointer p-4 transition',
        'hover:border-slate-300 hover:shadow-md dark:hover:border-slate-700',
        selected && 'border-accent-400 ring-1 ring-accent-400 dark:border-accent-500 dark:ring-accent-500',
      )}
    >
      <div className="flex gap-3">
        {selectionMode ? (
          <input
            type="checkbox"
            checked={selected}
            onClick={(event) => event.stopPropagation()}
            onChange={() => onToggleSelect(paper.id)}
            aria-label={`Select ${paper.title}`}
            className="mt-1 h-4 w-4 shrink-0 cursor-pointer rounded border-slate-300 text-accent-600 focus:ring-accent-500 dark:border-slate-600 dark:bg-slate-800"
          />
        ) : null}

        <div className="min-w-0 flex-1">
          <h2 className="text-[15px] font-semibold leading-snug text-slate-900 dark:text-slate-50">
            <a
              href={paper.url}
              target="_blank"
              rel="noreferrer"
              onClick={(event) => event.stopPropagation()}
              className="decoration-accent-400 underline-offset-2 hover:underline"
            >
              {paper.title}
            </a>
          </h2>

          <p className="mt-1 truncate text-sm text-slate-600 dark:text-slate-400">
            {formatAuthors(authorNames)}
          </p>

          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-slate-500 dark:text-slate-400">
            <SourceBadges sources={paper.also_found_in.length > 0
              ? [paper.source, ...paper.also_found_in]
              : [paper.source]} />
            <span>{formatDate(paper.published_date)}</span>
            {paper.journal ? (
              <span className="max-w-[16rem] truncate italic">{paper.journal}</span>
            ) : null}
            {paper.citation_count !== null && paper.citation_count !== undefined ? (
              <span>{paper.citation_count.toLocaleString()} citations</span>
            ) : null}
            {paper.score ? (
              <span className="ml-auto">
                <ScoreBar
                  value={paper.score.combined}
                  semantic={paper.score.semantic}
                  lexical={paper.score.lexical}
                />
              </span>
            ) : null}
          </div>

          {/* A paper with no abstract gets an extractive "summary" that is
              just its own title, which would render directly under the
              title itself. Saying the record has no abstract is both
              shorter and more informative than echoing it back. */}
          {paper.summary && paper.summary.grounding.status !== 'unverifiable' ? (
            <SummaryBlock summary={paper.summary} className="mt-3" />
          ) : paper.abstract ? (
            <p className="mt-3 line-clamp-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
              {paper.abstract}
            </p>
          ) : (
            <p className="mt-3 text-sm italic text-slate-400 dark:text-slate-500">
              No abstract deposited — this record can still be ranked and cited, but
              not summarized or verified.
            </p>
          )}

          {paper.entities.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-1">
              {paper.entities.slice(0, 6).map((entity) => (
                <span
                  key={`${entity.text}-${entity.label}`}
                  title={entity.label.replace(/_/g, ' ')}
                  className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                >
                  {entity.text}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </article>
  )
}
