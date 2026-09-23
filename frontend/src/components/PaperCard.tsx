import type { Paper } from '@/types/domain'

import { ScoreBar, SourceBadges, formatAuthors, formatDate } from './primitives'
import { SummaryBlock } from './SummaryBlock'

/**
 * One result row.
 *
 * Ordered by what a researcher scans for: rank, title, then who and where,
 * then the summary, then the machinery. The relevance meter sits on the
 * metadata line so the whole column can be read vertically without the eye
 * crossing the card.
 *
 * **On the click target.** An earlier version made the whole card
 * `role="button"` with `tabIndex={0}`, wrapping a real link and a real
 * checkbox. That is invalid — interactive elements must not nest — and it
 * flattens the card for assistive technology: the heading, the link to the
 * publisher and the select control all disappear into one announcement of
 * "button, open details for …".
 *
 * The accessible version of "click anywhere" is a stretched link: the
 * title anchor is the control, and a transparent pseudo-element extends
 * its hit area over the card. Everything genuinely interactive sits above
 * that overlay. The card keeps its `article` semantics, keyboard users get
 * one tab stop per meaningful control instead of two overlapping ones, and
 * the pointer affordance is unchanged.
 */
export function PaperCard({
  paper,
  rank,
  selected,
  selectionMode,
  onToggleSelect,
  onOpen,
}: {
  paper: Paper
  rank: number
  selected: boolean
  selectionMode: boolean
  onToggleSelect: (id: string) => void
  onOpen: (paper: Paper) => void
}) {
  const authorNames = paper.authors.map((author) => author.name)
  const sources =
    paper.also_found_in.length > 0 ? [paper.source, ...paper.also_found_in] : [paper.source]

  return (
    <article
      data-selected={selected}
      className="row group px-4 py-4 sm:px-6 sm:py-5"
    >
      <div className="flex gap-3 sm:gap-4">
        {selectionMode ? (
          <input
            type="checkbox"
            checked={selected}
            onChange={() => onToggleSelect(paper.id)}
            aria-label={`Select ${paper.title}`}
            className="card-raise mt-1 h-4 w-4 shrink-0 cursor-pointer rounded border-control bg-surface text-accent-600 focus:ring-accent-500"
          />
        ) : (
          <span
            aria-hidden="true"
            className="mt-1 hidden w-7 shrink-0 text-right text-sm font-medium tabular-nums text-faint sm:block"
          >
            {rank}
          </span>
        )}

        <div className="min-w-0 flex-1">
          <h2 className="text-xl font-semibold text-strong">
            {/* The stretched control. It opens the detail view rather than
                navigating away, because the detail view is where the
                entities, the full summary and the citation live — the
                publisher link is still one click away inside it. */}
            <button
              type="button"
              onClick={() => onOpen(paper)}
              // Deliberately *not* .card-raise: the overlay is positioned
              // against the nearest positioned ancestor, so making this
              // button relative would shrink the hit area back to the text.
              className="stretch-target text-left decoration-accent-400 decoration-2 underline-offset-[3px] group-hover:underline"
            >
              {paper.title}
            </button>
          </h2>

          <p className="mt-1.5 truncate text-sm text-muted">{formatAuthors(authorNames)}</p>

          <div className="meta-row mt-2 flex flex-wrap items-center text-xs text-muted">
            <SourceBadges sources={sources} />
            <span>{formatDate(paper.published_date)}</span>
            {paper.journal ? (
              <span className="max-w-[18rem] truncate italic">{paper.journal}</span>
            ) : null}
            {paper.citation_count !== null && paper.citation_count !== undefined ? (
              <span className="tabular-nums">
                {paper.citation_count.toLocaleString()} citations
              </span>
            ) : null}
            {paper.score ? (
              // Pushed to the end of the line rather than into the middot
              // run, so the meters line up down the list.
              <span className="no-sep ms-auto ps-3">
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
            <p className="mt-3 line-clamp-2 max-w-[70ch] text-base text-muted">{paper.abstract}</p>
          ) : (
            <p className="mt-3 max-w-[70ch] text-base italic text-faint">
              No abstract deposited — this record can still be ranked and cited, but not
              summarized or verified.
            </p>
          )}

          {paper.entities.length > 0 ? (
            <ul className="mt-3 flex flex-wrap items-center gap-1.5">
              {paper.entities.slice(0, 5).map((entity) => (
                <li
                  key={`${entity.text}-${entity.label}`}
                  title={entity.label.replace(/_/g, ' ')}
                  className="rounded border border-line bg-sunken px-1.5 py-px font-mono text-2xs text-muted"
                >
                  {entity.text}
                </li>
              ))}
              {paper.entities.length > 5 ? (
                <li className="text-2xs text-faint">+{paper.entities.length - 5} more</li>
              ) : null}
            </ul>
          ) : null}
        </div>
      </div>
    </article>
  )
}
