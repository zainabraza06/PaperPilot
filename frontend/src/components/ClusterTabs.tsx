import type { ReactNode } from 'react'

import type { TopicCluster } from '@/types/domain'

import { cx } from './primitives'

/**
 * Sub-topic filters.
 *
 * This solves the problem the brief opens with: fifty results in one flat
 * list. The backend refuses to split a set that has no real sub-topics
 * rather than inventing tabs, so when this renders at all the groups are
 * meaningful — and when it does not render, that absence is itself
 * accurate rather than a missing feature.
 *
 * Rendered as a scrolling filter rail rather than filled pills. Cluster
 * labels are generated from the papers, so they can be three words or
 * eight; a row of solid buttons at varying widths reads as clutter, while
 * a quiet rail with one selected item reads as a filter. The rail scrolls
 * horizontally on narrow screens instead of wrapping to three lines.
 */
export function ClusterTabs({
  clusters,
  total,
  active,
  onSelect,
}: {
  clusters: TopicCluster[]
  total: number
  active: number | null
  onSelect: (clusterId: number | null) => void
}) {
  if (clusters.length === 0) return null

  return (
    <div
      className="rail -mx-1 px-1 pb-1"
      role="tablist"
      aria-label="Filter by sub-topic"
    >
      <Tab active={active === null} onClick={() => onSelect(null)} count={total}>
        All results
      </Tab>

      <span aria-hidden="true" className="mx-1 h-4 w-px shrink-0 bg-line" />

      {clusters.map((cluster) => (
        <Tab
          key={cluster.id}
          active={active === cluster.id}
          onClick={() => onSelect(cluster.id)}
          count={cluster.paper_ids.length}
          title={cluster.terms.join(', ')}
        >
          {cluster.label}
        </Tab>
      ))}
    </div>
  )
}

function Tab({
  children,
  count,
  active,
  onClick,
  title,
}: {
  children: ReactNode
  count: number
  active: boolean
  onClick: () => void
  title?: string
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      title={title}
      className={cx(
        'inline-flex shrink-0 snap-start items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium transition duration-150',
        active
          ? 'bg-strong text-canvas'
          : 'text-muted hover:bg-sunken hover:text-strong',
      )}
    >
      <span className="max-w-[16rem] truncate">{children}</span>
      <span
        className={cx(
          'tabular-nums text-2xs',
          active ? 'text-canvas/60' : 'text-faint',
        )}
      >
        {count}
      </span>
    </button>
  )
}
