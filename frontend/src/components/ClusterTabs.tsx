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
      className="flex flex-wrap items-center gap-1.5"
      role="tablist"
      aria-label="Filter by sub-topic"
    >
      <Tab active={active === null} onClick={() => onSelect(null)} count={total}>
        All results
      </Tab>
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
        'inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium transition',
        active
          ? 'bg-accent-600 text-white shadow-sm'
          : 'bg-white text-slate-600 ring-1 ring-inset ring-slate-200 hover:bg-slate-50 hover:text-slate-900 dark:bg-slate-900 dark:text-slate-300 dark:ring-slate-700 dark:hover:bg-slate-800',
      )}
    >
      <span className="max-w-[18rem] truncate">{children}</span>
      <span
        className={cx(
          'rounded px-1 text-xs tabular-nums',
          active ? 'bg-white/20' : 'bg-slate-100 dark:bg-slate-800',
        )}
      >
        {count}
      </span>
    </button>
  )
}
