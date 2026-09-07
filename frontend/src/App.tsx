import { useCallback, useEffect, useMemo, useState } from 'react'

import { ClusterTabs } from '@/components/ClusterTabs'
import { ExportBar } from '@/components/ExportBar'
import { HistorySidebar } from '@/components/HistorySidebar'
import { PaperCard } from '@/components/PaperCard'
import { PaperDetail } from '@/components/PaperDetail'
import { PipelineStatus } from '@/components/PipelineStatus'
import { SearchBar } from '@/components/SearchBar'
import { IconButton, cx } from '@/components/primitives'
import {
  EmptyResults,
  ErrorState,
  ResultSkeleton,
  SearchingBanner,
  WelcomeState,
} from '@/components/states'
import { useSearch } from '@/hooks/useSearch'
import { useSearchHistory } from '@/hooks/useSearchHistory'
import { useTheme } from '@/hooks/useTheme'
import { exportFormats } from '@/lib/api'
import type { ExportFormat, Paper, SearchResponse } from '@/types/domain'

/** Fallback if the backend cannot be asked which formats it supports. */
const DEFAULT_FORMATS: ExportFormat[] = ['bibtex', 'ris', 'apa', 'vancouver']

export default function App() {
  const { theme, toggle } = useTheme()
  const { history, record, remove, clear } = useSearchHistory()

  const [queryInput, setQueryInput] = useState('')
  const [activeCluster, setActiveCluster] = useState<number | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [selectionMode, setSelectionMode] = useState(false)
  const [detail, setDetail] = useState<Paper | null>(null)
  const [formats, setFormats] = useState<ExportFormat[]>(DEFAULT_FORMATS)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const onSuccess = useCallback(
    (response: SearchResponse) => {
      record({
        query: response.query.raw,
        intent: response.query.intent,
        resultCount: response.total,
        ranAt: Date.now(),
      })
      // A new result set invalidates any filter or selection from the old
      // one: the cluster ids mean something different now, and a selection
      // the user can no longer see is a trap.
      setActiveCluster(null)
      setSelected(new Set())
    },
    [record],
  )

  const { state, run, reset } = useSearch(onSuccess)

  // The format picker is driven by the backend so it cannot drift from
  // what the server can actually produce.
  useEffect(() => {
    const controller = new AbortController()
    exportFormats(controller.signal)
      .then((available) => setFormats(available.map((entry) => entry.id)))
      .catch(() => {
        /* the default list is a fine answer */
      })
    return () => controller.abort()
  }, [])

  const handleSearch = useCallback(
    (query: string) => {
      setQueryInput(query)
      setSidebarOpen(false)
      void run(query)
    },
    [run],
  )

  const response = state.status === 'success' ? state.response : null

  const visiblePapers = useMemo(() => {
    if (!response) return []
    if (activeCluster === null) return response.papers
    const cluster = response.clusters.find((item) => item.id === activeCluster)
    if (!cluster) return response.papers
    const ids = new Set(cluster.paper_ids)
    return response.papers.filter((paper) => ids.has(paper.id))
  }, [response, activeCluster])

  const toggleSelect = useCallback((id: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const selectAllVisible = useCallback(() => {
    setSelected((current) => {
      const next = new Set(current)
      const allSelected = visiblePapers.every((paper) => next.has(paper.id))
      for (const paper of visiblePapers) {
        if (allSelected) next.delete(paper.id)
        else next.add(paper.id)
      }
      return next
    })
  }, [visiblePapers])

  const allVisibleSelected =
    visiblePapers.length > 0 && visiblePapers.every((paper) => selected.has(paper.id))

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-slate-50/85 backdrop-blur dark:border-slate-800 dark:bg-slate-950/85">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <button
            type="button"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label="Toggle search history"
            className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-100 lg:hidden dark:text-slate-400 dark:hover:bg-slate-800"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path d="M2 4.75A.75.75 0 0 1 2.75 4h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 4.75Zm0 5A.75.75 0 0 1 2.75 9h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 9.75Zm0 5a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75Z" />
            </svg>
          </button>

          <a href="/" className="flex shrink-0 items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-600 text-sm font-bold text-white">
              PP
            </span>
            <span className="hidden text-[15px] font-semibold tracking-tight sm:inline">
              PaperPilot
            </span>
          </a>

          <div className="ml-auto flex items-center gap-1">
            {response && response.papers.length > 0 ? (
              <button
                type="button"
                onClick={() => {
                  setSelectionMode((on) => {
                    if (on) setSelected(new Set())
                    return !on
                  })
                }}
                className={cx(
                  'rounded-lg px-3 py-1.5 text-sm font-medium transition',
                  selectionMode
                    ? 'bg-accent-600 text-white'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800',
                )}
              >
                {selectionMode ? 'Done' : 'Select'}
              </button>
            ) : null}

            <IconButton onClick={toggle} label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
              {theme === 'dark' ? (
                <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path d="M10 2a.75.75 0 0 1 .75.75v1a.75.75 0 0 1-1.5 0v-1A.75.75 0 0 1 10 2ZM10 15a.75.75 0 0 1 .75.75v1a.75.75 0 0 1-1.5 0v-1A.75.75 0 0 1 10 15ZM10 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM15.657 4.343a.75.75 0 0 1 0 1.061l-.707.707a.75.75 0 1 1-1.06-1.06l.706-.708a.75.75 0 0 1 1.06 0ZM6.11 13.89a.75.75 0 0 1 0 1.06l-.707.708a.75.75 0 0 1-1.06-1.06l.706-.708a.75.75 0 0 1 1.061 0ZM18 10a.75.75 0 0 1-.75.75h-1a.75.75 0 0 1 0-1.5h1A.75.75 0 0 1 18 10ZM4.5 10a.75.75 0 0 1-.75.75h-1a.75.75 0 0 1 0-1.5h1A.75.75 0 0 1 4.5 10ZM15.657 15.657a.75.75 0 0 1-1.06 0l-.708-.707a.75.75 0 0 1 1.061-1.06l.707.706a.75.75 0 0 1 0 1.061ZM6.11 6.11a.75.75 0 0 1-1.061 0l-.707-.707a.75.75 0 0 1 1.06-1.06l.708.706a.75.75 0 0 1 0 1.061Z" />
                </svg>
              ) : (
                <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path
                    fillRule="evenodd"
                    d="M7.455 2.004a.75.75 0 0 1 .26.77 7 7 0 0 0 9.958 7.967.75.75 0 0 1 1.067.853A8.5 8.5 0 1 1 6.647 1.921a.75.75 0 0 1 .808.083Z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </IconButton>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-7xl gap-6 px-4 py-6">
        {/* History is a persistent rail on wide screens and a drawer on
            narrow ones, rather than being dropped on mobile: a daily user
            reaching for yesterday's query is doing that on any device. */}
        <div
          className={cx(
            'fixed inset-0 z-40 bg-slate-900/40 lg:hidden',
            sidebarOpen ? 'block' : 'hidden',
          )}
          onClick={() => setSidebarOpen(false)}
          role="presentation"
        />
        <div
          className={cx(
            'z-40 w-64 shrink-0',
            'max-lg:fixed max-lg:inset-y-0 max-lg:left-0 max-lg:overflow-y-auto max-lg:border-r max-lg:bg-slate-50 max-lg:p-4 max-lg:transition-transform dark:max-lg:border-slate-800 dark:max-lg:bg-slate-950',
            sidebarOpen ? 'max-lg:translate-x-0' : 'max-lg:-translate-x-full',
          )}
        >
          <div className="lg:sticky lg:top-[4.5rem] lg:max-h-[calc(100vh-6rem)]">
            <HistorySidebar
              history={history}
              activeQuery={response?.query.raw ?? null}
              onSelect={handleSearch}
              onRemove={remove}
              onClear={clear}
            />
          </div>
        </div>

        <main className="min-w-0 flex-1 space-y-4">
          <SearchBar
            onSearch={handleSearch}
            busy={state.status === 'searching'}
            initialValue={queryInput}
          />

          {state.status === 'idle' ? <WelcomeState /> : null}

          {state.status === 'searching' ? (
            <>
              <SearchingBanner query={state.query} />
              <ResultSkeleton />
            </>
          ) : null}

          {state.status === 'error' ? (
            <ErrorState
              error={state.error}
              onRetry={() => {
                if (state.status === 'error') void run(state.query)
              }}
            />
          ) : null}

          {response ? (
            <>
              <PipelineStatus response={response} />

              {response.papers.length === 0 ? (
                <EmptyResults
                  query={response.query.raw}
                  isIdentifier={response.query.intent === 'identifier'}
                />
              ) : (
                <>
                  <ClusterTabs
                    clusters={response.clusters}
                    total={response.papers.length}
                    active={activeCluster}
                    onSelect={setActiveCluster}
                  />

                  {selectionMode ? (
                    <button
                      type="button"
                      onClick={selectAllVisible}
                      className="text-sm font-medium text-accent-600 hover:underline dark:text-accent-400"
                    >
                      {allVisibleSelected ? 'Deselect' : 'Select'} all {visiblePapers.length} shown
                    </button>
                  ) : null}

                  <div className="space-y-3">
                    {visiblePapers.map((paper) => (
                      <PaperCard
                        key={paper.id}
                        paper={paper}
                        selected={selected.has(paper.id)}
                        selectionMode={selectionMode}
                        onToggleSelect={toggleSelect}
                        onOpen={setDetail}
                      />
                    ))}
                  </div>
                </>
              )}
            </>
          ) : null}
        </main>
      </div>

      {detail ? (
        <PaperDetail paper={detail} formats={formats} onClose={() => setDetail(null)} />
      ) : null}

      <ExportBar
        selectedIds={[...selected]}
        formats={formats}
        onClear={() => {
          setSelected(new Set())
          setSelectionMode(false)
        }}
      />

      {state.status !== 'idle' ? (
        <button
          type="button"
          onClick={() => {
            reset()
            setQueryInput('')
          }}
          className="sr-only focus:not-sr-only focus:fixed focus:bottom-4 focus:left-4 focus:z-50 focus:rounded-lg focus:bg-slate-900 focus:px-3 focus:py-2 focus:text-sm focus:text-white"
        >
          Clear search
        </button>
      ) : null}
    </div>
  )
}
