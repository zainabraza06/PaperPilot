import { useCallback, useState } from 'react'

import type { HistoryEntry } from '@/types/domain'

const STORAGE_KEY = 'paperpilot:history'
const MAX_ENTRIES = 25

function load(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    // Stored data is from a previous version of this app as far as we
    // know, so every entry is validated rather than trusted.
    return parsed.filter(isHistoryEntry).slice(0, MAX_ENTRIES)
  } catch {
    return []
  }
}

function isHistoryEntry(value: unknown): value is HistoryEntry {
  const entry = value as Partial<HistoryEntry>
  return typeof entry?.query === 'string' && typeof entry?.ranAt === 'number'
}

/**
 * Recent queries, kept in localStorage.
 *
 * The brief frames this tool as something a researcher uses daily, and the
 * thing a daily user wants most is the query they ran yesterday. Repeats
 * move to the top rather than stacking, so the list stays a set of
 * distinct searches instead of a keystroke log.
 */
export function useSearchHistory(): {
  history: HistoryEntry[]
  record: (entry: HistoryEntry) => void
  remove: (query: string) => void
  clear: () => void
} {
  const [history, setHistory] = useState<HistoryEntry[]>(load)

  const persist = useCallback((next: HistoryEntry[]) => {
    setHistory(next)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      // A full or unavailable store costs the user their history, not
      // their search.
    }
  }, [])

  const record = useCallback(
    (entry: HistoryEntry) => {
      setHistory((current) => {
        const deduped = current.filter((item) => item.query !== entry.query)
        const next = [entry, ...deduped].slice(0, MAX_ENTRIES)
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
        } catch {
          /* ignored */
        }
        return next
      })
    },
    [],
  )

  const remove = useCallback(
    (query: string) => {
      persist(history.filter((item) => item.query !== query))
    },
    [history, persist],
  )

  const clear = useCallback(() => persist([]), [persist])

  return { history, record, remove, clear }
}
