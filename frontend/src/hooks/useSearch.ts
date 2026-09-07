import { useCallback, useRef, useState } from 'react'

import { ApiError, search } from '@/lib/api'
import type { SearchResponse } from '@/types/domain'

/**
 * The search lifecycle, as a state machine rather than a pile of booleans.
 *
 * `idle | searching | success | error` makes the impossible states
 * impossible: there is no way to render a spinner and an error together,
 * or to show stale results while a new search is in flight without
 * deciding to.
 */
export type SearchState =
  | { status: 'idle' }
  | { status: 'searching'; query: string; previous: SearchResponse | null }
  | { status: 'success'; response: SearchResponse }
  | { status: 'error'; error: ApiError; query: string }

export interface UseSearch {
  state: SearchState
  run: (query: string) => Promise<SearchResponse | null>
  reset: () => void
}

export function useSearch(onSuccess?: (response: SearchResponse) => void): UseSearch {
  const [state, setState] = useState<SearchState>({ status: 'idle' })
  const inFlight = useRef<AbortController | null>(null)
  // Kept in a ref so `run` does not change identity when results arrive,
  // which would re-create every callback that depends on it.
  const latest = useRef<SearchResponse | null>(null)

  const run = useCallback(
    async (query: string): Promise<SearchResponse | null> => {
      const trimmed = query.trim()
      if (!trimmed) return null

      // A new search supersedes the old one. Without this, a slow first
      // query can resolve after a fast second and overwrite it.
      inFlight.current?.abort()
      const controller = new AbortController()
      inFlight.current = controller

      setState({ status: 'searching', query: trimmed, previous: latest.current })

      try {
        const response = await search({ query: trimmed, signal: controller.signal })
        if (controller.signal.aborted) return null
        latest.current = response
        setState({ status: 'success', response })
        onSuccess?.(response)
        return response
      } catch (error: unknown) {
        // An abort means the user searched again; the newer request owns
        // the UI now and this one should say nothing.
        if (error instanceof DOMException && error.name === 'AbortError') return null
        setState({
          status: 'error',
          error:
            error instanceof ApiError
              ? error
              : new ApiError('Something went wrong running that search.', 0),
          query: trimmed,
        })
        return null
      } finally {
        if (inFlight.current === controller) inFlight.current = null
      }
    },
    [onSuccess],
  )

  const reset = useCallback(() => {
    inFlight.current?.abort()
    inFlight.current = null
    latest.current = null
    setState({ status: 'idle' })
  }, [])

  return { state, run, reset }
}
