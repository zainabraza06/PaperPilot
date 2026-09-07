import { useEffect, useRef, useState } from 'react'

import { useDebounced } from '@/hooks/useDebounced'
import { parseQuery } from '@/lib/api'
import type { ParsedQuery, QueryIntent } from '@/types/domain'

import { Spinner, cx } from './primitives'

/**
 * What each detected input type should tell the user.
 *
 * The point of showing this at all is that the search box accepts four
 * genuinely different things and a user has no way to know that. Naming
 * the detected type as they type both teaches the feature and confirms
 * the tool understood them — particularly for a pasted abstract, where
 * nothing else on screen would reveal that only keywords go upstream.
 */
const INTENT_COPY: Record<QueryIntent, { label: string; hint: string }> = {
  keyword: { label: 'Keyword', hint: 'searching all three sources' },
  topic: { label: 'Topic', hint: 'searching all three sources' },
  identifier: { label: 'Identifier', hint: 'looking this up directly' },
  abstract_snippet: {
    label: 'Abstract',
    hint: 'distilling keywords for the APIs, ranking on the full text',
  },
}

const EXAMPLES = [
  { label: 'a topic', value: 'CRISPR prime editing efficiency in human cells' },
  { label: 'a DOI', value: '10.1038/s41586-019-1711-4' },
  { label: 'an arXiv id', value: '1706.03762' },
]

export function SearchBar({
  onSearch,
  busy,
  initialValue = '',
}: {
  onSearch: (query: string) => void
  busy: boolean
  initialValue?: string
}) {
  const [value, setValue] = useState(initialValue)
  const [parsed, setParsed] = useState<ParsedQuery | null>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const debounced = useDebounced(value, 300)

  // Keep the box in step when a history entry is clicked.
  useEffect(() => setValue(initialValue), [initialValue])

  // Classify as the user types. Debounced and abortable: this is a hint,
  // so a stale one is worse than none, and it must never queue up.
  useEffect(() => {
    const trimmed = debounced.trim()
    if (trimmed.length < 3) {
      setParsed(null)
      return
    }
    const controller = new AbortController()
    parseQuery(trimmed, controller.signal)
      .then(setParsed)
      .catch(() => setParsed(null))
    return () => controller.abort()
  }, [debounced])

  // The textarea grows with a pasted abstract instead of scrolling a
  // single line, which is the difference between "this accepts a
  // paragraph" being obvious and being a surprise.
  useEffect(() => {
    const node = inputRef.current
    if (!node) return
    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, 200)}px`
  }, [value])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === '/' && document.activeElement !== inputRef.current) {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function submit() {
    const trimmed = value.trim()
    if (trimmed && !busy) onSearch(trimmed)
  }

  const intent = parsed ? INTENT_COPY[parsed.intent] : null

  return (
    <div className="w-full">
      <div
        className={cx(
          'surface flex items-start gap-3 p-3 shadow-sm transition',
          'focus-within:border-accent-400 focus-within:shadow-md dark:focus-within:border-accent-500',
        )}
      >
        <svg
          className="mt-2 h-5 w-5 shrink-0 text-slate-400"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
            clipRule="evenodd"
          />
        </svg>

        <textarea
          ref={inputRef}
          value={value}
          rows={1}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            // Enter searches; Shift+Enter is a newline, because a pasted
            // abstract legitimately contains them.
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder="Search a topic, keyword, DOI, arXiv id, PMID — or paste an abstract"
          aria-label="Search query"
          spellCheck={false}
          className="min-h-[28px] flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-6 outline-none placeholder:text-slate-400 dark:placeholder:text-slate-500"
        />

        <button
          type="button"
          onClick={submit}
          disabled={busy || !value.trim()}
          className="inline-flex h-9 shrink-0 items-center gap-2 rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white transition hover:bg-accent-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? <Spinner className="h-4 w-4" /> : null}
          {busy ? 'Searching' : 'Search'}
        </button>
      </div>

      <div className="mt-2 flex min-h-[24px] flex-wrap items-center gap-x-3 gap-y-1 px-1 text-xs text-slate-500 dark:text-slate-400">
        {intent && parsed ? (
          <span className="animate-fade-in inline-flex items-center gap-1.5">
            <span className="inline-flex items-center rounded bg-accent-50 px-1.5 py-0.5 font-semibold text-accent-700 ring-1 ring-inset ring-accent-600/20 dark:bg-accent-500/10 dark:text-accent-300 dark:ring-accent-400/20">
              {intent.label}
            </span>
            <span>{intent.hint}</span>
            {parsed.intent === 'abstract_snippet' && parsed.keywords.length > 0 ? (
              <span className="font-mono text-[11px] text-slate-400 dark:text-slate-500">
                → {parsed.keywords.slice(0, 6).join(' ')}
              </span>
            ) : null}
          </span>
        ) : (
          <>
            <span>Try</span>
            {EXAMPLES.map((example) => (
              <button
                key={example.value}
                type="button"
                onClick={() => {
                  setValue(example.value)
                  onSearch(example.value)
                }}
                className="rounded px-1 font-medium text-accent-600 underline-offset-2 transition hover:underline dark:text-accent-400"
              >
                {example.label}
              </button>
            ))}
            <span className="ml-auto hidden sm:inline">
              Press <kbd className="rounded border border-slate-300 px-1 font-sans dark:border-slate-600">/</kbd> to focus
            </span>
          </>
        )}
      </div>
    </div>
  )
}
