import { useEffect, useRef, useState } from 'react'

import { downloadBlob, exportPapers } from '@/lib/api'
import type { ExportFormat } from '@/types/domain'
import { EXPORT_FORMAT_LABELS } from '@/types/domain'

import { Spinner, cx } from './primitives'

/**
 * The bulk export action, docked to the bottom of the viewport while a
 * selection exists.
 *
 * The brief asks for no confirmation friction, so there is none: pick a
 * format and the file downloads. The only interstitial state is the
 * in-flight spinner, and the only message afterwards is when something is
 * worth saying — a partial export, or a failure.
 *
 * It appears only when papers are selected, so it costs nothing in the
 * common case of someone just reading results.
 */
export function ExportBar({
  selectedIds,
  formats,
  onClear,
}: {
  selectedIds: string[]
  formats: ExportFormat[]
  onClear: () => void
}) {
  const [busy, setBusy] = useState<ExportFormat | null>(null)
  const [notice, setNotice] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current)
    }
  }, [])

  function flash(tone: 'ok' | 'error', text: string) {
    setNotice({ tone, text })
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setNotice(null), 5000)
  }

  async function handleExport(format: ExportFormat) {
    setBusy(format)
    setNotice(null)
    try {
      const result = await exportPapers(selectedIds, format)
      downloadBlob(result.blob, result.filename)
      if (result.missing > 0) {
        // A partial export is a real outcome worth naming rather than
        // letting the user discover a short file later.
        flash(
          'ok',
          `Exported ${selectedIds.length - result.missing} of ${selectedIds.length} — ${result.missing} were no longer available.`,
        )
      }
    } catch (error: unknown) {
      flash('error', error instanceof Error ? error.message : 'Export failed.')
    } finally {
      setBusy(null)
    }
  }

  if (selectedIds.length === 0) return null

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center p-4">
      <div className="pointer-events-auto animate-fade-in w-full max-w-3xl">
        {notice ? (
          <div
            role="status"
            className={cx(
              'mb-2 rounded-lg px-3 py-2 text-sm shadow-lg',
              notice.tone === 'ok'
                ? 'bg-strong text-canvas'
                : 'bg-critical text-canvas',
            )}
          >
            {notice.text}
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 rounded-xl border border-line bg-surface/95 p-2.5 shadow-float backdrop-blur-xl">
          <span className="px-1.5 text-sm font-semibold text-strong">
            {selectedIds.length} selected
          </span>

          <span className="hidden text-sm text-faint sm:inline">Export as</span>

          <div className="flex flex-wrap gap-1.5">
            {formats.map((format) => (
              <button
                key={format}
                type="button"
                onClick={() => void handleExport(format)}
                disabled={busy !== null}
                // Outline rather than four solid accent buttons in a row:
                // they are four peers, and filling all of them makes the
                // bar shout without telling anyone which to pick.
                className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-body ring-1 ring-inset ring-control transition hover:bg-accent-50 hover:text-accent-700 hover:ring-accent-400 disabled:opacity-50"
              >
                {busy === format ? <Spinner className="h-3.5 w-3.5" /> : null}
                {EXPORT_FORMAT_LABELS[format]}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={onClear}
            className="ml-auto rounded-lg px-2.5 py-1.5 text-sm text-muted transition hover:bg-sunken hover:text-strong"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  )
}
