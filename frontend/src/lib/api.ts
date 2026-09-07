/**
 * The one place that talks to the backend.
 *
 * Two things this deliberately does not do: throw raw `Response` objects,
 * and swallow errors. Every failure becomes an `ApiError` carrying a
 * message a user can read, because the UI has to distinguish "the backend
 * is not running" from "that DOI does not exist" and cannot do that from a
 * rejected promise with no shape.
 */

import type {
  ExportFormat,
  HealthResponse,
  ParsedQuery,
  SearchResponse,
  SourceName,
} from '@/types/domain'

/** Requests go same-origin in dev; Vite proxies them to the backend. */
const BASE = import.meta.env['VITE_API_BASE'] ?? ''

export class ApiError extends Error {
  readonly status: number
  /** True when the backend could not be reached at all. */
  readonly offline: boolean

  constructor(message: string, status: number, offline = false) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.offline = offline
  }
}

export interface SearchParams {
  query: string
  limitPerSource?: number
  sources?: SourceName[]
  signal?: AbortSignal
}

/**
 * Run a search.
 *
 * POST rather than GET: a pasted abstract is a legitimate query and can
 * run to several thousand characters, which does not belong in a URL.
 */
export async function search({
  query,
  limitPerSource = 20,
  sources,
  signal,
}: SearchParams): Promise<SearchResponse> {
  return request<SearchResponse>('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      limit_per_source: limitPerSource,
      ...(sources ? { sources } : {}),
    }),
    ...(signal ? { signal } : {}),
  })
}

/**
 * Ask the backend how it would interpret a query.
 *
 * Used to show the detected input type as the user types. The
 * classification lives on the server so the hint can never disagree with
 * what the search will actually do — reimplementing DOI detection in the
 * browser would guarantee they drift.
 */
export async function parseQuery(query: string, signal?: AbortSignal): Promise<ParsedQuery> {
  const params = new URLSearchParams({ q: query })
  return request<ParsedQuery>(`/api/parse?${params}`, signal ? { signal } : {})
}

export async function health(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', signal ? { signal } : {})
}

export interface ExportFormatInfo {
  id: ExportFormat
  label: string
  extension: string
}

export async function exportFormats(signal?: AbortSignal): Promise<ExportFormatInfo[]> {
  return request<ExportFormatInfo[]>(
    '/api/export/formats/available',
    signal ? { signal } : {},
  )
}

export interface ExportResult {
  blob: Blob
  filename: string
  /** Ids that were requested but no longer held by the backend. */
  missing: number
}

/**
 * Export a selection, returning the file rather than triggering a download.
 *
 * Keeping the download side effect out of here means the caller decides
 * when a file lands in the user's Downloads folder, and the function stays
 * testable.
 */
export async function exportPapers(
  paperIds: string[],
  format: ExportFormat,
  signal?: AbortSignal,
): Promise<ExportResult> {
  const response = await fetch(`${BASE}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_ids: paperIds, format }),
    ...(signal ? { signal } : {}),
  }).catch((error: unknown) => {
    throw toOfflineError(error)
  })

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status)
  }

  return {
    blob: await response.blob(),
    filename: filenameFrom(response) ?? `paperpilot-references.${format}`,
    missing: Number(response.headers.get('X-PaperPilot-Missing') ?? '0'),
  }
}

/** Hand a blob to the browser as a download. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  // Revoking immediately can cancel the download in some browsers; one
  // frame is enough for the click to have been handled.
  requestAnimationFrame(() => URL.revokeObjectURL(url))
}

// --- internals ------------------------------------------------------------

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init).catch((error: unknown) => {
    throw toOfflineError(error)
  })

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status)
  }
  return (await response.json()) as T
}

function toOfflineError(error: unknown): ApiError {
  // An aborted request is the caller superseding itself, not a failure;
  // it is rethrown unchanged so callers can ignore it by name.
  if (error instanceof DOMException && error.name === 'AbortError') {
    throw error
  }
  return new ApiError(
    'Could not reach the PaperPilot API. Is the backend running on port 8000?',
    0,
    true,
  )
}

/**
 * Pull a useful message out of an error response.
 *
 * FastAPI reports validation failures as a list of per-field objects and
 * everything else as a plain string, so both shapes are handled rather
 * than rendering "[object Object]" at the user.
 */
async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    const detail = (body as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: unknown } | undefined
      if (first && typeof first.msg === 'string') return first.msg
    }
  } catch {
    // Not JSON — fall through to the status line.
  }
  return `Request failed (${response.status})`
}

function filenameFrom(response: Response): string | null {
  const disposition = response.headers.get('Content-Disposition')
  const match = disposition?.match(/filename="([^"]+)"/)
  return match?.[1] ?? null
}
