import type { Summary, SummaryOrigin } from '@/types/domain'

import { cx } from './primitives'

/**
 * A summary, always shown with where it came from.
 *
 * The backend goes to real trouble to distinguish generated text that
 * passed a grounding check, text that needed a correcting retry, and
 * extractive sentences lifted verbatim from the abstract. Rendering all
 * three identically would throw that away — and would let a reader assume
 * an AI wrote something the abstract actually said, or the reverse.
 *
 * Extractive summaries are labelled "from abstract" rather than being
 * dressed up as AI output. They are the honest fallback, and saying so is
 * the whole point of having tracked the origin.
 *
 * Visually this used to be a grey box inside a white card. Nesting a
 * filled panel inside a panel is what made the list look like a form, so
 * the summary now hangs off a coloured rule instead: the same "this is a
 * distinct kind of content" signal, at a fraction of the visual weight,
 * and the rule doubles as the origin's colour channel.
 */
const ORIGIN_COPY: Record<
  SummaryOrigin,
  { label: string; title: string; rule: string; chip: string }
> = {
  generated: {
    label: 'AI',
    title: 'Generated and verified against the abstract',
    rule: 'bg-accent-300',
    chip: 'text-accent-600',
  },
  regenerated: {
    label: 'AI · corrected',
    title:
      'The first attempt failed the grounding check and was regenerated with the specific problems fed back',
    rule: 'bg-accent-200',
    chip: 'text-accent-600',
  },
  extractive: {
    label: 'From abstract',
    title:
      'Sentences taken verbatim from the abstract — either no model was configured, or generated text could not be grounded',
    rule: 'bg-line-strong',
    chip: 'text-muted',
  },
}

export function SummaryBlock({
  summary,
  className,
  full = false,
}: {
  summary: Summary
  className?: string
  full?: boolean
}) {
  const origin = ORIGIN_COPY[summary.origin]
  const unverifiable = summary.grounding.status === 'unverifiable'
  const advisories = summary.grounding.issues.filter(
    (issue) => issue.severity === 'advisory',
  )

  return (
    <div className={cx('flex gap-3', className)}>
      <span
        aria-hidden="true"
        className={cx('w-0.5 shrink-0 rounded-full', origin.rule)}
      />

      <div className="min-w-0 flex-1">
        <div className="mb-1 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-2xs font-medium">
          <span title={origin.title} className={origin.chip}>
            {origin.label}
          </span>

          {unverifiable ? (
            <span
              title="This paper has no abstract, so nothing could be checked against"
              className="inline-flex items-center gap-1 text-caution"
            >
              <WarningIcon />
              unverifiable
            </span>
          ) : (
            <span
              title={`${Math.round(summary.grounding.overlap * 100)}% of the summary's content words appear in the abstract`}
              className="inline-flex items-center gap-1 text-positive"
            >
              <CheckIcon />
              grounded
            </span>
          )}

          {/* An advisory never rejected the summary, so it must not read
              like an error — but it is the only visible trace of the
              semantic support check, and hiding it entirely would waste
              the one signal that catches a claim the abstract never made. */}
          {advisories.length > 0 ? (
            <span
              title={advisories.map((issue) => issue.detail).join('\n')}
              className="inline-flex items-center gap-1 text-caution"
            >
              <WarningIcon />
              {advisories.length === 1 ? '1 claim to check' : `${advisories.length} claims to check`}
            </span>
          ) : null}

          {summary.cached ? <span className="text-faint">cached</span> : null}
        </div>

        <p className={cx('max-w-[70ch] text-base text-body', !full && 'line-clamp-3')}>
          {summary.text}
        </p>

        {/* What it took to get here, shown only in the detail view: it is
            useful provenance but would be noise in a scannable list. */}
        {full && summary.rejected_for.length > 0 ? (
          <div className="mt-3 rounded-lg border border-line bg-sunken p-3">
            <p className="text-2xs font-semibold uppercase tracking-wider text-faint">
              First attempt rejected for
            </p>
            <ul className="mt-1.5 space-y-1">
              {summary.rejected_for.map((issue, index) => (
                <li
                  key={`${issue.kind}-${index}`}
                  className="flex gap-2 text-xs text-muted"
                >
                  <span aria-hidden="true" className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-critical" />
                  {issue.detail}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  )
}

function CheckIcon() {
  return (
    <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function WarningIcon() {
  return (
    <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495ZM10 5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
        clipRule="evenodd"
      />
    </svg>
  )
}
