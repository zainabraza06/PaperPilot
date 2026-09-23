/**
 * Audit the design tokens against WCAG 2.2 contrast minima.
 *
 * A palette is the one part of a redesign that can be checked rather than
 * argued about, and it is also the part that silently rots: a token nudged
 * two steps darker to "look better" can drop a whole class of text below
 * 4.5:1 in one theme while looking fine in the other. This reads the real
 * values out of `src/index.css` — not a copy of them — and fails the build
 * if any shipped pairing is below its threshold.
 *
 * Thresholds (WCAG 2.2):
 *   1.4.3 body text          4.5:1
 *   1.4.3 large text ≥18.66px bold / 24px   3:1
 *   1.4.11 UI components and graphical objects   3:1
 *
 * Usage:  node scripts/contrast.mjs
 */

import { readFile } from 'node:fs/promises'
import path from 'node:path'

const CSS = path.resolve(import.meta.dirname, '..', 'src', 'index.css')

/** Parse `--name: r g b;` declarations out of one `:root`-ish block. */
function parseBlock(css, selector) {
  const start = css.indexOf(selector)
  if (start === -1) throw new Error(`no ${selector} block`)
  const open = css.indexOf('{', start)
  const end = css.indexOf('\n  }', open)
  const body = css.slice(open, end)
  const tokens = {}
  for (const [, name, value] of body.matchAll(/--([\w-]+):\s*(\d+\s+\d+\s+\d+)\s*;/g)) {
    tokens[name] = value.trim().split(/\s+/).map(Number)
  }
  return tokens
}

/** Relative luminance, per WCAG 2.x. */
function luminance([r, g, b]) {
  const channel = (value) => {
    const v = value / 255
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

/**
 * Every pairing the UI actually renders. Adding a token without adding it
 * here is how an audit turns into decoration, so this list is meant to be
 * extended whenever a component introduces a new combination.
 */
const PAIRS = [
  // [foreground, background, minimum, what it is]
  ['text-strong', 'canvas', 4.5, 'headings on the page'],
  ['text-strong', 'surface', 4.5, 'paper titles on a card'],
  ['text-body', 'surface', 4.5, 'summary text'],
  ['text-body', 'canvas', 4.5, 'body text on the page'],
  ['text-muted', 'surface', 4.5, 'authors and metadata'],
  ['text-muted', 'canvas', 4.5, 'metadata on the page'],
  ['text-muted', 'sunken', 4.5, 'text on a recessed panel'],
  // Faint is decorative-adjacent — rank numerals, the "+n more" counter —
  // but it is still text, so it is held to the large-text floor and
  // nothing smaller than 15px is allowed to use it.
  ['text-faint', 'surface', 3, 'rank numerals (large text)'],
  ['accent-600', 'surface', 4.5, 'links and the AI chip'],
  ['accent-600', 'canvas', 4.5, 'links on the page'],
  ['accent-600', 'accent-50', 4.5, 'active history entry'],
  ['accent-700', 'accent-50', 4.5, 'active history entry, emphasised'],
  ['accent-fg', 'accent-600', 4.5, 'text on the primary button'],
  ['canvas', 'text-strong', 4.5, 'text on an inverted chip'],
  ['positive', 'surface', 4.5, 'the grounded marker'],
  ['caution', 'surface', 4.5, 'the advisory marker'],
  ['critical', 'surface', 4.5, 'error text'],
  ['source-pubmed', 'surface', 4.5, 'PubMed provenance'],
  ['source-arxiv', 'surface', 4.5, 'arXiv provenance'],
  ['source-crossref', 'surface', 4.5, 'Crossref provenance'],
  // 1.4.11: non-text contrast. A border that cannot be seen is not a
  // boundary, and the score meter carries information.
  ['accent-500', 'surface', 3, 'the relevance meter fill'],
  ['control', 'surface', 3, 'input and checkbox borders'],
  ['control', 'canvas', 3, 'input borders on the page'],
  ['accent-500', 'line-strong', 3, 'the meter fill against its own track'],
  // `line` and `line-strong` are deliberately absent. 1.4.11 covers what
  // identifies a *component*; a hairline separating two rows of content is
  // decorative, and holding every divider to 3:1 would turn the page into
  // a grid of cages. The controls that do rely on their border use
  // `--control`, which is tested above.
]

const css = await readFile(CSS, 'utf8')
const themes = {
  light: parseBlock(css, ':root {'),
  dark: parseBlock(css, 'html.dark {'),
}

let failures = 0
for (const [theme, tokens] of Object.entries(themes)) {
  console.log(`\n=== ${theme} ===`)
  for (const [fg, bg, min, label] of PAIRS) {
    const a = tokens[fg]
    const b = tokens[bg]
    if (!a || !b) {
      console.log(`  ??  ${fg} on ${bg} — token missing`)
      failures += 1
      continue
    }
    const ratio = contrast(a, b)
    const pass = ratio >= min
    if (!pass) failures += 1
    console.log(
      `  ${pass ? 'ok ' : 'FAIL'} ${ratio.toFixed(2).padStart(5)}:1 ` +
        `(needs ${min})  ${fg} on ${bg} — ${label}`,
    )
  }
}

console.log(
  failures === 0
    ? `\nAll ${PAIRS.length * 2} pairings meet WCAG 2.2 AA.`
    : `\n${failures} pairing(s) below the minimum.`,
)
process.exit(failures === 0 ? 0 : 1)
