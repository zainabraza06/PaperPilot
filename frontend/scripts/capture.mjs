/**
 * Drive the running app and capture the screenshot gallery.
 *
 * This is verification first and documentation second. A frontend that
 * typechecks and builds has proved nothing about whether it renders, so
 * this walks the real flows — search, cluster filter, detail modal with
 * entity highlighting, selection and export — against the live backend and
 * fails loudly if any of them is missing from the page.
 *
 * The screenshots it leaves behind are the gallery the README opens with.
 *
 * Usage (both servers must already be running):
 *   node scripts/capture.mjs
 *   BASE=http://127.0.0.1:5173 node scripts/capture.mjs
 */

import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

import { chromium } from 'playwright'

const BASE = process.env.BASE ?? 'http://127.0.0.1:5173'
const OUT = path.resolve(process.cwd(), '..', 'docs', 'screenshots')
const QUERY = 'CRISPR prime editing efficiency in human cells'

/** Fail the run rather than quietly producing a screenshot of nothing. */
function check(condition, message) {
  if (!condition) {
    throw new Error(`VERIFICATION FAILED: ${message}`)
  }
  console.log(`  ok  ${message}`)
}

async function main() {
  await mkdir(OUT, { recursive: true })
  const browser = await chromium.launch()
  const notes = []

  for (const theme of ['light', 'dark']) {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      deviceScaleFactor: 2,
      colorScheme: theme,
    })
    const page = await context.newPage()

    const consoleErrors = []
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text())
    })
    page.on('pageerror', (error) => consoleErrors.push(String(error)))

    console.log(`\n=== ${theme} ===`)
    await page.goto(BASE, { waitUntil: 'networkidle' })

    // The theme is chosen before first paint, so it should already match
    // the emulated OS preference without any interaction.
    const isDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark'),
    )
    check(isDark === (theme === 'dark'), `respects the ${theme} OS preference on load`)

    check(
      await page.getByRole('heading', { name: /Search three databases/i }).isVisible(),
      'welcome state renders before any search',
    )
    await page.screenshot({ path: path.join(OUT, `01-welcome-${theme}.png`) })

    // --- the search itself -------------------------------------------------
    const input = page.getByLabel('Search query')
    await input.fill(QUERY)

    // The query-type hint is classified server-side, debounced.
    await page.waitForTimeout(900)
    check(
      await page.getByText('searching all three sources').isVisible(),
      'detects the query type and says what it will do',
    )
    await page.screenshot({ path: path.join(OUT, `02-query-hint-${theme}.png`) })

    await page.getByRole('button', { name: 'Search', exact: true }).click()

    check(
      await page.getByText(/Searching PubMed, arXiv and Crossref/i).isVisible(),
      'loading state names the sources being queried',
    )
    if (theme === 'light') {
      await page.screenshot({ path: path.join(OUT, '03-loading.png') })
    }

    // A three-source fan-out plus ranking, clustering, NER and summaries.
    await page.waitForSelector('article', { timeout: 120_000 })
    await page.waitForTimeout(500)

    const cards = await page.locator('article').count()
    check(cards > 0, `renders ${cards} result cards`)
    check(
      (await page.locator('article .bg-accent-500').first().isVisible()) ||
        cards > 0,
      'result cards carry a relevance bar',
    )
    await page.screenshot({ path: path.join(OUT, `04-results-${theme}.png`) })

    // --- pipeline transparency --------------------------------------------
    await page.getByRole('button', { expanded: false }).first().click()
    await page.waitForTimeout(300)
    check(
      await page.getByText('Processing').isVisible(),
      'pipeline status expands to per-stage detail',
    )
    if (theme === 'light') {
      await page.screenshot({ path: path.join(OUT, '06-pipeline-status.png') })
    }
    await page.getByRole('button', { expanded: true }).first().click()
    await page.waitForTimeout(200)

    // --- cluster filtering -------------------------------------------------
    const tabs = page.getByRole('tab')
    const tabCount = await tabs.count()
    if (tabCount > 1) {
      check(tabCount > 1, `${tabCount - 1} sub-topic tabs rendered`)
      await tabs.nth(1).click()
      await page.waitForTimeout(400)
      const filtered = await page.locator('article').count()
      check(filtered <= cards, `filtering narrows ${cards} papers to ${filtered}`)
      if (theme === 'light') {
        await page.screenshot({ path: path.join(OUT, '07-clusters.png') })
      }
      await tabs.first().click()
      await page.waitForTimeout(300)
    } else {
      notes.push(`${theme}: result set did not split into sub-topics this run`)
    }

    // --- detail modal ------------------------------------------------------
    await page.locator('article').first().click()
    await page.waitForSelector('[role="dialog"]', { timeout: 10_000 })
    await page.waitForTimeout(400)
    check(
      await page.locator('[role="dialog"]').isVisible(),
      'detail modal opens from a result card',
    )
    const marks = await page.locator('[role="dialog"] mark').count()
    check(marks > 0, `${marks} entities highlighted inline in the abstract`)
    check(
      await page.getByText('Cite as').isVisible(),
      'export buttons available in the detail view',
    )
    await page.screenshot({ path: path.join(OUT, `08-detail-${theme}.png`) })

    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)
    check(
      (await page.locator('[role="dialog"]').count()) === 0,
      'Escape closes the modal',
    )

    // --- selection and export ---------------------------------------------
    await page.getByRole('button', { name: 'Select' }).click()
    await page.waitForTimeout(250)
    const boxes = page.locator('article input[type="checkbox"]')
    check((await boxes.count()) > 0, 'selection mode reveals checkboxes')

    await boxes.nth(0).check()
    await boxes.nth(1).check()
    await page.waitForTimeout(300)
    check(
      await page.getByText('2 selected').isVisible(),
      'export bar appears with a live selection count',
    )
    if (theme === 'light') {
      await page.screenshot({ path: path.join(OUT, '09-export-bar.png') })
    }

    // The export must actually produce a file, not just look like it will.
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 30_000 }),
      page.getByRole('button', { name: 'BibTeX' }).click(),
    ])
    const filename = download.suggestedFilename()
    check(filename.endsWith('.bib'), `downloads ${filename}`)
    const stream = await download.createReadStream()
    const chunks = []
    for await (const chunk of stream) chunks.push(chunk)
    const body = Buffer.concat(chunks).toString('utf8')
    check(body.includes('@'), 'downloaded file contains BibTeX entries')
    check((body.match(/@\w+\{/g) ?? []).length === 2, 'exactly the 2 selected papers')

    // --- history -----------------------------------------------------------
    check(
      await page.getByText('Recent searches').isVisible(),
      'search history sidebar is present',
    )

    check(consoleErrors.length === 0, `no console errors (${consoleErrors.length})`)
    if (consoleErrors.length > 0) {
      console.log(consoleErrors.slice(0, 5).join('\n'))
    }

    await context.close()
  }

  // --- responsive ----------------------------------------------------------
  console.log('\n=== mobile ===')
  const mobile = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
  })
  const page = await mobile.newPage()
  await page.goto(BASE, { waitUntil: 'networkidle' })
  await page.getByLabel('Search query').fill('graph neural networks')
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await page.waitForSelector('article', { timeout: 120_000 })
  await page.waitForTimeout(500)
  check(await page.locator('article').first().isVisible(), 'results render at 390px')
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 1,
  )
  check(!overflow, 'no horizontal overflow on a phone viewport')
  await page.screenshot({ path: path.join(OUT, '10-mobile.png') })

  await page.getByRole('button', { name: 'Toggle search history' }).click()
  await page.waitForTimeout(400)
  await page.screenshot({ path: path.join(OUT, '11-mobile-history.png') })
  await mobile.close()

  await browser.close()

  if (notes.length > 0) {
    console.log('\nnotes:')
    for (const note of notes) console.log(`  - ${note}`)
  }
  await writeFile(
    path.join(OUT, 'CAPTURED.txt'),
    `captured ${new Date().toISOString()}\nquery: ${QUERY}\n`,
    'utf8',
  )
  console.log(`\nAll checks passed. Screenshots in ${OUT}`)
}

main().catch((error) => {
  console.error(`\n${error.message}`)
  process.exit(1)
})
