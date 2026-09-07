/**
 * Readable aliases over the generated OpenAPI schema.
 *
 * `api.ts` is generated from the backend's own OpenAPI document and must
 * never be edited by hand — which also makes it unpleasant to import from
 * directly, since every type is buried under
 * `components['schemas'][...]`. This module is the single place that
 * reaches into it, so the rest of the app imports plain names and a
 * backend field rename shows up as a type error here rather than as
 * `undefined` in a component.
 */

import type { components } from './api'

type Schemas = components['schemas']

export type Paper = Schemas['Paper']
export type Author = Schemas['Author']
export type SearchResponse = Schemas['SearchResponse']
export type ParsedQuery = Schemas['ParsedQuery']
export type SourceReport = Schemas['SourceReport']
export type RankingReport = Schemas['RankingReport']
export type ClusteringReport = Schemas['ClusteringReport']
export type EnrichmentReport = Schemas['EnrichmentReport']
export type SummaryReport = Schemas['SummaryReport']
export type TopicCluster = Schemas['TopicCluster']
export type Summary = Schemas['Summary']
export type GroundingVerdict = Schemas['GroundingVerdict']
export type GroundingIssue = Schemas['GroundingIssue']
export type Entity = Schemas['Entity']
export type EntitySpan = Schemas['EntitySpan']
export type RelevanceScore = Schemas['RelevanceScore']
export type HealthResponse = Schemas['HealthResponse']

export type SourceName = Schemas['SourceName']
export type SourceStatus = Schemas['SourceStatus']
export type QueryIntent = Schemas['QueryIntent']
export type EntityLabel = Schemas['EntityLabel']
export type ExportFormat = Schemas['ExportFormat']
export type SummaryOrigin = Schemas['SummaryOrigin']
export type GroundingStatus = Schemas['GroundingStatus']

/** Display metadata for the three providers, keyed by their API value. */
export const SOURCE_LABELS: Record<SourceName, string> = {
  pubmed: 'PubMed',
  arxiv: 'arXiv',
  crossref: 'Crossref',
}

/**
 * How each per-source outcome should read to a user.
 *
 * `empty` is deliberately not phrased as a failure: "arXiv has nothing on
 * this" is a true and useful statement about a biomedical query, and the
 * backend distinguishes it from an outage precisely so the UI can too.
 */
export const SOURCE_STATUS_COPY: Record<SourceStatus, { label: string; tone: Tone }> = {
  ok: { label: 'returned results', tone: 'ok' },
  empty: { label: 'no matches', tone: 'neutral' },
  timeout: { label: 'timed out', tone: 'warn' },
  rate_limited: { label: 'rate limited', tone: 'warn' },
  unavailable: { label: 'unavailable', tone: 'error' },
  parse_error: { label: 'unreadable response', tone: 'error' },
  skipped: { label: 'skipped', tone: 'neutral' },
}

export type Tone = 'ok' | 'neutral' | 'warn' | 'error'

/** Entity categories that are worth showing as filter chips. */
export const ENTITY_LABEL_COPY: Record<EntityLabel, string> = {
  gene_or_protein: 'Gene / protein',
  disease: 'Disease',
  chemical: 'Chemical',
  organism: 'Organism',
  anatomy: 'Anatomy',
  cell_type: 'Cell type',
  technical_term: 'Method / term',
  organization: 'Organisation',
  person: 'Person',
  location: 'Location',
  other: 'Other',
}

export const EXPORT_FORMAT_LABELS: Record<ExportFormat, string> = {
  bibtex: 'BibTeX',
  ris: 'RIS',
  apa: 'APA 7th',
  vancouver: 'Vancouver',
}

/** A query the user ran, kept for the history sidebar. */
export interface HistoryEntry {
  query: string
  intent: QueryIntent
  resultCount: number
  /** Epoch milliseconds — stored as a number so it survives JSON round-trips. */
  ranAt: number
}
