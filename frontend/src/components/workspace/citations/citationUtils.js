export function formatCitationSeconds(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value)) return ''
  const totalSeconds = Math.max(0, Math.floor(value))
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, '0')}`
}

export function compactCitationLabel(value = '', fallback = '근거 자료', maxLength = 34) {
  const label = String(value || '').replace(/\s+/g, ' ').trim()
  if (!label) return fallback
  return label.length > maxLength ? `${label.slice(0, maxLength).trim()}...` : label
}

export function escapeCitationHtml(value = '') {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

export function normalizeCitation(cite = {}, index = 0) {
  const isMaterial = cite?.source_type === 'material'
  const start = formatCitationSeconds(cite?.start_time)
  const end = formatCitationSeconds(cite?.end_time)
  const page = Number(cite?.page || 0)
  const title = isMaterial
    ? cite?.material_name || cite?.file_title || cite?.stored_name || `PDF ${index + 1}`
    : cite?.recording_title || cite?.session_title || cite?.file_title || `전사 ${index + 1}`
  const locationLabel = isMaterial
    ? (page > 0 ? `p.${page}` : 'PDF 원문')
    : (start && end ? `${start}~${end}` : '전사 원문')
  const fullText = String(cite?.full_transcript || cite?.text || '')
  const excerpt = String(cite?.text || '').trim()

  return {
    id: citationKey(cite, index),
    number: index + 1,
    type: isMaterial ? 'material' : 'transcript',
    icon: isMaterial ? 'picture_as_pdf' : 'graphic_eq',
    title,
    sourceCaption: isMaterial ? 'PDF 자료' : '오디오 전사',
    locationLabel,
    label: compactCitationLabel(
      isMaterial && page > 0 ? `${title} ${locationLabel}` : `${title} ${locationLabel}`,
      `근거 ${index + 1}`
    ),
    excerpt,
    fullText,
    raw: cite,
  }
}

export function normalizeCitations(citations = []) {
  const seen = new Set()
  const normalized = []

  citations.forEach((cite, index) => {
    const key = citationKey(cite, index)
    if (!key || seen.has(key)) return
    seen.add(key)
    normalized.push(normalizeCitation(cite, normalized.length))
  })

  return normalized
}

export function buildHighlightedCitationHtml(fullText = '', targetText = '') {
  const source = String(fullText || targetText || '')
  const target = String(targetText || '').trim()
  if (!source) return ''

  const ranges = findHighlightRanges(source, target)
  if (!ranges.length) return escapeCitationHtml(source)

  const parts = []
  let cursor = 0
  ranges.forEach(([start, end]) => {
    if (start < cursor) return
    parts.push(escapeCitationHtml(source.slice(cursor, start)))
    parts.push(`<mark class="cite-highlighted-script">${escapeCitationHtml(source.slice(start, end))}</mark>`)
    cursor = end
  })
  parts.push(escapeCitationHtml(source.slice(cursor)))
  return parts.join('')
}

function findHighlightRanges(source, target) {
  const primary = findHighlightRange(source, target)
  if (primary) return [expandRangeToSentence(source, primary)]

  const sentences = splitCitationSentences(target)
  const ranges = []
  sentences.forEach((sentence) => {
    const range = findHighlightRange(source, sentence)
    if (range) ranges.push(expandRangeToSentence(source, range))
  })

  return mergeRanges(ranges)
}

function splitCitationSentences(value = '') {
  const cleaned = String(value || '')
    .replace(/\[[^\]]+\]/g, ' ')
    .replace(/\([^)]*출처[^)]*\)/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (!cleaned) return []

  const matches = cleaned.match(/[^.!?。？！]+(?:다\.|요\.|입니다\.|습니다\.|[.!?。？！])?/g) || [cleaned]
  const seen = new Set()
  return matches
    .map((item) => item.trim())
    .filter((item) => normalizeWhitespace(item).length >= 8)
    .filter((item) => {
      const key = normalizeWhitespace(item)
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
}

function expandRangeToSentence(source = '', range = [0, 0]) {
  const text = String(source || '')
  let [start, end] = range
  const sentenceBoundary = /[.!?。？！]|\n/

  while (start > 0 && !sentenceBoundary.test(text[start - 1])) {
    start -= 1
  }
  while (end < text.length && !sentenceBoundary.test(text[end])) {
    end += 1
  }
  if (end < text.length && sentenceBoundary.test(text[end])) {
    end += 1
  }
  return [start, end]
}

function mergeRanges(ranges = []) {
  const sorted = ranges
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((a, b) => a[0] - b[0])
  const merged = []
  sorted.forEach(([start, end]) => {
    const last = merged[merged.length - 1]
    if (!last || start > last[1]) {
      merged.push([start, end])
      return
    }
    last[1] = Math.max(last[1], end)
  })
  return merged
}

function citationKey(cite = {}, index = 0) {
  return String(
    cite?.citation
    || cite?.material_id
    || cite?.transcript_id
    || cite?.recording_id
    || cite?.stored_name
    || cite?.text
    || index
  )
}

function findHighlightRange(source, target) {
  if (!target) return null

  const directIndex = source.indexOf(target)
  if (directIndex >= 0) return [directIndex, directIndex + target.length]

  const normalized = normalizedIndexMap(source)
  const normalizedTarget = normalizeWhitespace(target)
  if (normalizedTarget.length < 8) return null

  const normalizedIndex = normalized.text.indexOf(normalizedTarget)
  if (normalizedIndex < 0) return null

  const start = normalized.map[normalizedIndex]
  const endMapIndex = normalizedIndex + normalizedTarget.length - 1
  const end = (normalized.map[endMapIndex] ?? start) + 1
  return [start, end]
}

function normalizeWhitespace(value = '') {
  return String(value).replace(/\s+/g, ' ').trim()
}

function normalizedIndexMap(value = '') {
  let text = ''
  const map = []
  let previousWasSpace = true

  Array.from(String(value)).forEach((char, index) => {
    if (/\s/.test(char)) {
      if (previousWasSpace) return
      text += ' '
      map.push(index)
      previousWasSpace = true
      return
    }

    text += char
    map.push(index)
    previousWasSpace = false
  })

  return {
    text: text.trim(),
    map,
  }
}
