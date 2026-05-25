<script setup>
import { computed, ref, watch } from 'vue'
import { marked } from 'marked'
import { normalizeCitations } from './citationUtils'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const props = defineProps({
  text: { type: String, default: '' },
  citations: { type: Array, default: () => [] },
  enableCitations: { type: Boolean, default: false },
})

const emit = defineEmits(['citationClick'])

const normalizedCitations = computed(() => normalizeCitations(props.citations))
const isExpanded = ref(false)

const citationMarkerPattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g
const citationMarkerTestPattern = /\[(\d+(?:\s*,\s*\d+)*)\]/

watch(
  () => [props.text, props.citations],
  () => {
    isExpanded.value = false
  },
  { deep: true }
)

const renderedHtml = computed(() => {
  const citationCount = normalizedCitations.value.length
  const cleanedText = cleanupCitationText(props.text)

  if (!cleanedText) return ''

  const displayText = props.enableCitations && citationCount > 0
    ? addNotebookStyleFallbackMarkers(cleanedText, normalizedCitations.value)
    : cleanedText

  let html = marked.parse(displayText)

  html = html.replace(citationMarkerPattern, (match, rawNumbers) => {
    if (!props.enableCitations) return ''

    const numbers = uniqueCitationNumbers(rawNumbers, citationCount)
    if (!numbers.length) return ''

    return compactMarkerGroupHtml(numbers)
  })

  return html
})

function hasCitationMarkers(text) {
  return citationMarkerTestPattern.test(String(text || ''))
}

function addNotebookStyleFallbackMarkers(text, citations) {
  const fallbackState = { cursor: 0 }
  return String(text || '')
    .split('\n')
    .map((line) => addFallbackMarkersToLine(line, citations, fallbackState))
    .join('\n')
}

function addFallbackMarkersToLine(line, citations, fallbackState) {
  if (!shouldAttachFallbackMarker(line)) return line

  const sentencePattern = /[^.!?。！？]+[.!?。！？]+(?:["'”’)]*)?(?:\s*\[\d+(?:\s*,\s*\d+)*\])*/g
  let matched = false
  let lastIndex = 0
  let result = ''

  String(line).replace(sentencePattern, (sentence, index) => {
    matched = true
    result += line.slice(lastIndex, index)
    result += appendFallbackMarker(sentence, citations, fallbackState)
    lastIndex = index + sentence.length
    return sentence
  })

  if (!matched) return appendFallbackMarker(line, citations, fallbackState)

  const tail = line.slice(lastIndex)
  return `${result}${tail}`
}

function appendFallbackMarker(text, citations, fallbackState) {
  const trimmed = String(text || '').trim()
  if (!trimmed) return text
  const existingNumbers = extractCitationNumbers(trimmed, citations.length)
  const suggestedNumbers = bestCitationNumbers(trimmed, citations, fallbackState)
  const numbers = mergeCitationNumbers(existingNumbers, suggestedNumbers, citations.length)
  if (!numbers.length) return text

  const baseText = existingNumbers.length
    ? stripCitationMarkers(text)
    : text
  return `${String(baseText || '').trimEnd()} [${numbers.join(',')}]`
}

function bestCitationNumbers(sentence, citations, fallbackState) {
  const ranked = rankCitationCandidates(sentence, citations)
  const material = ranked.find((item) => item.citation.type === 'material' && item.score > 0)
  const transcript = ranked.find((item) => item.citation.type === 'transcript' && item.score > 0)
  const mixedNumbers = [material?.citation?.number, transcript?.citation?.number]
    .filter(Boolean)

  if (mixedNumbers.length) return [...new Set(mixedNumbers)].sort((a, b) => a - b)

  const best = ranked[0]
  if (best?.score > 0) return [best.citation.number]

  const fallback = citations[Math.min(fallbackState.cursor, citations.length - 1)] || citations[0]
  fallbackState.cursor += 1
  return fallback?.number ? [fallback.number] : []
}

function rankCitationCandidates(sentence, citations) {
  return citations
    .map((citation) => ({
      citation,
      score: citationOverlapScore(sentence, citationSearchText(citation)),
    }))
    .sort((a, b) => b.score - a.score || a.citation.number - b.citation.number)
}

function citationSearchText(citation) {
  return [
    citation.title,
    citation.locationLabel,
    citation.sourceCaption,
    citation.excerpt,
    citation.fullText,
  ].join(' ')
}

function citationOverlapScore(sentence, sourceText) {
  const sentenceTerms = keywordSet(sentence)
  if (!sentenceTerms.size) return 0
  const source = String(sourceText || '').toLowerCase()
  let score = 0
  sentenceTerms.forEach((term) => {
    if (source.includes(term)) score += 1
  })
  return score
}

function keywordSet(value) {
  const stopwords = new Set(['그리고', '하지만', '또한', '그래서', '예를', '들어', '이것', '저것', '수', '있습니다', '입니다', '합니다', '됩니다'])
  const terms = String(value || '')
    .toLowerCase()
    .match(/[가-힣a-z0-9_+#.-]{2,}/g) || []
  return new Set(terms.filter((term) => !stopwords.has(term)))
}

function shouldAttachFallbackMarker(line) {
  const trimmed = String(line || '').trim()
  if (!trimmed) return false
  if (!stripCitationMarkers(trimmed).trim()) return false
  if (/^#{1,6}\s+/.test(trimmed)) return false
  if (/^[-*]\s*$/.test(trimmed)) return false
  if (/^[\d.)\s-]*$/.test(trimmed)) return false
  if (trimmed.length <= 18 && /[:：]$/.test(trimmed)) return false
  return true
}

function cleanupCitationText(text) {
  const withoutInlineSourceLabels = String(text || '')
    .replace(/\s*\[출처[:：]?[^\]]*\][^\n]*(?=\n|$)/g, '')
    .replace(/(^|\n)\s*(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+\s*(?=\n|$)/g, '$1')
  return stripTrailingSourceSection(withoutInlineSourceLabels)
    .trim()
}

function stripTrailingSourceSection(text) {
  const sourceHeadingPattern = /(?:^|\n)\s*(?:#{1,6}\s*)?(?:출처|참고자료|참고 문헌|Sources?|References?)\s*[:：]?\s*(?:\n|$)/i
  const match = text.match(sourceHeadingPattern)
  if (!match) return text

  const before = text.slice(0, match.index).trimEnd()
  const after = text.slice((match.index || 0) + match[0].length).trim()
  if (!before || !looksLikeSourceList(after)) return text
  return before
}

function looksLikeSourceList(text) {
  const lines = text.split(/\n+/).map((line) => line.trim()).filter(Boolean)
  if (!lines.length) return false

  return lines.every((line) => (
    /^\d+[\).]?\s+/.test(line)
    || /^\[\d+\]\s+/.test(line)
    || /^[-*]\s+/.test(line)
    || /(?:\.pdf|\.m4a|\.wav|p\.\d+|페이지|녹음|길이|시간)/i.test(line)
  ))
}

function extractCitationNumbers(value, maxNumber) {
  const numbers = []
  String(value || '').replace(citationMarkerPattern, (match, rawNumbers) => {
    numbers.push(...uniqueCitationNumbers(rawNumbers, maxNumber))
    return match
  })
  return numbers
}

function mergeCitationNumbers(existingNumbers, suggestedNumbers, maxNumber) {
  return [...existingNumbers, ...suggestedNumbers]
    .map((number) => Number(number))
    .filter((number) => Number.isInteger(number) && number >= 1 && number <= maxNumber)
    .filter((number, index, numbers) => numbers.indexOf(number) === index)
    .sort((a, b) => a - b)
}

function uniqueCitationNumbers(rawNumbers, maxNumber) {
  return String(rawNumbers || '')
    .split(',')
    .map((number) => Number(number.trim()))
    .filter((number) => Number.isInteger(number) && number >= 1 && number <= maxNumber)
    .filter((number, index, numbers) => numbers.indexOf(number) === index)
}

function stripCitationMarkers(value) {
  return String(value || '')
    .replace(citationMarkerPattern, '')
    .replace(/\s{2,}/g, ' ')
    .replace(/\s+([.,!?。！？])/g, '$1')
}

function markerButtonHtml(number) {
  return `<button type="button" class="inline-citation-marker" data-citation-number="${number}" aria-label="${number}번 근거 보기">${number}</button>`
}

function moreButtonHtml(count) {
  return `<button type="button" class="inline-citation-more" data-citation-more aria-label="${count}개 근거 펼치기">+</button>`
}

function markerGroupHtml(numbers) {
  const markers = numbers.map((number) => markerButtonHtml(number)).join('')
  return `<span class="inline-citation-group">${markers}</span>`
}

function compactMarkerGroupHtml(numbers) {
  if (numbers.length <= 1 || isExpanded.value) return markerGroupHtml(numbers)
  return `<span class="inline-citation-group">${markerButtonHtml(numbers[0])}${moreButtonHtml(numbers.length)}</span>`
}

function handleClick(event) {
  const moreButton = event.target?.closest?.('[data-citation-more]')
  if (moreButton) {
    isExpanded.value = true
    return
  }

  const button = event.target?.closest?.('[data-citation-number]')
  if (!button) return

  const number = Number(button.dataset.citationNumber)
  const citation = normalizedCitations.value[number - 1]
  if (!citation) return

  emit('citationClick', {
    cite: citation.raw,
    target: button,
  })
}
</script>

<template>
  <div
    class="citation-inline-text"
    @click="handleClick"
    v-html="renderedHtml"
  />
</template>

<style scoped>
.citation-inline-text :deep(p) {
  margin-bottom: 0.5em;
}

.citation-inline-text :deep(p:last-child) {
  margin-bottom: 0;
}

.citation-inline-text :deep(.inline-citation-group) {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: 4px;
  vertical-align: baseline;
}

.citation-inline-text :deep(.inline-citation-marker) {
  width: 22px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin: 0 2px;
  color: #475569;
  background: #f1f5f9;
  border: 1px solid rgba(226, 232, 240, 0.96);
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  line-height: 1;
  cursor: pointer;
  vertical-align: text-bottom;
  transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.citation-inline-text :deep(.inline-citation-marker:hover) {
  color: #111827;
  background: #e2e8f0;
  border-color: #cbd5e1;
  transform: translateY(-1px);
}

.citation-inline-text :deep(.inline-citation-more) {
  width: 22px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin: 0 2px;
  color: #2563eb;
  background: #eff6ff;
  border: 1px solid rgba(191, 219, 254, 0.96);
  border-radius: 999px;
  font-size: 13px;
  font-weight: 900;
  line-height: 1;
  cursor: pointer;
  vertical-align: text-bottom;
  transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.citation-inline-text :deep(.inline-citation-more:hover) {
  color: #1d4ed8;
  background: #dbeafe;
  border-color: #93c5fd;
  transform: translateY(-1px);
}

</style>
