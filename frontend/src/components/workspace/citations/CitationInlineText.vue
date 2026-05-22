<script setup>
import { computed } from 'vue'
import { marked } from 'marked'
import { normalizeCitations } from './citationUtils'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const props = defineProps({
  text: { type: String, default: '' },
  citations: { type: Array, default: () => [] },
})

const emit = defineEmits(['citationClick'])

const normalizedCitations = computed(() => normalizeCitations(props.citations))

const renderedHtml = computed(() => {
  const citationCount = normalizedCitations.value.length
  const cleanedText = String(props.text || '')
    .replace(/\s*\[출처[:：]?[^\]]*\][^\n]*(?=\n|$)/g, '')
    .trim()

  if (!cleanedText) return ''

  let html = marked.parse(cleanedText)
  let hasMarker = false

  html = html.replace(/\[(\d+)\]/g, (match, rawNumber) => {
    const number = Number(rawNumber)
    if (!Number.isInteger(number) || number < 1 || number > citationCount) return ''
    hasMarker = true
    return markerButtonHtml(number)
  })

  if (!hasMarker && citationCount > 0) {
    const markers = normalizedCitations.value
      .map((citation) => markerButtonHtml(citation.number))
      .join('')
    html = appendMarkersToHtml(html, markers)
  }

  return html
})

function markerButtonHtml(number) {
  return `<button type="button" class="inline-citation-marker" data-citation-number="${number}" aria-label="${number}번 근거 보기">${number}</button>`
}

function appendMarkersToHtml(html, markers) {
  const markerGroup = `<span class="inline-citation-group">${markers}</span>`
  if (/<\/p>\s*$/.test(html)) {
    return html.replace(/<\/p>\s*$/, ` ${markerGroup}</p>`)
  }
  return `${html} ${markerGroup}`
}

function handleClick(event) {
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
</style>
