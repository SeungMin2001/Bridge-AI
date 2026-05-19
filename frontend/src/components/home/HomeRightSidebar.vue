<!-- 홈 화면에서 선택한 강의 자료나 AI 분석 결과를 상세하게 보여주는 오른쪽 사이드바입니다. -->
<script setup>
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps({
  isOpen: {
    type: Boolean,
    default: false
  },
  referenceData: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['close', 'openFile'])
const scriptScrollRef = ref(null)
const referenceSearchInput = ref(null)
const referenceSearch = ref('')
const isReferenceSearchOpen = ref(false)
const activeReferenceSearchIndex = ref(0)

const titleParts = computed(() => {
  const raw = props.referenceData?.raw || {}
  const rawTitle = raw.session_title || raw.recording_title || props.referenceData?.title || '스크립트'
  const parts = String(rawTitle)
    .split('>')
    .map((part) => part.trim())
    .filter(Boolean)

  if (parts.length >= 2) {
    return {
      title: parts.slice(0, -1).join(' > '),
      subtitle: parts.at(-1)
    }
  }

  const citationParts = String(raw.citation || '')
    .split('>')
    .map((part) => part.trim())
    .filter(Boolean)

  return {
    title: rawTitle,
    subtitle: citationParts.length >= 2 ? citationParts.at(-1) : ''
  }
})
const title = computed(() => titleParts.value.title)
const subtitle = computed(() => titleParts.value.subtitle)
const script = computed(() => (
  props.referenceData?.script
  || props.referenceData?.raw?.full_transcript
  || props.referenceData?.raw?.transcript
  || props.referenceData?.raw?.source_text
  || props.referenceData?.raw?.text
  || '스크립트 내용이 없습니다.'
))
const canOpenFile = computed(() => Boolean(props.referenceData?.raw?.session_id))

const escapeHtml = (value = '') => (
  String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
)

const findSearchMatches = (text = '', query = '') => {
  const source = String(text || '')
  const needle = String(query || '').trim().toLowerCase()
  if (!needle) return []

  const lowerSource = source.toLowerCase()
  const matches = []
  let cursor = 0
  while (cursor < lowerSource.length) {
    const index = lowerSource.indexOf(needle, cursor)
    if (index === -1) break
    matches.push({ start: index, end: index + needle.length })
    cursor = index + needle.length
  }
  return matches
}

const searchMatches = computed(() => findSearchMatches(script.value, referenceSearch.value))
const hasReferenceSearchTerm = computed(() => referenceSearch.value.trim().length > 0)
const hasReferenceSearchResults = computed(() => hasReferenceSearchTerm.value && searchMatches.value.length > 0)
const referenceSearchCountLabel = computed(() => {
  if (!hasReferenceSearchTerm.value) return ''
  if (!searchMatches.value.length) return '0개'
  return `${activeReferenceSearchIndex.value + 1}/${searchMatches.value.length}`
})
const highlightedScript = computed(() => {
  const fullText = script.value
  const matches = searchMatches.value

  if (hasReferenceSearchTerm.value && matches.length) {
    let html = ''
    let cursor = 0
    matches.forEach((match, index) => {
      html += escapeHtml(fullText.slice(cursor, match.start))
      const className = index === activeReferenceSearchIndex.value
        ? 'home-reference-search-highlight is-active'
        : 'home-reference-search-highlight'
      html += `<mark class="${className}" data-search-index="${index}">${escapeHtml(fullText.slice(match.start, match.end))}</mark>`
      cursor = match.end
    })
    html += escapeHtml(fullText.slice(cursor))
    return html
  }

  const target = String(props.referenceData?.raw?.text || '').trim()

  if (target && fullText.includes(target)) {
    const highlightedTarget = `<mark class="home-reference-highlight">${escapeHtml(target)}</mark>`
    return fullText.split(target).map((part) => escapeHtml(part)).join(highlightedTarget)
  }

  return escapeHtml(fullText)
})

const scrollToSearchResult = async (index = activeReferenceSearchIndex.value, behavior = 'smooth') => {
  if (!hasReferenceSearchResults.value) return
  await nextTick()
  const count = searchMatches.value.length
  activeReferenceSearchIndex.value = Math.min(Math.max(index, 0), count - 1)
  await nextTick()

  window.requestAnimationFrame(() => {
    const container = scriptScrollRef.value
    const target = container?.querySelector('.home-reference-search-highlight.is-active')
    if (!container || !target) return
    const containerRect = container.getBoundingClientRect()
    const targetRect = target.getBoundingClientRect()
    const targetTop = targetRect.top - containerRect.top + container.scrollTop
    container.scrollTo({ top: Math.max(0, targetTop - 80), behavior })
  })
}

const openReferenceSearch = async () => {
  isReferenceSearchOpen.value = true
  await nextTick()
  referenceSearchInput.value?.focus()
}

const closeReferenceSearch = () => {
  isReferenceSearchOpen.value = false
}

const clearReferenceSearch = () => {
  referenceSearch.value = ''
  activeReferenceSearchIndex.value = 0
  closeReferenceSearch()
}

const moveReferenceSearch = (direction = 1) => {
  if (!hasReferenceSearchResults.value) return
  const count = searchMatches.value.length
  const nextIndex = (activeReferenceSearchIndex.value + direction + count) % count
  scrollToSearchResult(nextIndex)
}

const scrollToHighlightedScript = async () => {
  await nextTick()
  window.requestAnimationFrame(() => {
    const container = scriptScrollRef.value
    if (!container) return

    const firstHighlight = container.querySelector('.home-reference-highlight')
    if (!firstHighlight) {
      container.scrollTo({ top: 0, behavior: 'auto' })
      return
    }

    const containerRect = container.getBoundingClientRect()
    const highlightRect = firstHighlight.getBoundingClientRect()
    const highlightTop = highlightRect.top - containerRect.top + container.scrollTop
    container.scrollTo({ top: Math.max(0, highlightTop - 24), behavior: 'auto' })
  })
}

watch(
  () => [
    props.isOpen,
    props.referenceData?.id,
    props.referenceData?.title,
    props.referenceData?.raw?.transcript_id,
    props.referenceData?.raw?.citation,
    props.referenceData?.raw?.text,
  ],
  () => {
    if (props.isOpen) scrollToHighlightedScript()
  },
  { flush: 'post' }
)

watch([referenceSearch, () => searchMatches.value.length], () => {
  activeReferenceSearchIndex.value = 0
  if (hasReferenceSearchResults.value) scrollToSearchResult(0, 'auto')
})
</script>

<template>
  <div 
    :class="[
      'home-reference-sidebar fixed right-3 top-[10px] bottom-[10px] h-[calc(100%-20px)] w-[416px] transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] z-[100] flex flex-col',
      isOpen ? 'translate-x-0' : 'translate-x-[120%]'
    ]"
  >
    <!-- Header -->
    <div class="h-16 shrink-0 flex items-center justify-between px-6 border-b border-[#eeeaf3] bg-white">
      <h2 class="text-[16px] font-bold text-[#1d1d1f] flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center">
          <span class="material-symbols-outlined text-[16px] text-indigo-600">article</span>
        </div>
        <span class="min-w-0 flex flex-col">
          <span class="truncate pr-4">{{ title }}</span>
          <span v-if="subtitle" class="home-reference-subtitle truncate pr-4">{{ subtitle }}</span>
        </span>
      </h2>
      <button 
        @click="emit('close')" 
        class="w-8 h-8 flex items-center justify-center rounded-full hover:bg-black/5 text-gray-400 hover:text-gray-700 transition-colors"
      >
        <span class="material-symbols-outlined text-[20px]">close</span>
      </button>
    </div>

    <div class="home-reference-script-head" :class="{ 'is-search-open': isReferenceSearchOpen }">
      <h3 class="home-reference-script-title">전사 내용 스크립트</h3>
      <template v-if="!isReferenceSearchOpen">
        <button
          type="button"
          class="home-reference-search-trigger"
          :class="{ 'is-active': hasReferenceSearchTerm }"
          aria-label="전사 내용 검색"
          @click="openReferenceSearch"
        >
          <span class="material-symbols-outlined">search</span>
        </button>
      </template>

      <section
        v-else
        class="home-reference-search-inline"
        role="search"
        aria-label="전사 내용 검색"
      >
        <div class="home-reference-search-input-row">
          <span class="material-symbols-outlined home-reference-search-leading-icon">search</span>
          <input
            ref="referenceSearchInput"
            v-model="referenceSearch"
            class="home-reference-search-input"
            placeholder="검색"
            type="text"
            @keydown.esc="closeReferenceSearch"
            @keydown.enter.prevent="moveReferenceSearch($event.shiftKey ? -1 : 1)"
          />
          <span
            class="home-reference-search-count"
            :class="{
              'is-idle': !hasReferenceSearchTerm,
              'is-empty': hasReferenceSearchTerm && !hasReferenceSearchResults
            }"
            aria-live="polite"
          >
            {{ referenceSearchCountLabel }}
          </span>
          <button
            type="button"
            class="home-reference-search-nav"
            aria-label="이전 검색 결과"
            :disabled="!hasReferenceSearchResults"
            @click="moveReferenceSearch(-1)"
          >
            <span class="material-symbols-outlined">keyboard_arrow_up</span>
          </button>
          <button
            type="button"
            class="home-reference-search-nav"
            aria-label="다음 검색 결과"
            :disabled="!hasReferenceSearchResults"
            @click="moveReferenceSearch(1)"
          >
            <span class="material-symbols-outlined">keyboard_arrow_down</span>
          </button>
          <button
            type="button"
            class="home-reference-search-close"
            aria-label="검색 닫기"
            @click="clearReferenceSearch"
          >
            <span class="material-symbols-outlined">close</span>
          </button>
        </div>
      </section>
    </div>

    <!-- Content -->
    <div ref="scriptScrollRef" class="home-reference-content custom-scrollbar">
      <div class="home-reference-script-card">
        <div class="home-reference-script-text" v-html="highlightedScript"></div>
      </div>
    </div>

    <div class="home-reference-footer shrink-0 p-5 border-t border-slate-200 bg-white/55">
      <button
        type="button"
        class="home-reference-open-file-btn"
        :disabled="!canOpenFile"
        @click="emit('openFile', referenceData)"
      >
        <span>파일로 이동</span>
        <span class="material-symbols-outlined text-[18px]">arrow_forward</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 10px;
}

.home-reference-sidebar {
  background: #fff;
  border: 1px solid rgba(255, 255, 255, 0.92);
  border-radius: var(--copy-radius-lg);
  box-shadow: var(--copy-card-shadow);
  overflow: hidden;
}

.home-reference-subtitle {
  margin-top: 2px;
  color: #8f8b98;
  font-size: 12px;
  font-weight: 800;
  line-height: 1.1;
}

.home-reference-content {
  flex: 1;
  overflow-y: auto;
  padding: 24px 28px 26px;
  background: #fff;
}

.home-reference-script-card {
  min-height: 100%;
  position: relative;
}

.home-reference-script-head {
  position: relative;
  z-index: 30;
  min-height: 48px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 28px 8px;
  background: #fff;
  border-bottom: 1px solid #f0eef5;
}

.home-reference-script-head.is-search-open {
  min-height: 48px;
  align-items: center;
  box-shadow: 0 10px 20px rgba(48, 42, 58, 0.045);
}

.home-reference-script-title {
  flex: 0 0 auto;
  color: #9aa1af;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0;
}

.home-reference-search-trigger {
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  color: #8f96a3;
  background: transparent;
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.16s ease;
}

.home-reference-search-trigger:hover,
.home-reference-search-trigger.is-active {
  color: #17171c;
  background: #f2f3f7;
}

.home-reference-search-trigger:active {
  transform: scale(0.94);
}

.home-reference-search-trigger .material-symbols-outlined {
  font-size: 21px;
}

.home-reference-search-inline {
  width: 244px;
  max-width: calc(100% - 116px);
  min-width: 206px;
  flex: 0 1 244px;
  overflow: hidden;
  border-radius: 14px;
  background: #f7f8fb;
  border: 1px solid rgba(226, 224, 232, 0.9);
}

.home-reference-search-input-row {
  min-height: 36px;
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr) max-content 20px 20px 20px;
  align-items: center;
  gap: 4px;
  padding: 0 10px;
}

.home-reference-search-leading-icon {
  color: #17171c;
  font-size: 19px;
}

.home-reference-search-input {
  min-width: 0;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: #17171c;
  font-size: 12px;
  font-weight: 800;
  outline: none;
  box-shadow: none;
}

.home-reference-search-input:focus {
  outline: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
  --tw-ring-color: transparent;
  --tw-ring-shadow: 0 0 #0000;
}

.home-reference-search-input::placeholder {
  color: #a1a7b3;
  font-weight: 850;
}

.home-reference-search-count {
  width: 38px;
  overflow: hidden;
  text-align: right;
  color: #4b5563;
  font-size: 11px;
  font-weight: 900;
  line-height: 1;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.home-reference-search-count.is-idle {
  opacity: 0;
}

.home-reference-search-count.is-empty {
  color: #a1a7b3;
}

.home-reference-search-nav,
.home-reference-search-close {
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #8f96a3;
  background: transparent;
  transition: background-color 0.18s ease, color 0.18s ease;
}

.home-reference-search-nav:not(:disabled):hover,
.home-reference-search-close:hover {
  color: #17171c;
  background: #f2f3f7;
}

.home-reference-search-nav:disabled {
  opacity: 0.55;
  cursor: default;
}

.home-reference-search-nav .material-symbols-outlined,
.home-reference-search-close .material-symbols-outlined {
  font-size: 17px;
}

.home-reference-script-text {
  color: #3d4048;
  font-size: 14px;
  font-weight: 750;
  line-height: 1.78;
  white-space: pre-wrap;
  word-break: keep-all;
}

:deep(.home-reference-highlight) {
  background: #fff0a8;
  color: #111827;
  font-weight: 900;
  border-radius: 6px;
  padding: 1px 4px;
  margin: 0 -1px;
  box-shadow: none;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

:deep(.home-reference-search-highlight) {
  background: rgba(254, 240, 138, 0.72);
  color: #111827;
  font-weight: 900;
  border-radius: 5px;
  padding: 1px 3px;
  box-shadow: inset 0 -0.34em 0 rgba(250, 204, 21, 0.38);
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

:deep(.home-reference-search-highlight.is-active) {
  background: #fde047;
  outline: 2px solid rgba(47, 128, 237, 0.7);
  outline-offset: 2px;
}

.home-reference-footer {
  background: #fff;
  box-shadow: none;
}

.home-reference-open-file-btn {
  width: 100%;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-radius: 999px;
  background: var(--copy-black);
  color: #fff;
  font-size: 14px;
  font-weight: 800;
  border: 1px solid var(--copy-black);
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.home-reference-open-file-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}

.home-reference-open-file-btn:disabled {
  cursor: not-allowed;
  opacity: 0.35;
}
</style>
