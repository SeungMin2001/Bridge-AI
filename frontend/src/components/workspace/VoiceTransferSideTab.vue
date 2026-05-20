<!-- 실시간 음성 전사 결과를 확인하고 AI에게 질문하거나 노트에 추가하는 사이드 탭 컴포넌트입니다. -->
<script setup>
import { computed, ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useChat } from '../../composables/useChat'
import LoadingHourglass from '../ui/LoadingHourglass.vue'

const { selectWord } = useChat()

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  recordingMode: { type: String, default: 'lecture' },
  diarizationEnabled: { type: Boolean, default: false },
  diarizationStatus: { type: String, default: 'idle' },
  variant: { type: String, default: 'sidebar' },
  showToolbar: { type: Boolean, default: false },
  toolbarTitle: { type: String, default: '스크립트' },
  showFolderToggle: { type: Boolean, default: false },
  folderOpen: { type: Boolean, default: false }
})

const emit = defineEmits(['addToNote', 'askAi', 'toggleFolder'])

const transSearch = ref('')
const scrollContainer = ref(null)
const isSearchOpen = ref(false)
const searchTrigger = ref(null)
const searchInput = ref(null)
const searchPopoverStyle = ref({})
const searchResultRefs = ref([])
const activeSearchIndex = ref(0)
const emptyTranscriptAnimationRef = ref(null)
const isDiarizationBootstrapping = computed(() => (
  props.diarizationEnabled && props.diarizationStatus === 'bootstrapping'
))
const filteredTranscriptions = computed(() => (
  props.transcriptions.filter((item) => (
    String(item.text || '').toLowerCase().includes(transSearch.value.toLowerCase())
  ))
))

const playEmptyTranscriptAnimation = () => {
  emptyTranscriptAnimationRef.value?.playFromStart?.()
}
const hasSearchTerm = computed(() => transSearch.value.trim().length > 0)
const normalizedSearchTerm = computed(() => transSearch.value.trim().toLowerCase())
const hasSearchResults = computed(() => hasSearchTerm.value && filteredTranscriptions.value.length > 0)
const searchStatusText = computed(() => {
  if (!hasSearchTerm.value) return '검색어를 입력해주세요.'
  if (!filteredTranscriptions.value.length) return `"${transSearch.value}"에 대한 검색 결과가 없습니다`
  return `검색 결과 ${filteredTranscriptions.value.length}개 · ${activeSearchIndex.value + 1}/${filteredTranscriptions.value.length}`
})

// 최하단으로 스크롤 이동
const scrollToBottom = async () => {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTo({
      top: scrollContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}

// 전사 데이터가 변경될 때마다 스크롤 이동
watch(() => props.transcriptions, () => {
  if (hasSearchTerm.value) {
    scrollToSearchResult(activeSearchIndex.value, 'auto')
    return
  }
  scrollToBottom()
}, { deep: true })

watch([normalizedSearchTerm, () => filteredTranscriptions.value.length], () => {
  activeSearchIndex.value = 0
  searchResultRefs.value = []
  if (hasSearchResults.value) scrollToSearchResult(0, 'auto')
})

onMounted(() => {
  scrollToBottom()
  window.addEventListener('resize', updateSearchPopoverPosition)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateSearchPopoverPosition)
})

const updateSearchPopoverPosition = () => {
  if (!searchTrigger.value) return
  const rect = searchTrigger.value.getBoundingClientRect()
  const width = 320
  const gap = 8
  const viewportPadding = 12
  const preferredLeft = rect.right + gap
  const left = preferredLeft + width <= window.innerWidth - viewportPadding
    ? preferredLeft
    : Math.max(viewportPadding, rect.left - width - gap)

  searchPopoverStyle.value = {
    left: `${left}px`,
    top: `${rect.top}px`,
    width: `${width}px`
  }
}

const openSearchPopup = async () => {
  isSearchOpen.value = true
  await nextTick()
  updateSearchPopoverPosition()
  await nextTick()
  searchInput.value?.focus()
}

const closeSearchPopup = () => {
  isSearchOpen.value = false
}

const clearSearch = () => {
  transSearch.value = ''
  activeSearchIndex.value = 0
  closeSearchPopup()
}

const setSearchResultRef = (element, index) => {
  if (element) searchResultRefs.value[index] = element
}

const scrollToSearchResult = async (index = activeSearchIndex.value, behavior = 'smooth') => {
  if (!hasSearchResults.value) return
  await nextTick()
  const safeIndex = Math.min(Math.max(index, 0), filteredTranscriptions.value.length - 1)
  activeSearchIndex.value = safeIndex
  const target = searchResultRefs.value[safeIndex]
  target?.scrollIntoView?.({ behavior, block: 'center' })
}

const moveSearchResult = (direction = 1) => {
  if (!hasSearchResults.value) return
  const count = filteredTranscriptions.value.length
  const nextIndex = (activeSearchIndex.value + direction + count) % count
  scrollToSearchResult(nextIndex)
}

const isSearchHighlightedWord = (word = '') => (
  normalizedSearchTerm.value &&
  String(word || '').toLowerCase().includes(normalizedSearchTerm.value)
)

const formatElapsedTime = (seconds = 0) => {
  const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0))
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const remainSeconds = String(safeSeconds % 60).padStart(2, '0')
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${remainSeconds}`
  return `${minutes}:${remainSeconds}`
}

const getFiniteNumber = (value) => {
  if (value === null || value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const getTranscriptionTime = (transcription = {}) => {
  const starts = []
  const transcriptionStart = getFiniteNumber(transcription.start ?? transcription.start_time ?? transcription.startTime)
  if (transcriptionStart !== null) starts.push(transcriptionStart)

  if (Array.isArray(transcription.segments)) {
    transcription.segments.forEach((segment) => {
      const segmentStart = getFiniteNumber(segment?.start_time ?? segment?.startTime ?? segment?.start)
      if (segmentStart !== null) starts.push(segmentStart)
    })
  }

  if (starts.length) return formatElapsedTime(Math.min(...starts))
  return transcription.time || ''
}

// 단어 클릭 → 전역 상태로 전달하여 메인 컨텐츠 영역에 카드로 표시
const handleWordClick = (e, word, context = '') => {
  e.stopPropagation()
  selectWord(word, context)
}

const speakerProfiles = [
  { image: '/images/woman1.png', accent: 'blue' },
  { image: '/images/man1.png', accent: 'rose' },
  { image: '/images/woman2.png', accent: 'green' },
  { image: '/images/man2.png', accent: 'amber' }
]

// 전사 상태에 남아 있는 speaker_id를 화자별 아바타/색상으로 매핑할 때 쓰는 보조 함수들입니다.
const getSpeakerIndex = (speakerId = '') => {
  const match = String(speakerId).match(/^SPEAKER_(\d+)$/i)
  if (!match) return null
  return Number.parseInt(match[1], 10)
}

const getSpeakerProfile = (transcription) => {
  const speakerIndex = getSpeakerIndex(transcription?.speakerId)
  if (speakerIndex === null || Number.isNaN(speakerIndex)) return null
  return speakerProfiles[speakerIndex % speakerProfiles.length]
}

const shouldShowSpeaker = (transcription) => (
  props.recordingMode === 'meeting' || !!transcription.speaker || !!transcription.speakerId
)

const getSpeakerLabel = (transcription) => {
  if (transcription.speaker) return transcription.speaker
  const speakerIndex = getSpeakerIndex(transcription.speakerId)
  if (speakerIndex !== null && !Number.isNaN(speakerIndex)) return `화자 ${speakerIndex + 1}`
  if (shouldShowSpeaker(transcription)) return transcription.speakerId || '화자 미상'
  return '나'
}

const getSpeakerAvatarSrc = (transcription) => {
  const profile = getSpeakerProfile(transcription)
  if (profile) return profile.image
  const label = getSpeakerLabel(transcription)
  if (label === '화자 B' || label === '화자 2') return '/images/man1.png'
  return '/images/woman1.png'
}

const getSpeakerAccent = (transcription) => {
  const profile = getSpeakerProfile(transcription)
  if (profile) return profile.accent
  const label = getSpeakerLabel(transcription)
  if (label === '나') return 'blue'
  if (label === '화자 B' || label === '화자 2') return 'rose'
  if (label === '화자 C' || label === '화자 3') return 'green'
  return 'amber'
}

const getSpeakerAvatarClass = (transcription) => `speaker-avatar-${getSpeakerAccent(transcription)}`
</script>

<template>
  <div class="flex flex-col flex-1 overflow-hidden" :class="{ 'transcript-panel-content': variant === 'content' }">
    <div
      v-if="showToolbar"
      class="transcript-toolbar"
      :class="{ 'has-folder-toggle': showFolderToggle }"
    >
      <span class="transcript-toolbar-title">{{ toolbarTitle }}</span>
      <div class="transcript-toolbar-actions">
        <button
          v-if="showFolderToggle"
          type="button"
          class="transcript-source-toggle"
          :class="{ 'is-open': folderOpen }"
          :aria-label="folderOpen ? '소스파일 닫기' : '소스파일 열기'"
          @click="emit('toggleFolder')"
        >
          <span class="transcript-source-toggle-arrow" aria-hidden="true">
            {{ folderOpen ? '>' : '<' }}
          </span>
          <span>소스파일</span>
        </button>
        <button
          ref="searchTrigger"
          type="button"
          class="transcript-search-trigger transcript-toolbar-search"
          :class="{ 'is-active': isSearchOpen || hasSearchTerm }"
          aria-label="전사 내용 검색"
          @click="isSearchOpen ? closeSearchPopup() : openSearchPopup()"
        >
          <span class="material-symbols-outlined">search</span>
        </button>
      </div>
    </div>
    <!-- 검색 팝업 -->
    <div v-else class="transcript-search-anchor">
      <button
        ref="searchTrigger"
        type="button"
        class="transcript-search-trigger"
        :class="{ 'is-active': isSearchOpen || hasSearchTerm }"
        aria-label="전사 내용 검색"
        @click="isSearchOpen ? closeSearchPopup() : openSearchPopup()"
      >
        <span class="material-symbols-outlined">search</span>
      </button>
    </div>

    <!-- 전사 기록 리스트 -->
    <div 
      ref="scrollContainer"
      class="transcript-list flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-4 pb-4"
    >
      <template v-if="filteredTranscriptions.length === 0">
        <div v-if="isDiarizationBootstrapping" class="diarization-preparing-state" role="status" aria-live="polite">
          <div class="diarization-preparing-icon">
            <span class="material-symbols-outlined">graphic_eq</span>
          </div>
          <strong>화자 분리 중...</strong>
          <p>처음 몇 초의 음성을 분석한 뒤 전사를 표시합니다.</p>
        </div>
        <div v-else class="empty-transcript-state flex flex-col items-center justify-center h-full py-10">
          <button
            type="button"
            class="empty-transcript-animation-trigger"
            aria-label="전사 없음 애니메이션 재생"
            @click="playEmptyTranscriptAnimation"
          >
            <LoadingHourglass
              ref="emptyTranscriptAnimationRef"
              class="empty-transcript-animation"
              src="/animations/Boy%20And%20Girl%20Chat%20on%20Social%20Media.json"
              width="min(82%, 320px)"
              height="250px"
              :autoplay="false"
              :loop="false"
              fallback-icon="forum"
            />
          </button>
          <p class="text-[13px] font-medium text-[#8e8e93]">
            {{ recordingMode === 'meeting' ? '화자 분리된 회의 스크립트가 여기에 표시됩니다.' : '전사된 데이터가 없습니다.' }}
          </p>
        </div>
      </template>
      <template v-else>
        <div 
          v-for="(t, idx) in filteredTranscriptions" 
          :key="idx" 
          :ref="(el) => setSearchResultRef(el, idx)"
          class="transcription-row flex flex-col gap-1.5 mt-2 transcription-item-enter"
          :class="{ 'is-active-search-result': hasSearchTerm && idx === activeSearchIndex }"
          :style="{ animationDelay: `${idx * 0.06}s` }"
        >
          <span class="transcription-time text-[11px] font-bold text-[#aeaeb2] px-1.5">{{ getTranscriptionTime(t) }}</span>
          <div class="message-bubble voice-message-bubble px-3.5 py-3 text-[15px] leading-[1.6]" :class="{ 'is-content': variant === 'content', 'is-meeting': shouldShowSpeaker(t) }">
            <template v-if="t.segments && t.segments.length">
              <span
                v-for="(seg, sIdx) in t.segments"
                :key="seg.id ?? sIdx"
                class="segment-wrap"
                :class="{ 'segment-pending': seg.status === 'pending', 'segment-confirmed': seg.status === 'confirmed' }"
              >
                <span
                  v-for="(word, wIdx) in seg.text.split(' ')"
                  :key="wIdx"
                  class="clickable-word"
                  :class="{ 'search-highlighted-word': isSearchHighlightedWord(word) }"
                  @click="(e) => handleWordClick(e, word, seg.text)"
                >{{ word }}&nbsp;</span>
              </span>
            </template>
            <template v-else>
              <span
                v-for="(word, wIdx) in t.text.split(' ')"
                :key="wIdx"
                class="clickable-word"
                :class="{ 'search-highlighted-word': isSearchHighlightedWord(word) }"
                @click="(e) => handleWordClick(e, word, t.text)"
              >{{ word }}&nbsp;</span>
            </template>
          </div>
        </div>
      </template>
    </div>
  </div>

  <Teleport to="body">
    <transition name="transcript-search-popover">
      <section
        v-if="isSearchOpen"
        class="transcript-search-popover"
        :style="searchPopoverStyle"
        role="search"
        aria-label="전사 내용 검색"
      >
        <div class="transcript-search-input-row">
          <span class="material-symbols-outlined transcript-search-leading-icon">search</span>
          <input
            ref="searchInput"
            v-model="transSearch"
            class="transcript-search-input"
            placeholder="검색어를 입력해주세요"
            type="text"
            @keydown.esc="closeSearchPopup"
            @keydown.enter.prevent="moveSearchResult($event.shiftKey ? -1 : 1)"
          />
          <button
            type="button"
            class="transcript-search-nav"
            aria-label="이전 검색 결과"
            :disabled="!hasSearchResults"
            @click="moveSearchResult(-1)"
          >
            <span class="material-symbols-outlined">keyboard_arrow_up</span>
          </button>
          <button
            type="button"
            class="transcript-search-nav"
            aria-label="다음 검색 결과"
            :disabled="!hasSearchResults"
            @click="moveSearchResult(1)"
          >
            <span class="material-symbols-outlined">keyboard_arrow_down</span>
          </button>
          <button type="button" class="transcript-search-close" aria-label="검색 닫기" @click="clearSearch">
            <span class="material-symbols-outlined">close</span>
          </button>
        </div>
        <div class="transcript-search-status">
          {{ searchStatusText }}
        </div>
      </section>
    </transition>
  </Teleport>
</template>

<style scoped>
.popover-enter-active,
.popover-leave-active {
  transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.popover-enter-from,
.popover-leave-to {
  opacity: 0;
  transform: translateY(10px) scale(0.95);
}
.popover-enter-to,
.popover-leave-from {
  opacity: 1;
  transform: translateY(0) scale(1);
}

/* segment-wrap 인라인 표시 */
.segment-wrap {
  display: inline;
}

/* 전사 텍스트 상태 애니메이션 */
.segment-pending {
  opacity: 0.55;
  filter: blur(0.3px);
  transition: opacity 0.5s ease, filter 0.5s ease;
}
.segment-pending .clickable-word {
  color: #aeaeb2;
  transition: color 0.5s ease;
}
.segment-confirmed {
  opacity: 1;
  filter: none;
  animation: confirmSegment 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.segment-confirmed .clickable-word {
  color: #1d1d1f;
  animation: confirmWord 0.5s ease forwards;
}

.voice-message-bubble .clickable-word:hover {
  background-color: rgba(238, 240, 255, 0.92);
  color: #2f80ed;
}

.voice-message-bubble .clickable-word.search-highlighted-word {
  color: #111827;
  background: rgba(254, 240, 138, 0.72);
  box-shadow: inset 0 -0.32em 0 rgba(250, 204, 21, 0.36);
}

.voice-message-bubble .clickable-word.search-highlighted-word:hover {
  color: #111827;
  background: rgba(254, 240, 138, 0.9);
}

.voice-message-bubble .clickable-word:active {
  background-color: rgba(226, 224, 232, 0.86);
}

.diarization-preparing-state {
  min-height: 220px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 28px 18px;
  color: #475569;
  text-align: center;
}

.diarization-preparing-icon {
  width: 46px;
  height: 46px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #2563eb;
  background: #eef4ff;
  border: 1px solid #dbe7ff;
  animation: diarizationPulse 1.4s ease-in-out infinite;
}

.diarization-preparing-icon .material-symbols-outlined {
  font-size: 25px;
}

.diarization-preparing-state strong {
  color: #1f2937;
  font-size: 14px;
  font-weight: 950;
}

.diarization-preparing-state p {
  margin: 0;
  color: #94a3b8;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.5;
  word-break: keep-all;
}

@keyframes diarizationPulse {
  0%, 100% { transform: scale(1); opacity: 0.82; }
  50% { transform: scale(1.05); opacity: 1; }
}

@keyframes confirmSegment {
  0%   { opacity: 0.55; transform: translateY(2px); }
  60%  { opacity: 1;    transform: translateY(-1px); }
  100% { opacity: 1;    transform: translateY(0); }
}
@keyframes confirmWord {
  0%   { color: #aeaeb2; }
  40%  { color: #3b82f6; }
  100% { color: #1d1d1f; }
}

.voice-message-bubble {
  position: relative;
  width: fit-content;
  max-width: min(calc(100% - 18px), 440px);
  background:
    linear-gradient(160deg, rgba(255, 255, 255, 0.96), rgba(248, 246, 250, 0.9));
  border: 1px solid rgba(226, 224, 232, 0.94);
  border-radius: 18px;
  box-shadow:
    0 14px 30px rgba(48, 42, 58, 0.06),
    inset 0 1px 0 rgba(255, 255, 255, 0.96);
  overflow: hidden;
  color: #15161a;
}

.transcript-list {
  padding-right: 14px;
}

.empty-transcript-state {
  gap: 12px;
  color: #8e8e93;
  text-align: center;
}

.empty-transcript-animation-trigger {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: min(82%, 320px);
  height: 250px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}

.empty-transcript-animation-trigger:focus-visible {
  outline: 2px solid rgba(47, 128, 237, 0.54);
  outline-offset: 8px;
  border-radius: 14px;
}

.empty-transcript-animation {
  opacity: 0.9;
  user-select: none;
  pointer-events: none;
}

.speaker-avatar {
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  padding: 3px;
  overflow: hidden;
  border-radius: 999px;
  border: 1.5px solid #1d1d1f;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.85),
    0 1px 0 rgba(255, 255, 255, 0.9);
}

.speaker-avatar img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center bottom;
  border-radius: 999px;
  display: block;
}

.speaker-avatar-blue {
  background: #dbeafe;
}

.speaker-avatar-amber {
  background: #fff1d6;
}

.speaker-avatar-rose {
  background: #ffe4ea;
}

.speaker-avatar-green {
  background: #dcfce7;
}

.voice-message-bubble.is-meeting {
  background:
    linear-gradient(160deg, rgba(255, 255, 255, 0.97), rgba(238, 240, 255, 0.86));
  border-color: rgba(220, 216, 227, 0.98);
}

.is-active-search-result .voice-message-bubble {
  border-color: rgba(47, 128, 237, 0.72);
  box-shadow: 0 0 0 3px rgba(47, 128, 237, 0.12), 0 14px 30px rgba(47, 128, 237, 0.08);
}

.transcript-panel-content .transcript-list {
  gap: 34px;
  padding: 18px 18px 96px 0;
}

.transcript-panel-content .transcription-row {
  margin-top: 0;
  gap: 10px;
}

.transcript-panel-content .transcription-time {
  padding: 0;
  color: #9aa1ad;
  font-size: 13px;
  font-weight: 850;
  line-height: 1.2;
}

.transcript-panel-content .voice-message-bubble.is-content {
  width: 100%;
  max-width: 100%;
  padding: 0 !important;
  overflow: visible;
  color: #444b55;
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  font-size: 16px;
  font-weight: 500;
  line-height: 1.85;
  letter-spacing: 0;
}

.transcript-panel-content .voice-message-bubble.is-content .clickable-word {
  font-weight: 500;
}

.transcript-panel-content .voice-message-bubble.is-content::before {
  display: none;
}

.transcript-panel-content .voice-message-bubble.is-content.is-meeting {
  background: transparent;
  border-color: transparent;
}

.transcript-panel-content .voice-message-bubble.is-content .clickable-word:hover {
  color: #111827;
  background: rgba(226, 232, 240, 0.78);
}

.transcript-panel-content .voice-message-bubble.is-content .clickable-word.search-highlighted-word {
  color: #2f3742;
  background: #d7e2ec;
  box-shadow: none;
}

.transcript-panel-content .is-active-search-result .voice-message-bubble {
  border-color: transparent;
  box-shadow: none;
}

.transcript-panel-content .empty-transcript-animation {
  opacity: 0.86;
}

.transcript-panel-content .empty-transcript-animation-trigger {
  width: min(54%, 340px);
}

.transcript-toolbar {
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  min-height: 34px;
  margin-bottom: 2px;
  border-bottom: 1px solid rgba(226, 224, 232, 0.78);
}

.transcript-panel-content .transcript-toolbar {
  margin-top: 0;
  border-bottom: 0;
}

.transcript-toolbar.has-folder-toggle {
  grid-template-columns: minmax(0, 1fr) auto;
}

.transcript-toolbar-title {
  position: relative;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-self: start;
  color: #1d1d1f;
  font-size: 13px;
  font-weight: 950;
  letter-spacing: 0;
}

.transcript-toolbar-title::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: -1px;
  height: 2px;
  border-radius: 999px;
  background: #1d1d1f;
}

.transcript-source-toggle {
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  justify-self: end;
  gap: 6px;
  padding: 0 11px;
  border-radius: 999px;
  color: #5f6472;
  background: rgba(248, 249, 252, 0.96);
  border: 1px solid rgba(218, 223, 232, 0.98);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.92);
  font-size: 11.5px;
  font-weight: 900;
  letter-spacing: -0.01em;
  white-space: nowrap;
  transition: color 0.18s ease, border-color 0.18s ease, background-color 0.18s ease, transform 0.18s ease;
}

.transcript-source-toggle:hover,
.transcript-source-toggle.is-open {
  color: #15161a;
  border-color: rgba(156, 163, 175, 0.5);
  background: #ffffff;
}

.transcript-source-toggle:active {
  transform: scale(0.96);
}

.transcript-source-toggle-arrow {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 10px;
  color: #8e95a3;
  font-size: 14px;
  font-weight: 950;
  line-height: 1;
}

.transcript-toolbar-actions {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  justify-self: end;
  gap: 6px;
}

.transcript-toolbar-search {
  justify-self: end;
}

.transcript-search-anchor {
  position: relative;
  z-index: 12;
  min-height: 34px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin-bottom: 10px;
}

.transcript-search-trigger {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  color: #5f6472;
  background: transparent;
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.16s ease;
}

.transcript-search-trigger:hover,
.transcript-search-trigger.is-active {
  color: #15161a;
  background: rgba(255, 255, 255, 0.72);
}

.transcript-search-trigger:active {
  transform: scale(0.94);
}

.transcript-search-trigger .material-symbols-outlined {
  font-size: 23px;
}

.transcript-search-popover {
  position: fixed;
  z-index: 1000;
  overflow: hidden;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(226, 224, 232, 0.92);
  box-shadow: 0 14px 30px rgba(48, 42, 58, 0.1);
  backdrop-filter: blur(18px) saturate(150%);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
}

.transcript-search-input-row {
  min-height: 40px;
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr) 22px 22px 22px;
  align-items: center;
  gap: 4px;
  padding: 0 10px;
  border-bottom: 1px solid rgba(226, 224, 232, 0.78);
}

.transcript-search-leading-icon {
  color: #15161a;
  font-size: 20px;
}

.transcript-search-input {
  min-width: 0;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: #15161a;
  font-size: 12.5px;
  font-weight: 750;
  outline: none;
  box-shadow: none;
}

.transcript-search-input:focus {
  outline: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
  --tw-ring-color: transparent;
  --tw-ring-shadow: 0 0 #0000;
}

.transcript-search-input::placeholder {
  color: #9ca3af;
  font-weight: 800;
}

.transcript-search-nav,
.transcript-search-close {
  width: 22px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #8e8e93;
  background: transparent;
}

.transcript-search-nav:disabled {
  opacity: 0.7;
  cursor: default;
}

.transcript-search-nav:not(:disabled):hover {
  color: #15161a;
  background: rgba(229, 226, 235, 0.72);
}

.transcript-search-close:hover {
  color: #15161a;
  background: rgba(229, 226, 235, 0.72);
}

.transcript-search-nav .material-symbols-outlined,
.transcript-search-close .material-symbols-outlined {
  font-size: 18px;
}

.transcript-search-status {
  padding: 8px 12px 10px;
  color: #9ca3af;
  font-size: 11.5px;
  font-weight: 850;
  line-height: 1.45;
}

.transcript-search-popover-enter-active,
.transcript-search-popover-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.transcript-search-popover-enter-from,
.transcript-search-popover-leave-to {
  opacity: 0;
  transform: translateX(-4px) scale(0.98);
}

.voice-message-bubble::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background:
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.58), transparent 36%);
  pointer-events: none;
}

</style>
