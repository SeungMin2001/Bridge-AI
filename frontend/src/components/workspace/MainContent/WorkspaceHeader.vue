<!-- 워크스페이스 중앙 영역 상단에서 녹음 제어와 자료 업로드, 사이드바 토글을 담당하는 헤더입니다. -->
<script setup>
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps({
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingTimeText: String,
  recordingAudioLevel: { type: Number, default: 0 },
  tabs: { type: Array, default: () => [] },
  activeTab: { type: String, default: 'materials' },
  showClosePreview: Boolean,
  showPdfSearch: Boolean,
  pdfSearchTotal: { type: Number, default: 0 },
  pdfSearchActiveIndex: { type: Number, default: 0 },
  hasWordInsight: Boolean,
  wordInsightVisible: Boolean,
  embedded: { type: Boolean, default: false },
  folderDrawerOpen: { type: Boolean, default: false }
})

const emit = defineEmits([
  'start-recording',
  'pause-recording',
  'resume-recording',
  'stop-recording',
  'tab-change',
  'main-sidebar-toggle',
  'right-sidebar-toggle',
  'material-selected',
  'word-insight-click',
  'close-preview-material',
  'toggle-folder-drawer',
  'pdf-search-change',
  'pdf-search-next',
  'pdf-search-prev',
  'pdf-search-clear'
])

const fileInputRef = ref(null)
const pdfSearchInputRef = ref(null)
const isPdfSearchOpen = ref(false)
const pdfSearchDraft = ref('')

const hasPdfSearchTerm = computed(() => pdfSearchDraft.value.trim().length > 0)
const pdfSearchStatusText = computed(() => {
  if (!hasPdfSearchTerm.value) return '검색어를 입력해주세요.'
  if (!props.pdfSearchTotal) return `"${pdfSearchDraft.value}"에 대한 검색 결과가 없습니다`
  return `검색 결과 ${props.pdfSearchTotal}개 · ${props.pdfSearchActiveIndex + 1}/${props.pdfSearchTotal}`
})

const triggerMaterialPicker = () => {
  fileInputRef.value?.click()
}

const openPdfSearch = async () => {
  isPdfSearchOpen.value = true
  await nextTick()
  pdfSearchInputRef.value?.focus()
}

const closePdfSearch = () => {
  isPdfSearchOpen.value = false
}

const clearPdfSearch = () => {
  pdfSearchDraft.value = ''
  isPdfSearchOpen.value = false
  emit('pdf-search-clear')
}

const handleMaterialInputChange = (event) => {
  const [file] = Array.from(event.target.files || [])
  if (file) {
    emit('material-selected', file)
  }
  event.target.value = ''
}

watch(pdfSearchDraft, (value) => {
  emit('pdf-search-change', value)
})

watch(
  () => props.showPdfSearch,
  (visible) => {
    if (visible) return
    pdfSearchDraft.value = ''
    isPdfSearchOpen.value = false
    emit('pdf-search-clear')
  }
)
</script>

<template>
  <header class="workspace-embedded-header flex items-center px-6 shrink-0">
    <div class="flex items-center gap-2.5 shrink-0 min-w-0">
      <button
        v-if="!embedded"
        class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] shrink-0"
        title="사이드바 토글"
        @click="emit('main-sidebar-toggle')"
      >
        <span class="material-symbols-outlined text-[20px]">side_navigation</span>
      </button>

      <nav class="workspace-header-tabs" aria-label="워크스페이스 탭">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          type="button"
          class="workspace-header-tab"
          :class="{ 'is-active': activeTab === tab.key }"
          @click="emit('tab-change', tab.key)"
        >
          {{ tab.label }}
        </button>
      </nav>

    </div>

    <div class="workspace-header-actions ml-auto flex items-center gap-1.5 shrink-0 pl-3 self-center">
      <button
        v-if="showPdfSearch"
        class="preview-search-header-btn"
        :class="{ 'is-active': isPdfSearchOpen || hasPdfSearchTerm }"
        title="PDF 내용 검색"
        aria-label="PDF 내용 검색"
        @click="isPdfSearchOpen ? closePdfSearch() : openPdfSearch()"
      >
        <span class="material-symbols-outlined text-[19px]">search</span>
      </button>

      <transition name="pdf-header-search">
        <section
          v-if="showPdfSearch && isPdfSearchOpen"
          class="pdf-header-search-popover"
          role="search"
          aria-label="PDF 내용 검색"
        >
          <div class="pdf-header-search-input-row">
            <span class="material-symbols-outlined pdf-header-search-leading-icon">search</span>
            <input
              ref="pdfSearchInputRef"
              v-model="pdfSearchDraft"
              class="pdf-header-search-input"
              placeholder="검색어를 입력해주세요"
              type="text"
              @keydown.esc="closePdfSearch"
              @keydown.enter.prevent="$event.shiftKey ? emit('pdf-search-prev') : emit('pdf-search-next')"
            />
            <button
              type="button"
              class="pdf-header-search-nav"
              aria-label="이전 검색 결과"
              :disabled="!pdfSearchTotal"
              @click="emit('pdf-search-prev')"
            >
              <span class="material-symbols-outlined">keyboard_arrow_up</span>
            </button>
            <button
              type="button"
              class="pdf-header-search-nav"
              aria-label="다음 검색 결과"
              :disabled="!pdfSearchTotal"
              @click="emit('pdf-search-next')"
            >
              <span class="material-symbols-outlined">keyboard_arrow_down</span>
            </button>
            <button
              type="button"
              class="pdf-header-search-close"
              aria-label="검색 닫기"
              @click="clearPdfSearch"
            >
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>
          <div class="pdf-header-search-status">
            {{ pdfSearchStatusText }}
          </div>
        </section>
      </transition>

      <button
        v-if="showClosePreview"
        class="preview-close-header-btn"
        title="자료 닫기"
        @click="emit('close-preview-material')"
      >
        <span class="material-symbols-outlined text-[18px]">close</span>
        <span>닫기</span>
      </button>

      <button
        v-if="hasWordInsight"
        class="word-insight-toggle-btn"
        :aria-label="wordInsightVisible ? 'AI 결과 카드 접기' : 'AI 결과 카드 다시 보기'"
        :title="wordInsightVisible ? 'AI 결과 카드 접기' : 'AI 결과 카드 다시 보기'"
        @click="emit('word-insight-click')"
      >
        <span class="material-symbols-outlined text-[20px]">
          {{ wordInsightVisible ? 'unfold_less' : 'unfold_more' }}
        </span>
      </button>

      <input
        ref="fileInputRef"
        type="file"
        accept=".pdf,.ppt,.pptx,application/pdf,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"
        class="hidden"
        @change="handleMaterialInputChange"
      />

      <button
        v-if="!embedded"
        class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]"
        title="우측 사이드바 토글"
        @click="emit('right-sidebar-toggle')"
      >
        <span class="material-symbols-outlined text-[20px] scale-x-[-1]">side_navigation</span>
      </button>
    </div>
  </header>

  <div class="workspace-embedded-divider shrink-0"></div>
</template>

<style scoped>
.workspace-embedded-header {
  position: relative;
  background: #ffffff;
  min-height: 48px;
}

.workspace-embedded-divider {
  height: 1px;
  margin: 0 24px;
  background: rgba(0, 0, 0, 0.06);
}

.workspace-header-tabs {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 26px;
  height: 38px;
  margin-left: 8px;
  z-index: 1;
}

.workspace-header-tab {
  position: relative;
  height: 38px;
  border: 0;
  background: transparent;
  color: #8e8e93;
  font-size: 15px;
  font-weight: 900;
  letter-spacing: -0.02em;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.18s ease;
}

.workspace-header-tab:hover,
.workspace-header-tab.is-active {
  color: #1d1d1f;
}

.workspace-header-tab.is-active::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: -6px;
  height: 3px;
  border-radius: 999px;
  background: #1d1d1f;
}

.workspace-header-actions {
  position: relative;
}

.preview-search-header-btn {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid rgba(226, 232, 240, 0.95);
  background: rgba(255, 255, 255, 0.92);
  color: #6b7280;
  box-shadow: 0 10px 22px rgba(148, 163, 184, 0.08);
  transition: transform 0.2s ease, border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease;
}

.preview-search-header-btn:hover,
.preview-search-header-btn.is-active {
  color: #111827;
  border-color: rgba(148, 163, 184, 0.42);
  background: #ffffff;
}

.preview-search-header-btn:active {
  transform: scale(0.98);
}

.pdf-header-search-popover {
  position: absolute;
  top: calc(100% + 9px);
  right: 0;
  z-index: 60;
  width: 344px;
  overflow: hidden;
  border-radius: 17px;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(226, 224, 232, 0.96);
  box-shadow: 0 18px 38px rgba(48, 42, 58, 0.14);
  backdrop-filter: blur(18px) saturate(150%);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  transform-origin: top right;
}

.pdf-header-search-input-row {
  min-height: 44px;
  display: grid;
  grid-template-columns: 23px minmax(0, 1fr) 26px 26px 26px;
  align-items: center;
  gap: 7px;
  padding: 0 14px;
  border-bottom: 1px solid rgba(226, 224, 232, 0.78);
}

.pdf-header-search-leading-icon {
  color: #111827;
  font-size: 23px;
}

.pdf-header-search-input {
  min-width: 0;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: #17171c;
  font-size: 13px;
  font-weight: 850;
  outline: none;
  box-shadow: none;
}

.pdf-header-search-input:focus {
  outline: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
  --tw-ring-color: transparent;
  --tw-ring-shadow: 0 0 #0000;
}

.pdf-header-search-input::placeholder {
  color: #a1a7b3;
  font-weight: 850;
}

.pdf-header-search-nav,
.pdf-header-search-close {
  width: 26px;
  height: 26px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 9px;
  color: #9aa1af;
  background: transparent;
  transition: background-color 0.18s ease, color 0.18s ease;
}

.pdf-header-search-nav:not(:disabled):hover,
.pdf-header-search-close:hover {
  color: #17171c;
  background: #f2f3f7;
}

.pdf-header-search-nav:disabled {
  opacity: 0.5;
  cursor: default;
}

.pdf-header-search-nav .material-symbols-outlined,
.pdf-header-search-close .material-symbols-outlined {
  font-size: 20px;
}

.pdf-header-search-status {
  padding: 11px 16px 13px;
  color: #9aa1af;
  font-size: 12.5px;
  font-weight: 850;
  line-height: 1.45;
}

.pdf-header-search-enter-active,
.pdf-header-search-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.pdf-header-search-enter-from,
.pdf-header-search-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.985);
}

.preview-close-header-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 36px;
  padding: 0 13px;
  border-radius: 999px;
  border: 1px solid rgba(226, 232, 240, 0.95);
  background: rgba(255, 255, 255, 0.92);
  color: #374151;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: -0.01em;
  box-shadow: 0 10px 22px rgba(148, 163, 184, 0.08);
  transition: transform 0.2s ease, border-color 0.2s ease, background-color 0.2s ease;
}

.preview-close-header-btn:hover {
  border-color: rgba(148, 163, 184, 0.42);
  background: #ffffff;
}

.preview-close-header-btn:active {
  transform: scale(0.98);
}

.folder-drawer-header-btn {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  color: #8e95a3;
  background: transparent;
  transition: color 0.18s ease, background-color 0.18s ease, transform 0.18s ease;
}

.folder-drawer-header-btn:hover,
.folder-drawer-header-btn.is-open {
  color: #5f6472;
  background: rgba(245, 246, 250, 0.92);
}

.folder-drawer-header-btn:active {
  transform: scale(0.96);
}

.folder-drawer-header-glyph {
  position: relative;
  width: 28px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.folder-drawer-header-glyph::before {
  content: '';
  position: absolute;
  left: 1px;
  top: 2px;
  width: 18px;
  height: 20px;
  border: 2px solid currentColor;
  border-radius: 6px;
  opacity: 0.92;
}

.folder-drawer-header-panel {
  position: absolute;
  right: 1px;
  top: 2px;
  width: 18px;
  height: 20px;
  border: 2px solid currentColor;
  border-radius: 6px;
  background: #ffffff;
}

.folder-drawer-header-chevron {
  position: relative;
  z-index: 1;
  margin-left: 6px;
  font-size: 18px;
  font-weight: 700;
}

.word-insight-toggle-btn {
  width: 36px;
  height: 36px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 0;
  color: #8e95a3;
  background: transparent;
  transition: color 0.18s ease, background-color 0.18s ease, transform 0.18s ease;
}

.word-insight-toggle-btn:hover {
  color: #15161a;
  background: rgba(245, 246, 250, 0.92);
}

.word-insight-toggle-btn:active {
  transform: scale(0.96);
}

.recording-control-bar {
  position: relative;
  min-height: 36px;
  display: flex;
  align-items: center;
}

.word-insight-btn {
  transition: color 0.2s ease, transform 0.2s ease;
}

.word-insight-btn:hover {
  color: #1d1d1f;
}

.word-insight-btn:active {
  transform: scale(0.98);
}

.recording-control-inner {
  display: inline-flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
  white-space: nowrap;
}

.recording-time-text {
  font-size: 14px;
  font-weight: 800;
  color: #1d1d1f;
  letter-spacing: 0.02em;
}

.recording-icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 999px;
  background: #e5e5ea;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.recording-icon-btn:hover {
  background: #dbdbe2;
}

.recording-icon-btn:active {
  transform: scale(0.98);
}

.recording-icon-btn-sm {
  width: 36px;
  height: 36px;
}

.recording-icon-btn.is-paused {
  background: #fff0f1;
}

.recording-icon-btn.is-paused:hover {
  background: #ffe4e7;
}

.recording-pause-bars {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.recording-pause-bars span {
  display: block;
  width: 5px;
  height: 18px;
  border-radius: 999px;
  background: #5f6472;
}

.recording-play-triangle {
  width: 0;
  height: 0;
  margin-left: 2px;
  border-top: 9px solid transparent;
  border-bottom: 9px solid transparent;
  border-left: 14px solid #ef4444;
}

.recording-primary-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 88px;
  height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  background: #111111;
  color: #ffffff;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: -0.01em;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.recording-primary-btn:hover {
  background: #1f1f1f;
}

.recording-primary-btn:active {
  transform: scale(0.98);
}

.recording-control-enter-active,
.recording-control-leave-active {
  transition: opacity 0.26s ease, transform 0.32s cubic-bezier(0.22, 1, 0.36, 1);
}

.recording-control-move {
  transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1);
}

.recording-control-enter-from {
  opacity: 0;
  transform: translateY(8px) scale(0.96);
}

.recording-control-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.96);
}

.recording-control-leave-active {
  position: absolute;
}

.recording-voice-dots.is-paused .recording-voice-dot {
  animation-play-state: paused;
}

.recording-voice-dots.is-paused .recording-voice-dot {
  opacity: 0.82;
  transform: scaleY(0.85);
}

.recording-voice-dot {
  animation: none;
  transition: transform 90ms ease-out, opacity 90ms ease-out;
}
</style>
