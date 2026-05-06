<!-- 음성 녹음, 실시간 전사, AI 분석 및 교차 참조가 이루어지는 작업실 페이지 컴포넌트입니다. -->
<script setup>
import { computed, ref } from 'vue'
import LeftSidebar from '../../components/workspace/LeftSidebar.vue'
import MainContent from '../../components/workspace/MainContent.vue'
import RightSidebar from '../../components/workspace/RightSidebar.vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import { useChat } from '../../composables/useChat'

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  isRecording: { type: Boolean, default: false },
  isRecordingPaused: { type: Boolean, default: false },
  recordingMode: { type: String, default: 'lecture' },
  recordingTimeText: { type: String, default: '00:00:00' },
  activeFileName: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  activeFileType: { type: String, default: 'lecture' },
  currentPreviewMaterial: { type: Object, default: null },
  isRightSidebarVisible: { type: Boolean, default: true },
  scheduleExtractionNotice: { type: Object, default: null },
  summaryState: { type: Object, default: () => ({}) },
  summaryNotes: { type: Array, default: () => [] },
  aiInput: { type: String, default: '' }
})

const emit = defineEmits([
  'navigateHome',
  'navigate',
  'fileSelect',
  'update:fileTree',
  'update:favorites',
  'update:aiInput',
  'dismissScheduleNotice',
  'startRecording',
  'pauseRecording',
  'resumeRecording',
  'stopRecording',
  'rightSidebarToggle',
  'addToNote',
  'askAi',
  'uploadLectureMaterials',
  'closePreviewMaterial',
  'openStoredMaterial',
  'openRecording'
])

const isLeftSidebarCollapsed = ref(false)
const { showCitePopover, currentCite, citePopoverPos, closeCitePopover, clearHistory } = useChat()
const citationSourceRequest = ref(null)

const scheduleNoticeItems = computed(() => props.scheduleExtractionNotice?.items || [])
const visibleScheduleNoticeItems = computed(() => scheduleNoticeItems.value.slice(0, 3))
const hiddenScheduleNoticeCount = computed(() => Math.max(scheduleNoticeItems.value.length - 3, 0))
const scheduleNoticeTitle = computed(() => (
  scheduleNoticeItems.value.length > 1
    ? `새 일정 ${scheduleNoticeItems.value.length}개가 추가되었습니다`
    : '새 일정이 추가되었습니다'
))

function formatScheduleNoticeDate(value = '') {
  if (!value) return ''

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value

  const year = parsed.getFullYear()
  const month = parsed.getMonth() + 1
  const day = parsed.getDate()
  let hour = parsed.getHours()
  const minute = String(parsed.getMinutes()).padStart(2, '0')
  const meridiem = hour < 12 ? '오전' : '오후'
  hour %= 12
  if (hour === 0) hour = 12

  return `${year}년 ${month}월 ${day}일 ${meridiem} ${hour}:${minute}`
}

function closeScheduleNotice() {
  emit('dismissScheduleNotice')
}

function goSchedulePageFromNotice() {
  emit('dismissScheduleNotice')
  emit('navigate', 'schedule')
}

// 팝오버 내 버튼 액션
function askAboutCite(cite) {
  if (!cite) return
  closeCitePopover()
  if (!props.isRightSidebarVisible) {
    emit('rightSidebarToggle')
  }
  const shortened = cite.text.length > 15 ? cite.text.slice(0, 15) + '...' : cite.text
  emit('update:aiInput', `"${shortened}"에 대해 더 자세히 알려줘 `)
}

function noteAddDummy() {
  alert('노트에 추가되었습니다. (데모)')
  closeCitePopover()
}

function findNodeById(nodes = [], id = '') {
  for (const node of nodes) {
    if (node?.id === id) return node
    if (Array.isArray(node?.children)) {
      const found = findNodeById(node.children, id)
      if (found) return found
    }
  }
  return null
}

function openCitationSource(cite) {
  const sessionId = cite?.session_id
  if (!sessionId) return

  const node = findNodeById(props.fileTree, sessionId)
  if (!node) return

  emit('fileSelect', sessionId, node)
  isLeftSidebarCollapsed.value = false
  citationSourceRequest.value = {
    id: `${sessionId}-${cite?.transcript_id || cite?.citation || Date.now()}`,
    cite,
    node
  }
  clearHistory()
  emit('update:aiInput', '')
  closeCitePopover()
}

function escapeHtml(value = '') {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

const highlightedTranscript = computed(() => {
  const cite = currentCite.value
  if (!cite) return ''

  // 전체 전사가 있으면 그것을 쓰고, 없으면 기존 text 사용
  const fullText = String(cite.full_transcript || cite.text || '')
  // 하이라이팅 대상
  const target = String(cite.text || '').trim()

  if (target && fullText.includes(target)) {
    const highlightedTarget = `<mark class="cite-highlighted-script">${escapeHtml(target)}</mark>`
    return fullText.split(target).map((part) => escapeHtml(part)).join(highlightedTarget)
  }

  return escapeHtml(fullText)
})
</script>

<template>
  <div 
    class="p-[12px] flex relative h-full w-full bg-transparent text-[#1e293b] overflow-hidden transition-all duration-400"
    :class="[
      { 'gap-[12px]': !isLeftSidebarCollapsed || isRightSidebarVisible }
    ]"
  >
    <InfiniteGrid class="absolute inset-0 z-0" />

    <transition name="schedule-notice-fade">
      <section
        v-if="scheduleNoticeItems.length"
        class="workspace-schedule-notice"
        role="status"
        aria-live="polite"
      >
        <div class="workspace-schedule-notice-top">
          <div class="workspace-schedule-notice-icon">
            <span class="material-symbols-outlined">event_available</span>
          </div>
          <div class="workspace-schedule-notice-heading">
            <span>AI 일정 감지</span>
            <strong>{{ scheduleNoticeTitle }}</strong>
          </div>
          <button
            type="button"
            class="workspace-schedule-notice-close"
            aria-label="일정 알림 닫기"
            @click="closeScheduleNotice"
          >
            <span class="material-symbols-outlined">close</span>
          </button>
        </div>

        <div class="workspace-schedule-notice-list">
          <article
            v-for="item in visibleScheduleNoticeItems"
            :key="item.id"
            class="workspace-schedule-notice-item"
          >
            <strong>{{ item.title }}</strong>
            <span>{{ formatScheduleNoticeDate(item.dueDate) }}</span>
          </article>
          <div v-if="hiddenScheduleNoticeCount" class="workspace-schedule-notice-more">
            외 {{ hiddenScheduleNoticeCount }}개 일정
          </div>
        </div>

        <div class="workspace-schedule-notice-actions">
          <button type="button" class="workspace-schedule-notice-secondary" @click="closeScheduleNotice">
            확인
          </button>
          <button type="button" class="workspace-schedule-notice-primary" @click="goSchedulePageFromNotice">
            일정관리로 이동
          </button>
        </div>
      </section>
    </transition>

    <LeftSidebar
      class="relative z-10"
      :isCollapsed="isLeftSidebarCollapsed"
      :recordingMode="recordingMode"
      :transcriptions="transcriptions"
      :fileTree="fileTree"
      :favorites="favorites"
      :activeFileId="activeFileId"
      :citationSourceRequest="citationSourceRequest"
      @toggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
      @navigateHome="emit('navigateHome')"
      @fileSelect="(id, node) => emit('fileSelect', id, node)"
      @update:fileTree="emit('update:fileTree', $event)"
      @update:favorites="emit('update:favorites', $event)"
      @addToNote="(text, source) => emit('addToNote', text, source)"
      @askAi="(word) => emit('askAi', word)"
      @openStoredMaterial="emit('openStoredMaterial', $event)"
      @openRecording="emit('openRecording', $event)"
    />
    
    <MainContent
      class="relative z-10"
      :isRecording="isRecording"
      :isRecordingPaused="isRecordingPaused"
      :recordingMode="recordingMode"
      :recordingTimeText="recordingTimeText"
      :activeFileName="activeFileName"
      :activeFileId="activeFileId"
      :activeFileType="activeFileType"
      :transcriptions="transcriptions"
      :currentPreviewMaterial="currentPreviewMaterial"
      :summaryState="summaryState"
      :summaryNotes="summaryNotes"
      @startRecording="emit('startRecording')"
      @pauseRecording="emit('pauseRecording')"
      @resumeRecording="emit('resumeRecording')"
      @stopRecording="emit('stopRecording')"
      @mainSidebarToggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
      @rightSidebarToggle="emit('rightSidebarToggle')"
      @askAi="(word) => emit('askAi', word)"
      @addToNote="(text, source) => emit('addToNote', text, source)"
      @uploadLectureMaterials="emit('uploadLectureMaterials', $event)"
      @closePreviewMaterial="emit('closePreviewMaterial')"
    />
    
    <RightSidebar 
      class="relative z-10"
      :visible="isRightSidebarVisible" 
      :aiInput="aiInput"
      :activeFileId="activeFileId"
      @update:aiInput="emit('update:aiInput', $event)"
    />
  </div>

  <!-- ═══ 부유형 팝오버 (Workspace 수준 관리) ═══ -->
  <Teleport to="body">
    <transition name="popover-fade">
      <div v-if="showCitePopover" class="cite-popover-overlay" @click.self="closeCitePopover">
        <div 
          class="cite-popover"
          :style="{ left: citePopoverPos.x + 'px', top: citePopoverPos.y + 'px' }"
        >
          <!-- 헤더 -->
          <div class="flex items-center justify-between mb-5 px-1">
            <div class="flex items-center gap-3">
              <div class="cite-popover-badge">
                <span class="material-symbols-outlined text-[15px]">fact_check</span>
              </div>
              <span class="font-bold text-[#1c1c1e] text-[18px] tracking-tight">근거 정보</span>
            </div>
            <button class="cite-popover-close-btn" @click="closeCitePopover">
              <span class="material-symbols-outlined text-[20px]">close</span>
            </button>
          </div>

          <!-- 본문 (스크롤 영역) -->
          <div class="flex-1 overflow-y-auto mb-6 px-1 custom-scrollbar" style="max-height: 400px;">
            <div 
              class="cite-transcript-body whitespace-pre-wrap break-keep"
              v-html="highlightedTranscript"
            >
            </div>
          </div>

          <!-- 구분선 -->
          <div class="cite-popover-divider"></div>

          <!-- 출처 정보 -->
          <div class="cite-source-wrap shrink-0">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-[15px] text-[#8e8e93]">link</span>
              <span class="font-bold text-[#8e8e93] text-[11px] uppercase tracking-wider">Source</span>
              <button
                type="button"
                class="cite-source-title"
                @click="openCitationSource(currentCite)"
                :title="currentCite?.file_title || currentCite?.session_title || ''"
              >
                {{ currentCite?.recording_title || currentCite?.session_title || 'AI 분석 결과' }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<style scoped>
.workspace-schedule-notice {
  position: fixed;
  top: 24px;
  right: 24px;
  z-index: 80;
  width: min(380px, calc(100vw - 32px));
  padding: 16px;
  color: #1f2937;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: 20px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}

.workspace-schedule-notice-top {
  display: flex;
  align-items: center;
  gap: 12px;
}

.workspace-schedule-notice-icon {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #2563eb;
  background: #eef4ff;
  border: 1px solid #dbe7ff;
  flex: 0 0 auto;
}

.workspace-schedule-notice-icon .material-symbols-outlined {
  font-size: 20px;
}

.workspace-schedule-notice-heading {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.workspace-schedule-notice-heading span {
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

.workspace-schedule-notice-heading strong {
  color: #111827;
  font-size: 16px;
  font-weight: 900;
  line-height: 1.25;
}

.workspace-schedule-notice-close {
  width: 32px;
  height: 32px;
  border: 0;
  border-radius: 999px;
  color: #94a3b8;
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.workspace-schedule-notice-close:hover {
  color: #475569;
  background: #f1f5f9;
}

.workspace-schedule-notice-close .material-symbols-outlined {
  font-size: 19px;
}

.workspace-schedule-notice-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}

.workspace-schedule-notice-item {
  padding: 12px;
  border-radius: 14px;
  background: #f8fafc;
  border: 1px solid #e5edf6;
}

.workspace-schedule-notice-item strong,
.workspace-schedule-notice-item span {
  display: block;
  overflow-wrap: anywhere;
}

.workspace-schedule-notice-item strong {
  color: #111827;
  font-size: 14px;
  font-weight: 900;
  line-height: 1.35;
}

.workspace-schedule-notice-item span {
  margin-top: 5px;
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

.workspace-schedule-notice-more {
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
  padding: 0 4px;
}

.workspace-schedule-notice-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}

.workspace-schedule-notice-primary,
.workspace-schedule-notice-secondary {
  min-height: 38px;
  border: 0;
  border-radius: 999px;
  padding: 0 15px;
  font-size: 13px;
  font-weight: 900;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.workspace-schedule-notice-primary {
  color: #ffffff;
  background: #1f2937;
}

.workspace-schedule-notice-secondary {
  color: #475569;
  background: #f1f5f9;
}

.workspace-schedule-notice-primary:hover,
.workspace-schedule-notice-secondary:hover {
  transform: translateY(-1px);
}

.schedule-notice-fade-enter-active,
.schedule-notice-fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.schedule-notice-fade-enter-from,
.schedule-notice-fade-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

.cite-popover-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: transparent;
}

.cite-popover {
  position: fixed;
  width: 360px;
  background: linear-gradient(160deg, rgba(246, 240, 232, 0.94), rgba(241, 233, 223, 0.72));
  border-radius: 24px;
  box-shadow: 0 24px 48px rgba(148, 163, 184, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(255, 255, 255, 0.82);
  display: flex;
  flex-direction: column;
  padding: 18px;
  transform-origin: right top;
  backdrop-filter: blur(22px) saturate(145%);
  -webkit-backdrop-filter: blur(22px) saturate(145%);
}

:deep(.cite-highlighted-script) {
  background: #ffeb3b;
  color: #111827;
  font-weight: 900;
  border-radius: 5px;
  padding: 1px 5px;
  margin: 0 -2px;
  box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.45), 0 4px 14px rgba(255, 193, 7, 0.28);
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

.cite-transcript-body {
  color: #343437;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.65;
}

.cite-source-title {
  min-width: 0;
  color: #374151;
  font-size: 12px;
  font-weight: 800;
  margin-left: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
}

.cite-source-title:hover {
  color: #111827;
  text-decoration: underline;
}

/* 애니메이션 개선 */
.popover-fade-enter-active {
  transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.popover-fade-leave-active {
  transition: all 0.15s ease;
}
.popover-fade-enter-from {
  opacity: 0;
  transform: scale(0.9) translateY(10px);
}
.popover-fade-leave-to {
  opacity: 0;
  transform: scale(0.95) translateY(5px);
}

.cite-popover-badge {
  width: 30px;
  height: 30px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  background: linear-gradient(180deg, rgba(250,246,240,0.96), rgba(242,235,226,0.76));
  border: 1px solid rgba(255,255,255,0.84);
  box-shadow: 0 12px 24px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255,255,255,0.96);
}

.cite-popover-close-btn {
  color: #8e8e93;
  background: rgba(248,244,238,0.48);
  border: 1px solid rgba(255,255,255,0.72);
  padding: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  transition: all 0.2s ease;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
}

.cite-popover-close-btn:hover {
  background: rgba(250,246,240,0.78);
  color: #1c1c1e;
}

.cite-popover-divider {
  width: 100%;
  height: 1px;
  margin-bottom: 12px;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(206,212,218,0.7), rgba(255,255,255,0));
  flex-shrink: 0;
}

.cite-source-wrap {
  padding: 10px 12px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(249,244,238,0.78), rgba(241,233,224,0.56));
  border: 1px solid rgba(255,255,255,0.78);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.94);
}

/* 팝오버 스크롤바 디자인 */
.custom-scrollbar::-webkit-scrollbar {
  width: 5px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.08);
  border-radius: 10px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.15);
}
</style>
