<!-- 음성 녹음, 실시간 전사, AI 분석 및 교차 참조가 이루어지는 작업실 페이지 컴포넌트입니다. -->
<script setup>
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import LeftSidebar from '../../components/workspace/LeftSidebar.vue'
import MainContent from '../../components/workspace/MainContent.vue'
import RightSidebar from '../../components/workspace/RightSidebar.vue'
import { useChat } from '../../composables/useChat'

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  isRecording: { type: Boolean, default: false },
  isRecordingPaused: { type: Boolean, default: false },
  recordingMode: { type: String, default: 'lecture' },
  recordingTimeText: { type: String, default: '00:00:00' },
  recordingAudioLevel: { type: Number, default: 0 },
  diarizationEnabled: { type: Boolean, default: false },
  diarizationStatus: { type: String, default: 'idle' },
  activeFileName: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  activeFileType: { type: String, default: 'lecture' },
  currentAttachments: { type: Array, default: () => [] },
  currentRecordings: { type: Array, default: () => [] },
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
  'generateMaterialSummary',
  'deleteSummary',
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
const materialEvidenceRequest = ref(null)
const mainContentTabRequest = ref(null)
const isEmbeddedFolderOpen = ref(false)
const workspaceActiveMainTab = ref('materials')
const workspaceUnifiedCardRef = ref(null)
const scriptPaneWidth = ref(50)
const isScriptPaneResizing = ref(false)

const DEFAULT_SCRIPT_PANE_PERCENT = 50
const MIN_SCRIPT_PANE_WIDTH = 280
const RESIZE_KEY_STEP = 2

function getScriptPaneMinPercent() {
  const cardWidth = workspaceUnifiedCardRef.value?.getBoundingClientRect().width || 0
  if (!cardWidth) return 28
  return Math.min(DEFAULT_SCRIPT_PANE_PERCENT, Math.max(24, (MIN_SCRIPT_PANE_WIDTH / cardWidth) * 100))
}

function setScriptPaneWidth(nextPercent) {
  const minPercent = getScriptPaneMinPercent()
  const clamped = Math.min(DEFAULT_SCRIPT_PANE_PERCENT, Math.max(minPercent, nextPercent))
  scriptPaneWidth.value = Number(clamped.toFixed(2))
}

function setScriptPaneWidthFromPointer(clientX) {
  const rect = workspaceUnifiedCardRef.value?.getBoundingClientRect()
  if (!rect?.width) return
  setScriptPaneWidth(((clientX - rect.left) / rect.width) * 100)
}

function stopScriptPaneResize() {
  if (!isScriptPaneResizing.value) return
  isScriptPaneResizing.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.body.classList.remove('is-resizing')
  window.removeEventListener('pointermove', handleScriptPanePointerMove)
  window.removeEventListener('pointerup', stopScriptPaneResize)
  window.removeEventListener('pointercancel', stopScriptPaneResize)
}

function handleScriptPanePointerMove(event) {
  if (!isScriptPaneResizing.value) return
  setScriptPaneWidthFromPointer(event.clientX)
}

function handleScriptPanePointerDown(event) {
  if (event.button !== undefined && event.button !== 0) return
  event.preventDefault()
  isScriptPaneResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
  setScriptPaneWidthFromPointer(event.clientX)
  window.addEventListener('pointermove', handleScriptPanePointerMove)
  window.addEventListener('pointerup', stopScriptPaneResize)
  window.addEventListener('pointercancel', stopScriptPaneResize)
}

function handleScriptPaneResizeKeydown(event) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()

  if (event.key === 'Home') {
    setScriptPaneWidth(getScriptPaneMinPercent())
    return
  }

  if (event.key === 'End') {
    setScriptPaneWidth(DEFAULT_SCRIPT_PANE_PERCENT)
    return
  }

  const direction = event.key === 'ArrowLeft' ? -1 : 1
  setScriptPaneWidth(scriptPaneWidth.value + (direction * RESIZE_KEY_STEP))
}

function resetScriptPaneWidth() {
  setScriptPaneWidth(DEFAULT_SCRIPT_PANE_PERCENT)
}

onUnmounted(() => {
  stopScriptPaneResize()
})

watch(() => props.activeFileId, () => {
  citationSourceRequest.value = null
  materialEvidenceRequest.value = null
  clearHistory()
  closeCitePopover()
  emit('update:aiInput', '')
})

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

function findMaterialInNode(node, cite = {}) {
  const materialId = String(cite?.material_id || '')
  const storedName = String(cite?.stored_name || '')
  const weeks = Array.isArray(node?.weeks) ? node.weeks : []

  for (const week of weeks) {
    const materials = Array.isArray(week?.materials) ? week.materials : []
    for (const material of materials) {
      if (materialId && String(material?.id || '') === materialId) return material
      if (storedName && String(material?.storedName || '') === storedName) return material
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
  if (cite?.source_type === 'material') {
    const material = findMaterialInNode(node, cite)
    const materialId = material?.id || cite?.material_id
    if (materialId) {
      emit('openStoredMaterial', materialId)
    }
    closeCitePopover()
    return
  }

  citationSourceRequest.value = {
    id: `${sessionId}-${cite?.transcript_id || cite?.citation || Date.now()}`,
    cite,
    node
  }
  clearHistory()
  emit('update:aiInput', '')
  closeCitePopover()
}

async function openEvidenceSource(cite) {
  if (!cite) return

  if (cite.source_type !== 'material') {
    openCitationSource(cite)
    return
  }

  const sessionId = cite.session_id
  if (!sessionId) return

  const node = findNodeById(props.fileTree, sessionId)
  if (!node) return

  emit('fileSelect', sessionId, node)
  isLeftSidebarCollapsed.value = false

  const material = findMaterialInNode(node, cite)
  const materialId = material?.id || cite.material_id
  if (materialId) {
    emit('openStoredMaterial', materialId)
  }

  await nextTick()

  materialEvidenceRequest.value = {
    id: `${sessionId}-${materialId || cite.stored_name || cite.citation}-${cite.page || 0}-${Date.now()}`,
    cite,
    materialId,
    page: Number(cite.page || 1),
    text: cite.text || ''
  }
}

async function handleOpenRecording(payload) {
  emit('openRecording', payload)
  await nextTick()

  mainContentTabRequest.value = {
    id: `${payload?.sessionId || ''}-${payload?.recordingId || ''}-${Date.now()}`,
    tab: 'summary',
    summaryTab: 'summary'
  }
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

const currentCitationTitle = computed(() => (
  currentCite.value?.source_type === 'material'
    ? (currentCite.value?.material_name || currentCite.value?.file_title || '강의자료')
    : currentCite.value?.recording_title
  || currentCite.value?.session_title
  || currentCite.value?.file_title
  || 'AI 분석 결과'
))

const currentCitationLabel = computed(() => currentCite.value?.citation || (
  currentCite.value?.source_type === 'material' ? '연결된 PDF' : '연결된 전사'
))

const currentCitationSourceIcon = computed(() => (
  currentCite.value?.source_type === 'material' ? 'picture_as_pdf' : 'folder_open'
))

const currentCitationSourceCaption = computed(() => (
  currentCite.value?.source_type === 'material' ? 'PDF 자료' : '출처'
))

function collectTranscriptIds(recordings = []) {
  const ids = new Set()
  recordings.forEach((recording) => {
    const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
    transcriptions.forEach((transcription) => {
      const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
      segments.forEach((segment) => {
        const id = segment?.transcript_id || segment?.transcriptId
        if (id) ids.add(String(id))
      })
    })
  })
  return Array.from(ids)
}

const activeWorkspaceSource = computed(() => {
  if (!props.activeFileId) return null

  const materials = Array.isArray(props.currentAttachments) ? props.currentAttachments : []
  const recordings = Array.isArray(props.currentRecordings) ? props.currentRecordings : []
  const materialSources = materials.map((material, index) => ({
    id: material?.id || material?.storedName || material?.url || material?.name || `material-${index}`,
    type: 'material',
    title: material?.name || material?.title || `강의자료 ${index + 1}`,
    material,
    transcriptIds: []
  }))
  const recordingSources = recordings.map((recording, index) => ({
    id: recording?.id || recording?.recordingId || recording?.title || `recording-${index}`,
    type: 'recording',
    title: recording?.title || `녹음본 ${index + 1}`,
    recordingId: recording?.id || recording?.recordingId || '',
    recording,
    transcriptIds: collectTranscriptIds([recording])
  }))
  const sources = [...materialSources, ...recordingSources]
  const transcriptIds = Array.from(new Set(recordingSources.flatMap((source) => source.transcriptIds || [])))
  const sourceCount = sources.length

  return {
    type: sourceCount ? 'workspace' : 'empty',
    title: props.activeFileName ? `${props.activeFileName} 전체 자료` : '현재 파일 전체 자료',
    sessionId: props.activeFileId,
    sourceCount,
    sources,
    recordings,
    transcriptIds
  }
})
</script>

<template>
  <div 
    class="workspace-page-shell p-[12px] flex relative h-full w-full text-[#1e293b] overflow-hidden transition-all duration-400"
    :class="[
      { 'gap-[12px]': !isLeftSidebarCollapsed || isRightSidebarVisible }
    ]"
  >
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

    <section
      ref="workspaceUnifiedCardRef"
      class="workspace-unified-card card relative z-10"
      :class="{
        'has-script-tab-line': workspaceActiveMainTab === 'materials' || workspaceActiveMainTab === 'quiz',
        'is-resizing-script-pane': isScriptPaneResizing
      }"
      :style="{ '--workspace-script-pane-width': `${scriptPaneWidth}%` }"
    >
      <div id="workspace-unified-folder-drawer-host" class="workspace-unified-folder-drawer-host"></div>
      <div id="workspace-unified-audio-player-host" class="workspace-unified-audio-player-host"></div>

      <LeftSidebar
        embedded
        class="workspace-unified-script-pane"
        :isCollapsed="false"
        :isRecording="isRecording"
        :isRecordingPaused="isRecordingPaused"
        :recordingMode="recordingMode"
        :recordingTimeText="recordingTimeText"
        :recordingAudioLevel="recordingAudioLevel"
        :diarization-enabled="diarizationEnabled"
        :diarization-status="diarizationStatus"
        :transcriptions="transcriptions"
        :fileTree="fileTree"
        :favorites="favorites"
        :activeFileName="activeFileName"
        :activeFileId="activeFileId"
        :activeFileType="activeFileType"
        :citationSourceRequest="citationSourceRequest"
        :embedded-folder-open="isEmbeddedFolderOpen"
        :script-tab-line-visible="true"
        @update:embeddedFolderOpen="isEmbeddedFolderOpen = $event"
        @update:embedded-folder-open="isEmbeddedFolderOpen = $event"
        @navigateHome="emit('navigateHome')"
        @fileSelect="(id, node) => emit('fileSelect', id, node)"
        @update:fileTree="emit('update:fileTree', $event)"
        @update:favorites="emit('update:favorites', $event)"
        @addToNote="(text, source) => emit('addToNote', text, source)"
        @askAi="(word) => emit('askAi', word)"
        @openStoredMaterial="emit('openStoredMaterial', $event)"
        @openRecording="handleOpenRecording"
        @startRecording="emit('startRecording', $event)"
        @pauseRecording="emit('pauseRecording')"
        @resumeRecording="emit('resumeRecording')"
        @stopRecording="emit('stopRecording')"
      />

      <div
        class="workspace-unified-resizer"
        role="separator"
        aria-label="스크립트 영역 너비 조절"
        aria-orientation="vertical"
        aria-valuemin="24"
        aria-valuemax="50"
        :aria-valuenow="Math.round(scriptPaneWidth)"
        tabindex="0"
        title="드래그해서 스크립트 영역 너비 조절"
        @pointerdown="handleScriptPanePointerDown"
        @keydown="handleScriptPaneResizeKeydown"
        @dblclick="resetScriptPaneWidth"
      ></div>

      <MainContent
        embedded
        class="workspace-unified-main-pane"
        :isRecording="isRecording"
        :isRecordingPaused="isRecordingPaused"
        :recordingMode="recordingMode"
        :recordingTimeText="recordingTimeText"
        :recordingAudioLevel="recordingAudioLevel"
        :diarization-enabled="diarizationEnabled"
        :diarization-status="diarizationStatus"
        :activeFileName="activeFileName"
        :activeFileId="activeFileId"
        :activeFileType="activeFileType"
        :currentAttachments="currentAttachments"
        :currentRecordings="currentRecordings"
        :transcriptions="transcriptions"
        :currentPreviewMaterial="currentPreviewMaterial"
        :materialEvidenceRequest="materialEvidenceRequest"
        :summaryState="summaryState"
        :summaryNotes="summaryNotes"
        :quizSource="activeWorkspaceSource"
        :tabRequest="mainContentTabRequest"
        :folder-drawer-open="isEmbeddedFolderOpen"
        @startRecording="emit('startRecording', $event)"
        @pauseRecording="emit('pauseRecording')"
        @resumeRecording="emit('resumeRecording')"
        @stopRecording="emit('stopRecording')"
        @generateMaterialSummary="emit('generateMaterialSummary', $event)"
        @deleteSummary="emit('deleteSummary', $event)"
        @mainSidebarToggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
        @rightSidebarToggle="emit('rightSidebarToggle')"
        @askAi="(word) => emit('askAi', word)"
        @addToNote="(text, source) => emit('addToNote', text, source)"
        @uploadLectureMaterials="emit('uploadLectureMaterials', $event)"
        @closePreviewMaterial="emit('closePreviewMaterial')"
        @openStoredMaterial="emit('openStoredMaterial', $event)"
        @toggleFolderDrawer="isEmbeddedFolderOpen = !isEmbeddedFolderOpen"
        @activeTabChange="workspaceActiveMainTab = $event"
      />
    </section>
    
    <RightSidebar 
      class="relative z-10"
      :visible="isRightSidebarVisible" 
      :aiInput="aiInput"
      :activeFileId="activeFileId"
      :chatSource="activeWorkspaceSource"
      @update:aiInput="emit('update:aiInput', $event)"
      @openEvidenceSource="openEvidenceSource"
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
          <div class="cite-popover-header">
            <div class="cite-popover-title">
              <div class="cite-popover-badge">
                <span class="material-symbols-outlined">fact_check</span>
              </div>
              <div>
                <span>근거 정보</span>
                <p>{{ currentCitationLabel }}</p>
              </div>
            </div>
            <button class="cite-popover-close-btn" aria-label="근거 정보 닫기" @click="closeCitePopover">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>

          <div class="cite-transcript-scroll custom-scrollbar">
            <div class="cite-transcript-kicker">
              <span class="material-symbols-outlined">subject</span>
              <span>발췌 원문</span>
            </div>
            <div 
              class="cite-transcript-body whitespace-pre-wrap break-keep"
              v-html="highlightedTranscript"
            >
            </div>
          </div>

          <div class="cite-source-wrap shrink-0">
            <button
              type="button"
              class="cite-source-title"
              @click="openEvidenceSource(currentCite)"
              :title="currentCite?.file_title || currentCite?.session_title || ''"
            >
              <span class="material-symbols-outlined">{{ currentCitationSourceIcon }}</span>
              <span>
                <small>{{ currentCitationSourceCaption }}</small>
                <strong>{{ currentCitationTitle }}</strong>
              </span>
              <span class="material-symbols-outlined cite-source-arrow">open_in_new</span>
            </button>
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

.workspace-unified-card {
  --workspace-script-pane-width: 50%;
  flex: 1 1 0%;
  min-width: 0;
  height: 100%;
  display: flex;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(226, 224, 232, 0.9);
  border-radius: 24px;
  box-shadow:
    0 26px 52px rgba(148, 163, 184, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.workspace-unified-card.is-resizing-script-pane,
.workspace-unified-card.is-resizing-script-pane * {
  cursor: col-resize !important;
  user-select: none;
}

.workspace-unified-card::before,
.workspace-unified-card::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  z-index: 12;
  height: 1px;
  background: rgba(0, 0, 0, 0.06);
  pointer-events: none;
}

.workspace-unified-card::before {
  top: 82px;
  display: none;
}

.workspace-unified-card.has-script-tab-line::before {
  display: none;
}

.workspace-unified-card::after {
  top: 48px;
}

.workspace-unified-folder-drawer-host {
  position: absolute;
  inset: 0;
  z-index: 90;
  pointer-events: none;
}

.workspace-unified-folder-drawer-host :deep(.embedded-folder-drawer) {
  pointer-events: auto;
}

.workspace-unified-audio-player-host {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 75;
  height: 96px;
  pointer-events: none;
}

.workspace-unified-audio-player-host :deep(*) {
  pointer-events: auto;
}

.workspace-unified-script-pane {
  position: relative;
  flex: 0 0 var(--workspace-script-pane-width);
  width: var(--workspace-script-pane-width) !important;
  transition: flex-basis 0.16s ease, width 0.16s ease;
}

.workspace-unified-script-pane::after {
  display: none;
}

.workspace-unified-main-pane {
  flex: 1 1 0%;
  min-width: 50% !important;
}

.workspace-unified-resizer {
  position: relative;
  z-index: 68;
  flex: 0 0 12px;
  width: 12px;
  align-self: stretch;
  margin-left: -7px;
  margin-right: -5px;
  cursor: col-resize;
  touch-action: none;
  outline: none;
}

.workspace-unified-resizer::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 2px;
  transform: translateX(-50%);
  background: rgba(226, 224, 232, 0.95);
  transition: width 0.16s ease, background-color 0.16s ease;
}

.workspace-unified-resizer::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 6px;
  height: 54px;
  border-radius: 999px;
  transform: translate(-50%, -50%);
  background: #c7ccd6;
  opacity: 0;
  transition: opacity 0.16s ease, background-color 0.16s ease;
}

.workspace-unified-resizer:hover::before,
.workspace-unified-resizer:focus-visible::before,
.workspace-unified-card.is-resizing-script-pane .workspace-unified-resizer::before {
  width: 3px;
  background: #9aa3b2;
}

.workspace-unified-resizer:hover::after,
.workspace-unified-resizer:focus-visible::after,
.workspace-unified-card.is-resizing-script-pane .workspace-unified-resizer::after {
  opacity: 1;
}

.workspace-unified-card.is-resizing-script-pane .workspace-unified-script-pane {
  transition: none;
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
  width: min(340px, calc(100vw - 32px));
  background: rgba(255, 255, 255, 0.96);
  border-radius: 22px;
  box-shadow: 0 24px 64px rgba(15, 23, 42, 0.16), 0 1px 0 rgba(255, 255, 255, 0.86) inset;
  border: 1px solid rgba(226, 232, 240, 0.88);
  display: flex;
  flex-direction: column;
  padding: 16px;
  transform-origin: right top;
  backdrop-filter: blur(20px) saturate(140%);
  -webkit-backdrop-filter: blur(20px) saturate(140%);
}

.cite-popover-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.84);
}

.cite-popover-title {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  min-width: 0;
}

.cite-popover-title span:not(.material-symbols-outlined) {
  display: block;
  color: #111827;
  font-size: 16px;
  font-weight: 800;
  line-height: 1.25;
}

.cite-popover-title p {
  max-width: 220px;
  margin-top: 4px;
  color: #64748b;
  font-size: 11px;
  font-weight: 650;
  line-height: 1.45;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cite-transcript-scroll {
  max-height: 320px;
  margin: 12px 0;
  overflow-y: auto;
  padding: 1px 2px 2px;
}

.cite-transcript-kicker {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
  color: #64748b;
  font-size: 11px;
  font-weight: 800;
}

.cite-transcript-kicker .material-symbols-outlined {
  font-size: 15px;
}

:deep(.cite-highlighted-script) {
  background: rgba(253, 224, 71, 0.42);
  color: #111827;
  font-weight: 850;
  border-radius: 6px;
  padding: 2px 4px;
  margin: 0 -2px;
  box-shadow: none;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

.cite-transcript-body {
  color: #1f2937;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.7;
}

.cite-source-title {
  width: 100%;
  min-width: 0;
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) 18px;
  align-items: center;
  gap: 10px;
  color: #334155;
  text-align: left;
  cursor: pointer;
}

.cite-source-title > .material-symbols-outlined:first-child {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #475569;
  font-size: 17px;
  border-radius: 10px;
  background: #f1f5f9;
}

.cite-source-title small {
  display: block;
  color: #94a3b8;
  font-size: 10px;
  font-weight: 850;
  line-height: 1.1;
  letter-spacing: 0.04em;
}

.cite-source-title strong {
  display: block;
  margin-top: 3px;
  color: #1e293b;
  font-size: 13px;
  font-weight: 800;
  line-height: 1.25;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cite-source-title:hover {
  color: #0f172a;
}

.cite-source-title:hover strong {
  text-decoration: underline;
  text-underline-offset: 3px;
}

.cite-source-arrow {
  color: #94a3b8;
  font-size: 17px;
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
  width: 32px;
  height: 32px;
  border-radius: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #2563eb;
  background: #eff6ff;
  border: 1px solid #dbeafe;
}

.cite-popover-badge .material-symbols-outlined {
  font-size: 18px;
}

.cite-popover-close-btn {
  width: 32px;
  height: 32px;
  color: #64748b;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  transition: all 0.2s ease;
}

.cite-popover-close-btn:hover {
  background: #f1f5f9;
  color: #0f172a;
}

.cite-popover-close-btn .material-symbols-outlined {
  font-size: 20px;
}

.cite-source-wrap {
  padding: 10px;
  border-radius: 14px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
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
