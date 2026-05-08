<!-- 워크스페이스의 왼쪽 사이드바 본체로, 폴더 탐색기와 음성 전사 탭을 전환하며 보여줍니다. -->
<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import FolderSideTab from './FolderSideTab.vue'
import VoiceTransferSideTab from './VoiceTransferSideTab.vue'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  transcriptions: { type: Array, default: () => [] },
  recordingMode: { type: String, default: 'lecture' },
  activeFileId: { type: String, default: '' },
  citationSourceRequest: { type: Object, default: null },
  isCollapsed: { type: Boolean, default: false }
})

const emit = defineEmits([
  'navigateHome',
  'fileSelect',
  'update:fileTree',
  'update:favorites',
  'addToNote',
  'askAi',
  'openStoredMaterial',
  'openRecording',
  'quizSourceSelect',
  'toggle'
])

const activeTab = ref('voice')
const width = ref(450)
const toastMsg = ref('')
const isResizing = ref(false)
const selectedTranscriptSource = ref(null)

const visibleTranscriptions = computed(() => selectedTranscriptSource.value?.transcriptions || props.transcriptions)

const KOREAN_WEEKDAYS_SHORT = ['일', '월', '화', '수', '목', '금', '토']

const formatTranscriptSourceDate = (endedAt) => {
  if (!endedAt) return '날짜 정보 없음'
  const date = new Date(endedAt)
  if (Number.isNaN(date.getTime())) return '날짜 정보 없음'

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const weekday = KOREAN_WEEKDAYS_SHORT[date.getDay()]
  const hour = String(date.getHours()).padStart(2, '0')
  const minute = String(date.getMinutes()).padStart(2, '0')

  return `${year}.${month}.${day} · ${weekday} · ${hour}:${minute}`
}

const collectTranscriptIds = (recordings = []) => {
  const ids = new Set()
  recordings.forEach((recording) => {
    const transcriptions = recording?.transcriptions || []
    transcriptions.forEach((transcription) => {
      const segments = transcription?.segments || []
      segments.forEach((segment) => {
        const id = segment?.transcript_id || segment?.transcriptId
        if (id) ids.add(String(id))
      })
    })
  })
  return Array.from(ids)
}

const emitQuizSource = ({ title, type, recordings = [], materialId = '', recordingId = '', material = null }) => {
  emit('quizSourceSelect', {
    type,
    title,
    materialId,
    recordingId,
    material,
    transcriptIds: collectTranscriptIds(recordings)
  })
}

const handleMouseMove = (e) => {
  if (!isResizing.value) return
  const newWidth = e.clientX - 12
  if (newWidth > 160 && newWidth < 600) width.value = newWidth
}

const handleMouseUp = () => {
  if (!isResizing.value) return
  isResizing.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.body.classList.remove('is-resizing')
}

onMounted(() => {
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})

const handleResizerMouseDown = () => {
  isResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
}

const showToast = (msg) => {
  toastMsg.value = msg
  setTimeout(() => toastMsg.value = '', 2000)
}

const handleShowLiveTranscripts = () => {
  selectedTranscriptSource.value = null
  activeTab.value = 'voice'
}

const handleOpenMaterial = ({ fileId, node, materialId, material, recording, recordings = [] }) => {
  if (fileId && node) {
    emit('fileSelect', fileId, node)
  }

  const relatedRecordings = recordings.length ? recordings : (recording ? [recording] : [])

  if (relatedRecordings[0]) {
    selectedTranscriptSource.value = {
      title: relatedRecordings[0].title || '연결된 녹음',
      meta: formatTranscriptSourceDate(relatedRecordings[0].endedAt),
      transcriptions: relatedRecordings[0].transcriptions || []
    }
  } else {
    selectedTranscriptSource.value = {
      title: '연결된 전사 없음',
      meta: '날짜 정보 없음',
      transcriptions: []
    }
  }

  emitQuizSource({
    type: 'material',
    title: material?.name || selectedTranscriptSource.value?.title || '강의자료',
    materialId,
    material,
    recordings: relatedRecordings
  })
  emit('openStoredMaterial', materialId)
}

const handleOpenRecording = ({ fileId, node, recording }) => {
  if (fileId && node) {
    emit('fileSelect', fileId, node)
  }

  selectedTranscriptSource.value = {
    title: recording?.title || '저장된 녹음',
    meta: formatTranscriptSourceDate(recording?.endedAt),
    transcriptions: recording?.transcriptions || []
  }
  emitQuizSource({
    type: 'recording',
    title: recording?.title || '저장된 녹음',
    recordingId: recording?.id || recording?.recordingId || '',
    recordings: recording ? [recording] : []
  })
  emit('openRecording', {
    sessionId: fileId,
    recordingId: recording?.id || recording?.recordingId || '',
    recording
  })
  activeTab.value = 'voice'
}

const handleQuizSourceChange = (source) => {
  emit('quizSourceSelect', source)
}

const getNodeRecordings = (node) => {
  if (!node) return []

  const weekRecordings = Array.isArray(node.weeks)
    ? node.weeks.flatMap((week) => Array.isArray(week?.recordings) ? week.recordings : [])
    : []
  const directRecordings = Array.isArray(node.recordings) ? node.recordings : []
  const seen = new Set()

  return [...weekRecordings, ...directRecordings].filter((recording) => {
    const key = recording?.id || recording?.title
    if (!key || seen.has(key)) return false
    seen.add(key)
    return true
  })
}

const splitFullTranscript = (text = '', cite = {}) => {
  const lines = String(text)
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)

  if (!lines.length && cite?.text) {
    lines.push(String(cite.text).trim())
  }

  return lines.map((line, index) => ({
    time: index === 0 && cite?.start_time != null && cite?.end_time != null
      ? `${Math.floor(cite.start_time / 60)}:${String(Math.floor(cite.start_time % 60)).padStart(2, '0')}~${Math.floor(cite.end_time / 60)}:${String(Math.floor(cite.end_time % 60)).padStart(2, '0')}`
      : '',
    speakerId: null,
    speaker: null,
    text: line,
    segments: [{
      id: `${cite?.transcript_id || 'cite'}-${index}`,
      text: line,
      status: 'confirmed'
    }]
  }))
}

const findCitationRecording = (node, cite = {}) => {
  const recordings = getNodeRecordings(node)
  const recordingTitle = String(cite?.recording_title || '').trim()
  const citationText = String(cite?.citation || '')

  return recordings.find((recording) => (
    recordingTitle && recording?.title === recordingTitle
  )) || recordings.find((recording) => (
    recording?.title && citationText.includes(recording.title)
  )) || recordings[0] || null
}

watch(() => props.citationSourceRequest, (request) => {
  if (!request?.cite || !request?.node) return

  const recording = findCitationRecording(request.node, request.cite)
  if (recording) {
    selectedTranscriptSource.value = {
      title: recording.title || request.cite.recording_title || '저장된 녹음',
      meta: formatTranscriptSourceDate(recording.endedAt),
      transcriptions: recording.transcriptions || []
    }
  } else {
    selectedTranscriptSource.value = {
      title: request.cite.recording_title || request.cite.session_title || '출처 전사',
      meta: request.cite.session_date || '날짜 정보 없음',
      transcriptions: splitFullTranscript(request.cite.full_transcript, request.cite)
    }
  }

  activeTab.value = 'voice'
})
</script>

<template>
  <aside
    :class="[{ 'sidebar-collapsed': isCollapsed }]"
    id="sidebar"
    class="transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden rounded-[24px]"
    :style="{ width: isCollapsed ? '0px' : width + 'px', flexShrink: 0 }"
  >
    <div class="card workspace-sidebar-card h-full flex flex-col p-5 overflow-hidden min-w-[280px]">
      <!-- Header -->
      <div class="flex items-center justify-between mb-5">
        <div class="flex items-center gap-2 font-extrabold tracking-tight">
          <div class="w-8 h-8 bg-[#1d1d1f] rounded-[10px] flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px] text-white">menu_book</span>
          </div>
          <span class="text-[20px] collapsible-content">LectoAI</span>
        </div>
        <div class="flex gap-1">
          <button class="btn-ghost-icon p-1.5 rounded-[10px]" title="사이드바 접기" @click="emit('toggle')">
            <span class="material-symbols-outlined text-[22px] text-[#8e8e93]">menu_open</span>
          </button>
        </div>
      </div>

      <!-- Tab Buttons -->
      <div class="workspace-inset-shell p-1 rounded-[18px] flex gap-1.5 mb-3 collapsible-content">
        <button
          class="workspace-inset-pill flex-1 py-2.5 rounded-[15px] text-[12px] font-bold text-gray-500"
          :class="{ 'is-active text-black': activeTab === 'voice' }"
          @click="handleShowLiveTranscripts"
        >전사 내용</button>
        <button
          class="workspace-inset-pill flex-1 py-2.5 rounded-[15px] text-[12px] font-bold text-gray-500"
          :class="{ 'is-active text-black': activeTab === 'folders' }"
          @click="activeTab = 'folders'"
        >폴더</button>
      </div>

      <!-- Tab Content -->
      <div class="flex-1 flex flex-col overflow-hidden">
        <FolderSideTab 
          v-if="activeTab === 'folders'"
          :fileTree="fileTree"
          :favorites="favorites"
          :active-file-id="activeFileId"
          @update:fileTree="emit('update:fileTree', $event)"
          @update:favorites="emit('update:favorites', $event)"
          @fileSelect="(id, node) => emit('fileSelect', id, node)"
          @openMaterial="handleOpenMaterial"
          @openRecording="handleOpenRecording"
          @quizSourceChange="handleQuizSourceChange"
          @showToast="showToast"
          class="sidebar-content-animate"
        />
        <div v-else class="flex flex-col flex-1 min-h-0 sidebar-content-animate">
          <div v-if="selectedTranscriptSource" class="selected-transcript-source">
            <div class="min-w-0">
              <p>{{ selectedTranscriptSource.title }}</p>
              <span>{{ selectedTranscriptSource.meta }}</span>
            </div>
            <button type="button" title="실시간 전사로 돌아가기" @click="handleShowLiveTranscripts">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>
          <VoiceTransferSideTab
            :transcriptions="visibleTranscriptions"
            :recording-mode="recordingMode"
            @addToNote="(text, source) => emit('addToNote', text, source)"
            @askAi="emit('askAi', $event)"
          />
        </div>
      </div>

      <!-- Home Button -->
      <footer class="mt-auto pt-4 flex items-center justify-center">
        <button class="btn-ghost-icon p-2.5 rounded-xl cursor-pointer flex items-center text-[#aeaeb2] justify-center" @click="emit('navigateHome')">
          <span class="material-symbols-outlined text-[24px]" style="font-variation-settings: 'FILL' 1">home</span>
        </button>
      </footer>
    </div>
  </aside>

  <!-- Resizer -->
  <div
    v-show="!isCollapsed"
    class="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
    id="resizer-left"
    :class="{ 'is-collapsed': isCollapsed }"
    @mousedown="handleResizerMouseDown"
  >
    <div class="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
  </div>



  <!-- Toast -->
  <div :class="['toast', { show: toastMsg }]" id="toast">{{ toastMsg }}</div>
</template>

<style scoped>
.workspace-sidebar-card {
  background: var(--workspace-sidebar-card-bg);
  border: 1px solid var(--workspace-sidebar-card-border);
  box-shadow: var(--workspace-sidebar-card-shadow);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.workspace-sidebar-card::before {
  background: var(--workspace-sidebar-card-overlay);
}

.workspace-sidebar-card::after {
  border-color: var(--workspace-sidebar-card-inner-border);
}

.selected-transcript-source {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  padding: 8px 9px 8px 11px;
  border-radius: 16px;
  background: rgba(250, 247, 242, 0.86);
  border: 1px solid rgba(222, 205, 182, 0.58);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.76);
}

.selected-transcript-source p {
  margin: 0;
  overflow: hidden;
  color: #1d1d1f;
  font-size: 12px;
  font-weight: 900;
  line-height: 1.18;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.selected-transcript-source span {
  display: block;
  margin-top: 2px;
  color: #8e8e93;
  font-size: 10.5px;
  font-weight: 800;
}

.selected-transcript-source button {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  color: #8e8e93;
  background: rgba(242, 239, 234, 0.88);
}

.selected-transcript-source button .material-symbols-outlined {
  font-size: 14px;
}
</style>
