<!-- 워크스페이스의 왼쪽 사이드바 본체로, 폴더 탐색기와 음성 전사 탭을 전환하며 보여줍니다. -->
<script setup>
import { ref, computed, nextTick, onMounted, onUnmounted, watch } from 'vue'
import FolderSideTab from './FolderSideTab.vue'
import VoiceTransferSideTab from './VoiceTransferSideTab.vue'
import { isWorkspaceUuid, updateWorkspaceFile } from '../../api/workspaceApi.js'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  transcriptions: { type: Array, default: () => [] },
  isRecording: { type: Boolean, default: false },
  isRecordingPaused: { type: Boolean, default: false },
  recordingMode: { type: String, default: 'lecture' },
  recordingTimeText: { type: String, default: '00:00:00' },
  recordingAudioLevel: { type: Number, default: 0 },
  activeFileName: { type: String, default: '' },
  activeFileType: { type: String, default: 'lecture' },
  diarizationEnabled: { type: Boolean, default: false },
  diarizationStatus: { type: String, default: 'idle' },
  activeFileId: { type: String, default: '' },
  citationSourceRequest: { type: Object, default: null },
  isCollapsed: { type: Boolean, default: false },
  embedded: { type: Boolean, default: false },
  embeddedFolderOpen: { type: Boolean, default: false },
  scriptTabLineVisible: { type: Boolean, default: false }
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
  'toggle',
  'startRecording',
  'pauseRecording',
  'resumeRecording',
  'stopRecording',
  'update:embeddedFolderOpen'
])

const activeTab = ref('voice')
const width = ref(450)
const toastMsg = ref('')
const isResizing = ref(false)
const selectedTranscriptSource = ref(null)
const activePlaybackRecording = ref(null)
const localEmbeddedFolderOpen = ref(false)
const playbackAudioRef = ref(null)
const playbackMediaDuration = ref(0)
const isPlaybackPlaying = ref(false)
const playbackProgress = ref(0)
const playbackSpeed = ref(1)
const isEditingFileTitle = ref(false)
const isFileTitleSaving = ref(false)
const fileTitleDraft = ref('')
const fileTitleInputRef = ref(null)
const isFileTitleComposing = ref(false)
const shouldCommitTitleAfterComposition = ref(false)
let playbackTimer = null

const playbackSpeeds = [1, 1.25, 1.5, 2]

const isEmbeddedFolderOpen = computed({
  get: () => props.embedded ? props.embeddedFolderOpen : localEmbeddedFolderOpen.value,
  set: (value) => {
    if (props.embedded) {
      emit('update:embeddedFolderOpen', value)
      return
    }
    localEmbeddedFolderOpen.value = value
  }
})

const visibleTranscriptions = computed(() => selectedTranscriptSource.value?.transcriptions || props.transcriptions)
const sidebarFileTitle = computed(() => props.activeFileName || '파일을 선택하세요')
const playbackSourceTitle = computed(() => (
  selectedTranscriptSource.value?.title
  || activePlaybackRecording.value?.title
  || '저장된 녹음'
))
const playbackSourceMeta = computed(() => (
  selectedTranscriptSource.value?.meta
  || formatTranscriptSourceDate(activePlaybackRecording.value?.endedAt)
))

const findNodeById = (nodes = [], id = '') => {
  for (const node of nodes) {
    if (node?.id === id) return node
    if (Array.isArray(node?.children)) {
      const found = findNodeById(node.children, id)
      if (found) return found
    }
  }
  return null
}

const replaceNodeById = (nodes = [], id = '', replacement = null) => (
  nodes.map((node) => {
    if (node?.id === id) return replacement || node
    if (Array.isArray(node?.children)) {
      return { ...node, children: replaceNodeById(node.children, id, replacement) }
    }
    return node
  })
)

const startFileTitleEdit = async () => {
  if (!isWorkspaceUuid(props.activeFileId)) return
  fileTitleDraft.value = sidebarFileTitle.value
  isEditingFileTitle.value = true
  await nextTick()
  fileTitleInputRef.value?.focus()
  fileTitleInputRef.value?.select()
}

const cancelFileTitleEdit = () => {
  isEditingFileTitle.value = false
  isFileTitleSaving.value = false
  isFileTitleComposing.value = false
  shouldCommitTitleAfterComposition.value = false
  fileTitleDraft.value = sidebarFileTitle.value
}

const commitFileTitleEdit = async () => {
  if (!isEditingFileTitle.value || isFileTitleSaving.value || isFileTitleComposing.value) return

  const nextTitle = fileTitleDraft.value.trim()
  if (!nextTitle) {
    cancelFileTitleEdit()
    return
  }

  if (nextTitle === props.activeFileName) {
    isEditingFileTitle.value = false
    return
  }

  if (!isWorkspaceUuid(props.activeFileId)) {
    cancelFileTitleEdit()
    return
  }

  isFileTitleSaving.value = true
  try {
    const updatedNode = await updateWorkspaceFile(props.activeFileId, { title: nextTitle })
    const currentNode = findNodeById(props.fileTree, props.activeFileId) || {}
    const mergedNode = { ...currentNode, ...updatedNode, name: nextTitle }
    emit('update:fileTree', replaceNodeById(props.fileTree, props.activeFileId, mergedNode))
    emit('fileSelect', props.activeFileId, mergedNode)
    isEditingFileTitle.value = false
  } catch (error) {
    console.error('[workspace] file title update failed:', error)
    showToast('제목 수정에 실패했습니다.')
  } finally {
    isFileTitleSaving.value = false
  }
}

const handleFileTitleEnter = async (event) => {
  if (event.isComposing || event.keyCode === 229 || isFileTitleComposing.value) {
    shouldCommitTitleAfterComposition.value = true
    return
  }

  event.preventDefault()
  await commitFileTitleEdit()
}

const handleFileTitleCompositionStart = () => {
  isFileTitleComposing.value = true
}

const handleFileTitleCompositionEnd = async () => {
  isFileTitleComposing.value = false
  if (!shouldCommitTitleAfterComposition.value) return

  shouldCommitTitleAfterComposition.value = false
  await commitFileTitleEdit()
}

const sidebarVoiceDotStyles = computed(() => {
  const level = props.isRecordingPaused ? 0 : Math.min(1, Math.max(0, Number(props.recordingAudioLevel) || 0))
  const weights = [0.65, 1.05, 1.35, 0.95, 0.7]
  return weights.map((weight, index) => {
    const scale = 0.5 + Math.min(1.45, level * weight * 1.75)
    const opacity = Math.min(1, 0.34 + level * (0.46 + index * 0.025))
    return {
      animation: 'none',
      transform: `scaleY(${scale.toFixed(2)})`,
      opacity: opacity.toFixed(2)
    }
  })
})

const handleSidebarStartRecording = () => {
  closePlaybackBar()
  selectedTranscriptSource.value = null
  activeTab.value = 'voice'

  emit('startRecording', {
    mode: props.activeFileType === 'meeting' ? 'meeting' : 'lecture',
    diarizationEnabled: false
  })
}

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

const parsePlaybackDuration = (durationText = '') => {
  const parts = String(durationText || '').split(':').map((part) => Number(part))
  if (!parts.length || parts.some((part) => Number.isNaN(part))) return 0

  const [hours = 0, minutes = 0, seconds = 0] = parts.length === 3
    ? parts
    : [0, parts[0] || 0, parts[1] || 0]
  return (hours * 3600) + (minutes * 60) + seconds
}

const getRecordingTranscriptDuration = (recording = {}) => {
  const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
  const maxEnd = transcriptions.reduce((max, transcription) => {
    const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
    return segments.reduce((segmentMax, segment) => {
      const end = Number(segment?.end_time ?? segment?.endTime ?? segment?.end ?? 0)
      return Number.isFinite(end) ? Math.max(segmentMax, end) : segmentMax
    }, max)
  }, 0)
  return Math.ceil(maxEnd)
}

const playbackDurationSeconds = computed(() => {
  const recording = activePlaybackRecording.value
  if (!recording) return 0
  return playbackMediaDuration.value
    || Number(recording.durationSeconds)
    || parsePlaybackDuration(recording.durationText)
    || getRecordingTranscriptDuration(recording)
    || 540
})

const playbackCurrentSeconds = computed(() => {
  const duration = playbackDurationSeconds.value
  if (!duration) return 0
  return Math.round(duration * (Number(playbackProgress.value) || 0) / 100)
})

const formatPlaybackTime = (seconds = 0) => {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0))
  const minutes = Math.floor(safeSeconds / 60)
  const remainSeconds = String(safeSeconds % 60).padStart(2, '0')
  return `${minutes}:${remainSeconds}`
}

const setPlaybackSecond = (seconds) => {
  const duration = playbackDurationSeconds.value || 1
  const next = Math.min(duration, Math.max(0, Number(seconds) || 0))
  playbackProgress.value = Number(((next / duration) * 100).toFixed(2))
  const audio = playbackAudioRef.value
  if (activePlaybackRecording.value?.audioUrl && audio && Number.isFinite(audio.duration)) {
    audio.currentTime = next
  }
}

const clearPlaybackTimer = () => {
  if (!playbackTimer) return
  clearInterval(playbackTimer)
  playbackTimer = null
}

const closePlaybackBar = () => {
  clearPlaybackTimer()
  playbackAudioRef.value?.pause()
  activePlaybackRecording.value = null
  isPlaybackPlaying.value = false
  playbackProgress.value = 0
  playbackMediaDuration.value = 0
}

const setActivePlaybackRecording = (recording = {}, sessionId = '') => {
  activePlaybackRecording.value = {
    ...recording,
    sessionId,
    recordingId: recording?.id || recording?.recordingId || ''
  }
  playbackProgress.value = 0
  playbackMediaDuration.value = 0
  isPlaybackPlaying.value = false
}

const togglePlayback = async () => {
  if (!activePlaybackRecording.value) return
  const audio = playbackAudioRef.value
  if (activePlaybackRecording.value?.audioUrl && audio) {
    audio.playbackRate = playbackSpeed.value
    if (isPlaybackPlaying.value) {
      audio.pause()
      isPlaybackPlaying.value = false
      return
    }

    try {
      await audio.play()
      isPlaybackPlaying.value = true
    } catch (error) {
      console.error('[workspace] audio playback failed:', error)
      isPlaybackPlaying.value = false
    }
    return
  }

  isPlaybackPlaying.value = !isPlaybackPlaying.value
}

const skipPlayback = (amount) => {
  setPlaybackSecond(playbackCurrentSeconds.value + amount)
}

const cyclePlaybackSpeed = () => {
  const index = playbackSpeeds.indexOf(playbackSpeed.value)
  playbackSpeed.value = playbackSpeeds[(index + 1) % playbackSpeeds.length]
}

const handlePlaybackLoadedMetadata = () => {
  const duration = Number(playbackAudioRef.value?.duration)
  playbackMediaDuration.value = Number.isFinite(duration) ? Math.round(duration) : 0
}

const handlePlaybackTimeUpdate = () => {
  const audio = playbackAudioRef.value
  const duration = Number(audio?.duration)
  const currentTime = Number(audio?.currentTime)
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(currentTime)) return
  playbackProgress.value = Number(((currentTime / duration) * 100).toFixed(2))
}

const handlePlaybackRangeInput = () => {
  const audio = playbackAudioRef.value
  if (!activePlaybackRecording.value?.audioUrl || !audio) return
  const duration = Number(audio.duration) || playbackDurationSeconds.value
  audio.currentTime = Math.min(duration, Math.max(0, (Number(playbackProgress.value) || 0) * duration / 100))
}

const handlePlaybackEnded = () => {
  setPlaybackSecond(playbackDurationSeconds.value)
  isPlaybackPlaying.value = false
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
  clearPlaybackTimer()
})

watch(isPlaybackPlaying, (playing) => {
  clearPlaybackTimer()
  if (activePlaybackRecording.value?.audioUrl) return
  if (!playing) return

  playbackTimer = setInterval(() => {
    const nextSecond = playbackCurrentSeconds.value + playbackSpeed.value
    if (nextSecond >= playbackDurationSeconds.value) {
      setPlaybackSecond(playbackDurationSeconds.value)
      isPlaybackPlaying.value = false
      return
    }
    setPlaybackSecond(nextSecond)
  }, 1000)
})

watch(playbackSpeed, (speed) => {
  if (playbackAudioRef.value) {
    playbackAudioRef.value.playbackRate = speed
  }
})

watch(() => props.activeFileId, () => {
  cancelFileTitleEdit()
})

watch(() => props.activeFileName, (nextTitle) => {
  if (!isEditingFileTitle.value) fileTitleDraft.value = nextTitle || ''
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

const handleSwitchVoiceTab = () => {
  activeTab.value = 'voice'
}

const handleCloseTranscriptSource = () => {
  closePlaybackBar()
  selectedTranscriptSource.value = null
  activeTab.value = 'voice'
}

const handleOpenMaterial = ({ fileId, node, materialId, material, recording, recordings = [] }) => {
  closePlaybackBar()
  isEmbeddedFolderOpen.value = false
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
    selectedTranscriptSource.value = null
  }

  emit('openStoredMaterial', materialId)
}

const handleOpenRecording = ({ fileId, node, recording }) => {
  isEmbeddedFolderOpen.value = false
  if (fileId && node) {
    emit('fileSelect', fileId, node)
  }

  selectedTranscriptSource.value = {
    title: recording?.title || '저장된 녹음',
    meta: formatTranscriptSourceDate(recording?.endedAt),
    transcriptions: recording?.transcriptions || []
  }
  if (recording) {
    setActivePlaybackRecording(recording, fileId)
  }
  emit('openRecording', {
    sessionId: fileId,
    recordingId: recording?.id || recording?.recordingId || '',
    recording
  })
  activeTab.value = 'voice'
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
    setActivePlaybackRecording(recording, request.node?.id || request.cite?.session_id || '')
  } else {
    selectedTranscriptSource.value = {
      title: request.cite.recording_title || request.cite.session_title || '출처 전사',
      meta: request.cite.session_date || '날짜 정보 없음',
      transcriptions: splitFullTranscript(request.cite.full_transcript, request.cite)
    }
    closePlaybackBar()
  }

  activeTab.value = 'voice'
})
</script>

<template>
  <aside
    :class="[{
      'sidebar-collapsed': isCollapsed && !embedded,
      'workspace-left-embedded': embedded,
      'has-script-tab-line': embedded && scriptTabLineVisible
    }]"
    id="sidebar"
    class="transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden rounded-[24px]"
    :style="embedded ? { width: '100%', flexShrink: 0 } : { width: isCollapsed ? '0px' : width + 'px', flexShrink: 0 }"
  >
    <div
      class="card workspace-sidebar-card h-full flex flex-col p-5 overflow-hidden min-w-[280px]"
      :class="{ 'is-embedded': embedded }"
    >
      <Teleport defer to="#workspace-unified-folder-drawer-host" :disabled="!embedded">
        <transition name="embedded-folder-drawer">
          <aside
            v-if="embedded && isEmbeddedFolderOpen"
            class="embedded-folder-drawer"
            aria-label="워크스페이스 폴더"
          >
            <div class="embedded-folder-drawer-header">
              <div class="embedded-folder-session-heading">
                <button
                  class="embedded-folder-session-home"
                  type="button"
                  aria-label="홈으로 이동"
                  title="홈으로 이동"
                  @click="emit('navigateHome')"
                >
                  <img class="embedded-folder-session-logo" src="/images/logo.png" alt="" draggable="false" />
                </button>
                <strong>{{ sidebarFileTitle }}</strong>
              </div>
              <button
                class="embedded-folder-drawer-close"
                type="button"
                aria-label="폴더 닫기"
                @click="isEmbeddedFolderOpen = false"
              >
                <span class="material-symbols-outlined">close</span>
              </button>
            </div>
            <FolderSideTab
              :fileTree="fileTree"
              :favorites="favorites"
              :active-file-id="activeFileId"
              :show-search="false"
              @update:fileTree="emit('update:fileTree', $event)"
              @update:favorites="emit('update:favorites', $event)"
              @fileSelect="(id, node) => { isEmbeddedFolderOpen = false; emit('fileSelect', id, node) }"
              @openMaterial="handleOpenMaterial"
              @openRecording="handleOpenRecording"
              @showToast="showToast"
            />
          </aside>
        </transition>
      </Teleport>

      <!-- Header -->
      <div class="workspace-file-header flex items-center justify-between mb-5">
        <div class="workspace-file-heading">
          <button
            class="workspace-file-back-btn"
            type="button"
            aria-label="홈으로 이동"
            title="홈으로 이동"
            @click="emit('navigateHome')"
          >
            <img class="workspace-file-back-logo" src="/images/logo.png" alt="" draggable="false" />
          </button>
          <input
            v-if="isEditingFileTitle"
            ref="fileTitleInputRef"
            v-model="fileTitleDraft"
            class="workspace-file-title-input collapsible-content"
            type="text"
            :disabled="isFileTitleSaving"
            aria-label="파일 제목 수정"
            @blur="commitFileTitleEdit"
            @keydown.enter="handleFileTitleEnter"
            @keydown.esc.prevent="cancelFileTitleEdit"
            @compositionstart="handleFileTitleCompositionStart"
            @compositionend="handleFileTitleCompositionEnd"
          />
          <button
            v-else
            class="workspace-file-title collapsible-content"
            type="button"
            :title="sidebarFileTitle"
            @click="startFileTitleEdit"
          >
            {{ sidebarFileTitle }}
          </button>
        </div>
        <div v-if="embedded || activeTab === 'voice'" class="sidebar-recording-control">
          <transition-group name="sidebar-recording-control" tag="div" class="sidebar-recording-inner">
            <button
              v-if="!isRecording"
              key="start"
              class="sidebar-recording-primary"
              type="button"
              @click="handleSidebarStartRecording"
            >
              녹음시작
            </button>
            <template v-else>
              <div
                key="voice-dots"
                class="sidebar-recording-voice-dots"
                :class="{ 'is-paused': isRecordingPaused }"
                aria-hidden="true"
              >
                <span
                  v-for="(_, dotIndex) in sidebarVoiceDotStyles"
                  :key="dotIndex"
                  class="sidebar-recording-voice-dot"
                  :style="sidebarVoiceDotStyles[dotIndex]"
                ></span>
              </div>
              <span key="time" class="sidebar-recording-time tabular-nums">
                {{ recordingTimeText }}
              </span>
              <button
                key="pause-toggle"
                class="sidebar-recording-icon"
                :class="{ 'is-paused': isRecordingPaused }"
                :aria-label="isRecordingPaused ? '녹음 재개' : '일시정지'"
                type="button"
                @click="isRecordingPaused ? emit('resumeRecording') : emit('pauseRecording')"
              >
                <span v-if="!isRecordingPaused" class="sidebar-recording-pause-bars" aria-hidden="true">
                  <span></span>
                  <span></span>
                </span>
                <span v-else class="sidebar-recording-play-triangle" aria-hidden="true"></span>
              </button>
              <button
                key="stop"
                class="sidebar-recording-primary is-stop"
                type="button"
                @click="emit('stopRecording')"
              >
                종료
              </button>
            </template>
          </transition-group>
        </div>
      </div>

      <!-- Tab Buttons -->
      <div
        v-if="!embedded"
        class="workspace-inset-shell p-1 rounded-[18px] flex gap-1.5 mb-3 collapsible-content"
      >
        <button
          class="workspace-inset-pill flex-1 py-2.5 rounded-[15px] text-[12px] font-bold text-gray-500"
          :class="{ 'is-active text-black': activeTab === 'voice' }"
          @click="handleSwitchVoiceTab"
        >스크립트</button>
        <button
          class="workspace-inset-pill flex-1 py-2.5 rounded-[15px] text-[12px] font-bold text-gray-500"
          :class="{ 'is-active text-black': activeTab === 'folders' }"
          @click="activeTab = 'folders'"
        >폴더</button>
      </div>

      <!-- Tab Content -->
      <div class="flex-1 flex flex-col overflow-hidden">
        <div
          v-show="!embedded && activeTab === 'folders'"
          class="flex flex-col flex-1 min-h-0 sidebar-content-animate"
        >
          <FolderSideTab
            :fileTree="fileTree"
            :favorites="favorites"
            :active-file-id="activeFileId"
            @update:fileTree="emit('update:fileTree', $event)"
            @update:favorites="emit('update:favorites', $event)"
            @fileSelect="(id, node) => emit('fileSelect', id, node)"
            @openMaterial="handleOpenMaterial"
            @openRecording="handleOpenRecording"
            @showToast="showToast"
          />
        </div>
        <div
          v-show="embedded || activeTab === 'voice'"
          class="flex flex-col flex-1 min-h-0 sidebar-content-animate"
          :class="{ 'has-sidebar-audio-player': activePlaybackRecording }"
        >
          <div v-if="selectedTranscriptSource && !activePlaybackRecording" class="selected-transcript-source">
            <div class="min-w-0">
              <span>{{ selectedTranscriptSource.meta }}</span>
            </div>
            <button type="button" title="스크립트로 돌아가기" @click="handleCloseTranscriptSource">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>
          <VoiceTransferSideTab
            :transcriptions="visibleTranscriptions"
            :recording-mode="recordingMode"
            :diarization-enabled="diarizationEnabled"
            :diarization-status="diarizationStatus"
            :variant="embedded ? 'content' : 'sidebar'"
            :show-toolbar="embedded"
            :show-folder-toggle="embedded"
            :folder-open="isEmbeddedFolderOpen"
            toolbar-title="스크립트"
            @addToNote="(text, source) => emit('addToNote', text, source)"
            @askAi="emit('askAi', $event)"
            @toggleFolder="isEmbeddedFolderOpen = !isEmbeddedFolderOpen"
          />
          <Teleport defer to="#workspace-unified-audio-player-host" :disabled="!embedded">
            <transition name="sidebar-audio-player">
              <section
                v-if="activePlaybackRecording"
                class="sidebar-audio-player"
                :class="{ 'is-unified-audio-player': embedded }"
                aria-label="녹음 재생바"
              >
                <audio
                  v-if="activePlaybackRecording.audioUrl"
                  ref="playbackAudioRef"
                  :src="activePlaybackRecording.audioUrl"
                  preload="metadata"
                  @loadedmetadata="handlePlaybackLoadedMetadata"
                  @timeupdate="handlePlaybackTimeUpdate"
                  @ended="handlePlaybackEnded"
                ></audio>
                <div class="sidebar-audio-source-header">
                  <div class="sidebar-audio-source-text">
                    <p>{{ playbackSourceTitle }}</p>
                    <span>{{ playbackSourceMeta }}</span>
                  </div>
                  <button
                    type="button"
                    class="sidebar-audio-close"
                    title="스크립트로 돌아가기"
                    aria-label="스크립트로 돌아가기"
                    @click="handleCloseTranscriptSource"
                  >
                    <span class="material-symbols-outlined">close</span>
                  </button>
                </div>

                <div class="sidebar-audio-track-row">
                  <span>{{ formatPlaybackTime(playbackCurrentSeconds) }}</span>
                  <input
                    v-model.number="playbackProgress"
                    class="sidebar-audio-range"
                    :style="{ '--progress': `${playbackProgress}%` }"
                    type="range"
                    min="0"
                    max="100"
                    step="0.1"
                    aria-label="녹음 재생 위치"
                    @input="handlePlaybackRangeInput"
                  />
                  <span>{{ formatPlaybackTime(playbackDurationSeconds) }}</span>
                </div>

                <div class="sidebar-audio-actions">
                  <button type="button" aria-label="5초 뒤로" @click="skipPlayback(-5)">
                    <span class="material-symbols-outlined">replay_5</span>
                  </button>
                  <button
                    type="button"
                    class="sidebar-audio-play-compact"
                    :aria-label="isPlaybackPlaying ? '일시정지' : '재생'"
                    @click="togglePlayback"
                  >
                    <span class="material-symbols-outlined">
                      {{ isPlaybackPlaying ? 'pause' : 'play_arrow' }}
                    </span>
                  </button>
                  <button type="button" class="sidebar-audio-speed" @click="cyclePlaybackSpeed">
                    {{ playbackSpeed }}x
                  </button>
                  <button type="button" aria-label="5초 앞으로" @click="skipPlayback(5)">
                    <span class="material-symbols-outlined">forward_5</span>
                  </button>
                </div>
              </section>
            </transition>
          </Teleport>
        </div>
      </div>

    </div>
  </aside>

  <!-- Resizer -->
  <div
    v-if="!embedded"
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
  position: relative;
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

.workspace-left-embedded {
  position: relative;
  min-width: 0;
  overflow: visible !important;
  border-radius: 0 !important;
  border-right: 2px solid rgba(226, 224, 232, 0.9);
  background: rgba(255, 255, 255, 0.72);
}

.workspace-left-embedded::after {
  content: '';
  position: absolute;
  top: 82px;
  left: 0;
  right: 2px;
  z-index: 60;
  display: none;
  height: 1px;
  background: rgba(0, 0, 0, 0.06);
  pointer-events: none;
}

.workspace-left-embedded.has-script-tab-line::after {
  display: block;
}

.workspace-sidebar-card.is-embedded {
  position: static;
  padding: 10px 22px 18px 32px !important;
  min-width: 0;
  overflow: visible !important;
  border: 0;
  border-radius: 0;
  background: #ffffff;
  box-shadow: none;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}

.workspace-sidebar-card.is-embedded::before,
.workspace-sidebar-card.is-embedded::after {
  display: none;
}

.embedded-folder-drawer {
  position: absolute;
  inset: 0 auto 0 0;
  z-index: 40;
  width: min(360px, calc(100% - 42px));
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 20px 18px 18px;
  background: rgba(255, 255, 255, 0.98);
  border-right: 1px solid rgba(226, 224, 232, 0.9);
  box-shadow: 24px 0 48px rgba(48, 42, 58, 0.12);
  backdrop-filter: blur(18px) saturate(145%);
  -webkit-backdrop-filter: blur(18px) saturate(145%);
}

.embedded-folder-drawer-header {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.embedded-folder-session-heading {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.embedded-folder-session-home {
  width: 42px;
  height: 42px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 999px;
  background: transparent;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.embedded-folder-session-home:hover {
  box-shadow: 0 0 0 4px rgba(29, 29, 31, 0.06);
}

.embedded-folder-session-home:active {
  transform: scale(0.96);
}

.embedded-folder-session-logo {
  width: 42px;
  height: 42px;
  flex: 0 0 auto;
  border-radius: 50%;
  object-fit: cover;
  filter: drop-shadow(0 10px 18px rgba(0, 0, 0, 0.12));
}

.embedded-folder-session-heading strong {
  overflow: hidden;
  color: #15161a;
  font-size: 18px;
  font-weight: 950;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.embedded-folder-drawer-close {
  width: 32px;
  height: 32px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #8e8e93;
  background: rgba(239, 237, 244, 0.88);
  transition: color 0.18s ease, background-color 0.18s ease, transform 0.18s ease;
}

.embedded-folder-drawer-close:hover {
  color: #15161a;
  background: rgba(229, 226, 235, 0.94);
}

.embedded-folder-drawer-close:active {
  transform: scale(0.96);
}

.embedded-folder-drawer-close .material-symbols-outlined {
  font-size: 18px;
}

.embedded-folder-drawer-enter-active,
.embedded-folder-drawer-leave-active {
  transition: opacity 0.22s ease, transform 0.24s cubic-bezier(0.22, 1, 0.36, 1);
}

.embedded-folder-drawer-enter-from,
.embedded-folder-drawer-leave-to {
  opacity: 0;
  transform: translateX(-18px);
}

.workspace-sidebar-card.is-embedded .workspace-file-header {
  margin-bottom: 6px !important;
}

.workspace-sidebar-card.is-embedded .workspace-file-heading {
  margin-left: 0;
}

.workspace-sidebar-card.is-embedded .has-sidebar-audio-player {
  padding-bottom: 76px;
}

.workspace-file-heading {
  min-width: 0;
  flex: 1 1 auto;
  display: flex;
  align-items: center;
  gap: 10px;
  padding-right: 12px;
  font-weight: 900;
  letter-spacing: 0;
}

.workspace-file-back-btn {
  width: 32px;
  height: 32px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  background: transparent;
  color: #15161a;
  cursor: pointer;
  transition: transform 0.18s ease, background-color 0.18s ease;
}

.workspace-file-back-btn:hover {
  background: rgba(29, 29, 31, 0.06);
}

.workspace-file-back-btn:active {
  transform: scale(0.96);
}

.workspace-file-back-logo {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
  border-radius: 999px;
  user-select: none;
  pointer-events: none;
}

.workspace-file-title {
  flex: 1 1 auto;
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  padding: 0;
  border: 0;
  background: transparent;
  color: #15161a;
  cursor: text;
  font-size: 15px;
  font-weight: 850;
  line-height: 1.2;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: color 0.16s ease;
}

.workspace-file-title:hover {
  color: #3b404a;
}

.workspace-file-title-input {
  flex: 1 1 auto;
  min-width: 0;
  height: 31px;
  padding: 0 2px 2px;
  border: 0;
  border-radius: 0;
  outline: none !important;
  appearance: none;
  -webkit-appearance: none;
  box-shadow: none !important;
  background: transparent;
  color: #15161a;
  font-size: 15px;
  font-weight: 850;
  line-height: 1.2;
}

.workspace-file-title-input:focus,
.workspace-file-title-input:focus-visible {
  border-color: transparent;
  outline: none !important;
  box-shadow: none !important;
}

.workspace-file-title-input:disabled {
  opacity: 0.66;
}

.sidebar-recording-control {
  flex: 0 0 auto;
  display: flex;
  justify-content: flex-end;
  min-width: 96px;
}

.sidebar-recording-inner {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  min-width: 0;
  white-space: nowrap;
}

.sidebar-recording-primary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 82px;
  height: 34px;
  padding: 0 13px;
  border-radius: 999px;
  background: #111111;
  color: #ffffff;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: -0.01em;
  box-shadow: none;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.sidebar-recording-primary:hover {
  background: #1f1f1f;
}

.sidebar-recording-primary:active,
.sidebar-recording-icon:active {
  transform: scale(0.98);
}

.sidebar-recording-primary.is-stop {
  min-width: 48px;
  padding: 0 12px;
}

.sidebar-recording-time {
  color: #1d1d1f;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.01em;
}

.sidebar-recording-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 999px;
  background: #e5e5ea;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.sidebar-recording-icon:hover {
  background: #dbdbe2;
}

.sidebar-recording-icon.is-paused {
  background: #fff0f1;
}

.sidebar-recording-pause-bars {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.sidebar-recording-pause-bars span {
  display: block;
  width: 4px;
  height: 14px;
  border-radius: 999px;
  background: #5f6472;
}

.sidebar-recording-play-triangle {
  width: 0;
  height: 0;
  margin-left: 2px;
  border-top: 7px solid transparent;
  border-bottom: 7px solid transparent;
  border-left: 11px solid #ef4444;
}

.sidebar-recording-voice-dots {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  height: 20px;
  padding: 0 4px;
}

.sidebar-recording-voice-dot {
  display: block;
  width: 3px;
  height: 16px;
  border-radius: 999px;
  background: linear-gradient(180deg, #111111, #5f6472);
  transform-origin: center;
  transition: transform 0.12s linear, opacity 0.12s linear;
}

.sidebar-recording-voice-dots.is-paused .sidebar-recording-voice-dot {
  opacity: 0.35 !important;
  transform: scaleY(0.45) !important;
}

.sidebar-recording-control-enter-active,
.sidebar-recording-control-leave-active {
  transition: opacity 0.22s ease, transform 0.26s cubic-bezier(0.22, 1, 0.36, 1);
}

.sidebar-recording-control-move {
  transition: transform 0.26s cubic-bezier(0.22, 1, 0.36, 1);
}

.sidebar-recording-control-enter-from {
  opacity: 0;
  transform: translateY(6px) scale(0.96);
}

.sidebar-recording-control-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.96);
}

.sidebar-recording-control-leave-active {
  position: absolute;
}

.selected-transcript-source {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  padding: 8px 9px 8px 11px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(226, 224, 232, 0.9);
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
  background: rgba(239, 237, 244, 0.9);
}

.selected-transcript-source button .material-symbols-outlined {
  font-size: 14px;
}

.sidebar-audio-player {
  flex: 0 0 auto;
  width: 100%;
  margin-top: 12px;
  padding: 12px;
  color: #15161a;
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(248, 246, 251, 0.94));
  border: 1px solid rgba(226, 224, 232, 0.92);
  box-shadow: none;
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
}

.sidebar-audio-source-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.sidebar-audio-source-text {
  min-width: 0;
}

.sidebar-audio-source-text p {
  margin: 0;
  overflow: hidden;
  color: #15161a;
  font-size: 12px;
  font-weight: 950;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sidebar-audio-source-text span {
  display: block;
  margin-top: 2px;
  overflow: hidden;
  color: #8b8794;
  font-size: 10.5px;
  font-weight: 850;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sidebar-audio-close {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 999px;
  color: #8e8e93;
  background: rgba(239, 237, 244, 0.9);
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.sidebar-audio-close:hover {
  color: #15161a;
  background: rgba(229, 226, 235, 0.98);
}

.sidebar-audio-close:active {
  transform: scale(0.96);
}

.sidebar-audio-close .material-symbols-outlined {
  font-size: 16px;
}

.sidebar-audio-track-row {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) 34px;
  align-items: center;
  gap: 8px;
}

.sidebar-audio-track-row span {
  color: #8b8794;
  font-size: 10.5px;
  font-weight: 850;
  font-variant-numeric: tabular-nums;
}

.sidebar-audio-track-row span:last-child {
  text-align: right;
}

.sidebar-audio-range {
  width: 100%;
  height: 5px;
  border-radius: 999px;
  appearance: none;
  background: linear-gradient(90deg, #15161a 0%, #15161a var(--progress, 0%), rgba(230, 227, 236, 0.92) var(--progress, 0%), rgba(230, 227, 236, 0.92) 100%);
  cursor: pointer;
  outline: none;
}

.sidebar-audio-range::-webkit-slider-thumb {
  width: 14px;
  height: 14px;
  appearance: none;
  border-radius: 999px;
  background: #ffffff;
  border: 3px solid #15161a;
  box-shadow: none;
}

.sidebar-audio-range::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border: 3px solid #15161a;
  border-radius: 999px;
  background: #ffffff;
  box-shadow: none;
}

.sidebar-audio-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 10px;
}

.sidebar-audio-actions button {
  min-width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #5f5b66;
  background: rgba(245, 243, 248, 0.88);
  border: 1px solid rgba(229, 226, 235, 0.9);
  font-size: 10.5px;
  font-weight: 950;
  transition: transform 0.18s ease, background-color 0.18s ease, color 0.18s ease;
}

.sidebar-audio-actions button:hover {
  color: #15161a;
  background: rgba(255, 255, 255, 0.98);
}

.sidebar-audio-actions .material-symbols-outlined {
  font-size: 17px;
}

.sidebar-audio-actions .sidebar-audio-play-compact {
  color: #ffffff;
  background: #15161a;
  border-color: #15161a;
  box-shadow: none;
}

.sidebar-audio-actions .sidebar-audio-play-compact:hover {
  color: #ffffff;
  background: #22242a;
}

.sidebar-audio-actions .sidebar-audio-play-compact .material-symbols-outlined {
  font-size: 19px;
  font-variation-settings: 'FILL' 1;
}

.sidebar-audio-speed {
  padding: 0 10px;
}

.sidebar-audio-player.is-unified-audio-player {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 70;
  width: auto;
  height: 64px;
  min-height: 64px;
  display: block;
  margin: 0;
  padding: 0 18px 8px;
  border: 0;
  border-top: 1px solid rgba(226, 224, 232, 0.92);
  border-radius: 0;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 -10px 24px rgba(48, 42, 58, 0.08);
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-source-header {
  display: none;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-track-row {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  min-width: 0;
  height: 30px;
  grid-template-columns: 54px minmax(0, 1fr) 54px;
  gap: 8px;
  padding: 0 18px;
  align-items: start;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-track-row span {
  padding-top: 14px;
  font-size: 10px;
  font-weight: 850;
  color: #8d93a1;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-range {
  height: 8px;
  margin-top: 1px;
  background: linear-gradient(90deg, #2f7df6 0%, #2f7df6 var(--progress, 0%), rgba(230, 234, 241, 0.96) var(--progress, 0%), rgba(230, 234, 241, 0.96) 100%);
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-range::-webkit-slider-thumb {
  width: 8px;
  height: 18px;
  border-width: 0;
  border-radius: 999px;
  background: #2f7df6;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-range::-moz-range-thumb {
  width: 8px;
  height: 18px;
  border-width: 0;
  border-radius: 999px;
  background: #2f7df6;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-actions {
  position: absolute;
  left: 50%;
  bottom: 8px;
  transform: translateX(-50%);
  justify-content: center;
  gap: 16px;
  margin-top: 0;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-actions button {
  min-width: 32px;
  width: 32px;
  height: 32px;
  color: #6f7582;
  background: transparent;
  border: 0;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-actions .sidebar-audio-play-compact {
  width: 34px;
  height: 34px;
  color: #15161a;
  background: transparent;
  border: 0;
  box-shadow: none;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-actions .sidebar-audio-play-compact:hover {
  color: #15161a;
  background: transparent;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-actions .sidebar-audio-play-compact .material-symbols-outlined {
  font-size: 28px;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-actions .material-symbols-outlined {
  font-size: 18px;
}

.sidebar-audio-player.is-unified-audio-player .sidebar-audio-speed {
  order: 4;
  width: auto;
  min-width: 30px;
  padding: 0 2px;
  color: #15161a;
}

@media (max-width: 1280px) {
  .sidebar-audio-player.is-unified-audio-player {
    height: 66px;
  }

  .sidebar-audio-player.is-unified-audio-player .sidebar-audio-track-row {
    grid-template-columns: 50px minmax(0, 1fr) 50px;
  }
}

.sidebar-audio-player-enter-active,
.sidebar-audio-player-leave-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
}

.sidebar-audio-player-enter-from,
.sidebar-audio-player-leave-to {
  opacity: 0;
  transform: translateY(10px) scale(0.98);
}
</style>
