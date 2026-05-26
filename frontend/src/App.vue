<!-- 애플리케이션의 루트 컴포넌트로, 현재 경로에 맞는 페이지를 렌더링합니다. -->
<script setup>
import { ref } from 'vue'
import Workspace from './pages/Workspace/Workspace.vue'
import Home from './pages/Home/Home.vue'
import Workfolder from './pages/Workfolder/Workfolder.vue'
import AiHistory from './pages/AiHistory/AiHistory.vue'
import ScheduleManagement from './pages/Schedule/ScheduleManagement.vue'
import { collectFileRecordings, findNodeById } from './composables/appState/fileTreeState'
import { useAppState } from './composables/useAppState'
import { usePageNavigation } from './composables/usePageNavigation'

const { currentView, navigateTo } = usePageNavigation()
const {
  fileTree,
  favorites,
  recentFiles,
  transcriptions,
  isRecording,
  isRecordingPaused,
  recordingMode,
  recordingTimeText,
  recordingAudioLevel,
  diarizationEnabled,
  diarizationStatus,
  activeFileName,
  activeFileId,
  activeFileType,
  currentAttachments,
  currentRecordings,
  currentPreviewMaterial,
  isRightSidebarVisible,
  scheduleExtractionNotice,
  summaryState,
  summaryNotes,
  aiInput,
  dismissScheduleExtractionNotice,
  handleFileTreeUpdate,
  handleFavoritesUpdate,
  handleAiInputUpdate,
  handleFileSelect,
  startRecording,
  pauseRecording,
  resumeRecording,
  stopRecording,
  generateMaterialSummaryForSource,
  generateRecordingSummaryForSource,
  deleteSummary,
  handleRightSidebarToggle,
  handleAddToNote,
  handleAskAi,
  handleUploadLectureMaterials,
  handleUploadRecordingFile,
  handleClosePreviewMaterial,
  handleOpenStoredMaterial,
  handleOpenRecording
} = useAppState()

const workspaceSourceOpenRequest = ref(null)

const getScheduleFileId = (item = {}) => (
  item.workspaceFileId || item.session_id || item.sessionId || ''
)

const getRecordingId = (recording = {}) => (
  String(recording?.id || recording?.recordingId || '')
)

const getTranscriptIdsFromRecording = (recording = {}) => {
  const ids = new Set()
  const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
  transcriptions.forEach((transcription) => {
    const transcriptionIds = [
      transcription?.transcript_id,
      transcription?.transcriptId
    ]
    transcriptionIds.filter(Boolean).forEach((id) => ids.add(String(id)))

    const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
    segments.forEach((segment) => {
      const segmentIds = [
        segment?.transcript_id,
        segment?.transcriptId
      ]
      segmentIds.filter(Boolean).forEach((id) => ids.add(String(id)))
    })
  })
  return ids
}

const getRecordingTranscriptText = (recording = {}) => {
  const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
  return transcriptions
    .flatMap((transcription) => {
      const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
      return [
        transcription?.text,
        ...segments.map((segment) => segment?.text)
      ]
    })
    .filter(Boolean)
    .join('\n')
}

const getRecordingTimeRange = (recording = {}) => {
  const starts = []
  const ends = []
  const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []

  transcriptions.forEach((transcription) => {
    const start = Number(transcription?.start ?? transcription?.start_time ?? transcription?.startTime)
    const end = Number(transcription?.end ?? transcription?.end_time ?? transcription?.endTime)
    if (Number.isFinite(start)) starts.push(start)
    if (Number.isFinite(end)) ends.push(end)

    const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
    segments.forEach((segment) => {
      const segmentStart = Number(segment?.start ?? segment?.start_time ?? segment?.startTime)
      const segmentEnd = Number(segment?.end ?? segment?.end_time ?? segment?.endTime)
      if (Number.isFinite(segmentStart)) starts.push(segmentStart)
      if (Number.isFinite(segmentEnd)) ends.push(segmentEnd)
    })
  })

  if (!starts.length) return null
  return {
    start: Math.min(...starts),
    end: ends.length ? Math.max(...ends) : Math.min(...starts)
  }
}

const findScheduleRecording = (node, item = {}) => {
  const recordings = collectFileRecordings(node)
  if (!recordings.length) return null

  const targetRecordingId = String(item.recordingId || item.recording_id || '').trim()
  if (targetRecordingId) {
    const byRecordingId = recordings.find((recording) => getRecordingId(recording) === targetRecordingId)
    if (byRecordingId) return byRecordingId
  }

  const transcriptId = String(item.transcriptId || item.transcript_id || '').trim()
  if (transcriptId) {
    const byTranscriptId = recordings.find((recording) => (
      getTranscriptIdsFromRecording(recording).has(transcriptId)
    ))
    if (byTranscriptId) return byTranscriptId
  }

  const sourceText = String(item.sourceText || item.source_text || '').trim()
  if (sourceText) {
    const normalizedSourceText = sourceText.replace(/\s+/g, ' ')
    const bySourceText = recordings.find((recording) => (
      getRecordingTranscriptText(recording).replace(/\s+/g, ' ').includes(normalizedSourceText)
    ))
    if (bySourceText) return bySourceText
  }

  const sourceStartTime = Number(item.sourceStartTime ?? item.source_start_time)
  if (Number.isFinite(sourceStartTime)) {
    const byTimeRange = recordings.find((recording) => {
      const range = getRecordingTimeRange(recording)
      return range && sourceStartTime >= range.start - 0.5 && sourceStartTime <= range.end + 0.5
    })
    if (byTimeRange) return byTimeRange
  }

  return recordings.length === 1 ? recordings[0] : null
}

const handleScheduleWorkspaceOpen = (item = {}) => {
  const fileId = getScheduleFileId(item)
  const node = fileId ? findNodeById(fileTree.value, fileId) : null

  if (fileId && node) {
    handleFileSelect(fileId, node)
  }

  const recording = node ? findScheduleRecording(node, item) : null
  workspaceSourceOpenRequest.value = recording
    ? {
        id: `schedule-recording-${item.id || fileId}-${Date.now()}`,
        type: 'recording',
        fileId,
        node,
        recording,
        recordingId: getRecordingId(recording),
        sourceStartTime: item.sourceStartTime ?? item.source_start_time,
        sourceEndTime: item.sourceEndTime ?? item.source_end_time
      }
    : {
        id: `schedule-file-${item.id || fileId || Date.now()}-${Date.now()}`,
        type: 'file',
        fileId,
        node
      }

  navigateTo('workspace')
}
</script>

<template>
  <AiHistory 
    v-if="currentView === 'ai-history'" 
    @navigateBack="navigateTo('home')" 
  />

  <Workfolder 
    v-else-if="currentView === 'workfolder'"
    :fileTree="fileTree"
    :favorites="favorites"
    @update:fileTree="handleFileTreeUpdate"
    @update:favorites="handleFavoritesUpdate"
    @fileSelect="handleFileSelect"
    @navigate="navigateTo"
  />

  <Home 
    v-else-if="currentView === 'home'"
    :fileTree="fileTree"
    :favorites="favorites"
    :recentFiles="recentFiles"
    @update:fileTree="handleFileTreeUpdate"
    @update:favorites="handleFavoritesUpdate"
    @fileSelect="handleFileSelect"
    @navigate="navigateTo"
  />

  <ScheduleManagement
    v-else-if="currentView === 'schedule'"
    :fileTree="fileTree"
    :favorites="favorites"
    @navigate="navigateTo"
    @openWorkspace="handleScheduleWorkspaceOpen"
  />

  <Workspace 
    v-else
    :fileTree="fileTree"
    :favorites="favorites"
    :transcriptions="transcriptions"
    :isRecording="isRecording"
    :isRecordingPaused="isRecordingPaused"
    :recordingMode="recordingMode"
    :recordingTimeText="recordingTimeText"
    :recordingAudioLevel="recordingAudioLevel"
    :diarizationEnabled="diarizationEnabled"
    :diarizationStatus="diarizationStatus"
    :activeFileName="activeFileName"
    :activeFileId="activeFileId"
    :activeFileType="activeFileType"
    :currentAttachments="currentAttachments"
    :currentRecordings="currentRecordings"
    :currentPreviewMaterial="currentPreviewMaterial"
    :isRightSidebarVisible="isRightSidebarVisible"
    :scheduleExtractionNotice="scheduleExtractionNotice"
    :summaryState="summaryState"
    :summaryNotes="summaryNotes"
    :aiInput="aiInput"
    :sourceOpenRequest="workspaceSourceOpenRequest"
    @update:fileTree="handleFileTreeUpdate"
    @update:favorites="handleFavoritesUpdate"
    @update:aiInput="handleAiInputUpdate"
    @navigateHome="navigateTo('home')"
    @navigate="navigateTo"
    @fileSelect="handleFileSelect"
    @dismissScheduleNotice="dismissScheduleExtractionNotice"
    @startRecording="startRecording"
    @pauseRecording="pauseRecording"
    @resumeRecording="resumeRecording"
    @stopRecording="stopRecording"
    @generateMaterialSummary="generateMaterialSummaryForSource"
    @generateRecordingSummary="generateRecordingSummaryForSource"
    @deleteSummary="deleteSummary"
    @rightSidebarToggle="handleRightSidebarToggle"
    @addToNote="handleAddToNote"
    @askAi="handleAskAi"
    @uploadLectureMaterials="handleUploadLectureMaterials"
    @uploadRecordingFile="handleUploadRecordingFile"
    @closePreviewMaterial="handleClosePreviewMaterial"
    @openStoredMaterial="handleOpenStoredMaterial"
    @openRecording="handleOpenRecording"
  />
</template>
