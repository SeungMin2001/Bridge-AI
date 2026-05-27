<!-- 애플리케이션의 루트 컴포넌트로, 현재 경로에 맞는 페이지를 렌더링합니다. -->
<script setup>
import { ref } from 'vue'
import Workspace from './pages/Workspace/Workspace.vue'
import Home from './pages/Home/Home.vue'
import Workfolder from './pages/Workfolder/Workfolder.vue'
import AiHistory from './pages/AiHistory/AiHistory.vue'
import ScheduleManagement from './pages/Schedule/ScheduleManagement.vue'
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
  confirmAndSyncToNotion,
  ignoreSchedule,
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

const scheduleWorkspaceRequest = ref(null)

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

function handleScheduleWorkspaceOpen(item = {}) {
  const sessionId = item.workspaceFileId || item.sessionId || item.session_id || ''
  const node = sessionId ? findNodeById(fileTree.value, sessionId) : null

  if (sessionId && node) {
    handleFileSelect(sessionId, node)
  }

  scheduleWorkspaceRequest.value = {
    id: `${item.id || item.apiId || sessionId || 'schedule'}-${Date.now()}`,
    scheduleId: item.id || item.apiId || '',
    sessionId,
    recordingId: item.recordingId || item.recording_id || '',
    transcriptId: item.transcriptId || item.transcript_id || '',
    sourceStartTime: item.sourceStartTime ?? item.source_start_time ?? null,
    sourceEndTime: item.sourceEndTime ?? item.source_end_time ?? null,
    sourceText: item.sourceText || item.source_text || ''
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
    @open-workspace-source="handleScheduleWorkspaceOpen"
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
    :scheduleWorkspaceRequest="scheduleWorkspaceRequest"
    :summaryState="summaryState"
    :summaryNotes="summaryNotes"
    :aiInput="aiInput"
    @update:fileTree="handleFileTreeUpdate"
    @update:favorites="handleFavoritesUpdate"
    @update:aiInput="handleAiInputUpdate"
    @navigateHome="navigateTo('home')"
    @navigate="navigateTo"
    @fileSelect="handleFileSelect"
    @dismissScheduleNotice="dismissScheduleExtractionNotice"
    @confirmAndSyncSchedule="confirmAndSyncToNotion"
    @ignoreSchedule="ignoreSchedule"
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
