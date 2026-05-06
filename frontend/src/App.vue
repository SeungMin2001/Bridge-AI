<!-- 애플리케이션의 루트 컴포넌트로, 현재 경로에 맞는 페이지를 렌더링합니다. -->
<script setup>
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
  activeFileName,
  activeFileId,
  activeFileType,
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
  handleRightSidebarToggle,
  handleAddToNote,
  handleAskAi,
  handleUploadLectureMaterials,
  handleClosePreviewMaterial,
  handleOpenStoredMaterial
} = useAppState()
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
    @navigate="navigateTo"
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
    :activeFileName="activeFileName"
    :activeFileId="activeFileId"
    :activeFileType="activeFileType"
    :currentPreviewMaterial="currentPreviewMaterial"
    :isRightSidebarVisible="isRightSidebarVisible"
    :scheduleExtractionNotice="scheduleExtractionNotice"
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
    @startRecording="startRecording"
    @pauseRecording="pauseRecording"
    @resumeRecording="resumeRecording"
    @stopRecording="stopRecording"
    @rightSidebarToggle="handleRightSidebarToggle"
    @addToNote="handleAddToNote"
    @askAi="handleAskAi"
    @uploadLectureMaterials="handleUploadLectureMaterials"
    @closePreviewMaterial="handleClosePreviewMaterial"
    @openStoredMaterial="handleOpenStoredMaterial"
  />
</template>
