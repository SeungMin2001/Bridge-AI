import { onUnmounted, ref } from 'vue'
import { useAiState } from './appState/aiState'
import {
  ensureLectureOneFile,
  updateNodeById,
  useFileTreeState
} from './appState/fileTreeState'
import { useMaterialsState } from './appState/materialsState'
import { useRecordingState } from './appState/recordingState'

export function useAppState() {
  const isRightSidebarVisible = ref(true)

  const {
    fileTree,
    favorites,
    activeFileName,
    activeFileId,
    currentAttachments,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleFileSelect
  } = useFileTreeState()

  const {
    transcriptions,
    isRecording,
    isRecordingPaused,
    recordingTimeText,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording
  } = useRecordingState()

  const {
    aiInput,
    summaryNotes,
    handleRightSidebarToggle,
    handleAddToNote,
    handleAskAi,
    handleAiInputUpdate
  } = useAiState({ isRightSidebarVisible })

  const {
    currentPreviewMaterial,
    handleUploadLectureMaterials,
    handleClosePreviewMaterial,
    handleOpenStoredMaterial,
    handleDeleteStoredMaterial
  } = useMaterialsState({
    fileTree,
    activeFileId,
    activeFileName,
    currentAttachments,
    ensureLectureOneFile,
    updateNodeById
  })

  onUnmounted(() => {
    stopRecording()
  })

  return {
    fileTree,
    favorites,
    transcriptions,
    isRecording,
    isRecordingPaused,
    recordingTimeText,
    activeFileName,
    activeFileId,
    currentAttachments,
    currentPreviewMaterial,
    isRightSidebarVisible,
    summaryNotes,
    aiInput,
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
    handleOpenStoredMaterial,
    handleDeleteStoredMaterial
  }
}
