import { onUnmounted, ref } from 'vue'
import { useAiState } from './appState/aiState'
import {
  ensureLectureOneFile,
  updateNodeById,
  useFileTreeState
} from './appState/fileTreeState'
import { useMaterialsState } from './appState/materialsState'
import { useRecordingState } from './appState/recordingState'

// 앱 전체에서 공유하는 상태 모듈들을 하나로 묶어 App.vue에 전달합니다.
export function useAppState() {
  const isRightSidebarVisible = ref(true)

  // 파일 트리, 즐겨찾기, 현재 선택 파일 상태입니다.
  const {
    fileTree,
    favorites,
    activeFileName,
    activeFileId,
    activeFileType,
    currentAttachments,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleFileSelect
  } = useFileTreeState()

  // 녹음 상태, 전사 목록, 녹음 제어 함수입니다.
  const {
    transcriptions,
    isRecording,
    isRecordingPaused,
    recordingMode,
    recordingTimeText,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording
  } = useRecordingState()

  // AI 입력, 정리 노트, 우측 사이드바 상태입니다.
  const {
    aiInput,
    summaryNotes,
    handleRightSidebarToggle,
    handleAddToNote,
    handleAskAi,
    handleAiInputUpdate
  } = useAiState({ isRightSidebarVisible })

  // PDF/PPT 강의자료 첨부와 현재 미리보기 상태입니다.
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

  // 앱이 내려갈 때 마이크/WebSocket 등 녹음 리소스를 정리합니다.
  onUnmounted(() => {
    stopRecording()
  })

  return {
    fileTree,
    favorites,
    transcriptions,
    isRecording,
    isRecordingPaused,
    recordingMode,
    recordingTimeText,
    activeFileName,
    activeFileId,
    activeFileType,
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
