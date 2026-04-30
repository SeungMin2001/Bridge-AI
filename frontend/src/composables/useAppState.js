import { onUnmounted, ref } from 'vue'
import { isWorkspaceUuid, saveSessionResources } from '../api/workspaceApi.js'
import { useAiState } from './appState/aiState'
import {
  addRecordingToCurrentWeek,
  findNodeById,
  normalizeFileTree,
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
    currentRecordings,
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
    stopRecording: stopActiveRecording
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
    handleOpenStoredMaterial
  } = useMaterialsState({
    fileTree,
    activeFileId,
    activeFileName,
    currentAttachments,
    normalizeFileTree,
    updateNodeById
  })

  const createLocalId = (prefix) => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return `${prefix}-${crypto.randomUUID()}`
    }
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }

  const getMaterialTitle = (materialName = '') => {
    return materialName
      .replace(/\.(pdf|ppt|pptx)$/i, '')
      .replace(/\s+/g, ' ')
      .trim()
  }

  const getNextMaterialRecordingIndex = (materialId) => {
    if (!materialId) return 1
    return currentRecordings.value.filter((recording) => (
      Array.isArray(recording.materialIds) && recording.materialIds.includes(materialId)
    )).length + 1
  }

  const formatRecordingTitle = (material = null) => {
    if (material?.id) {
      const materialTitle = getMaterialTitle(material.name) || '강의자료'
      const recordingIndex = getNextMaterialRecordingIndex(material.id)
      return `${materialTitle} 녹음 ${recordingIndex}`
    }

    const now = new Date()
    const period = now.getHours() >= 12 ? '오후' : '오전'
    const hour = now.getHours() % 12 || 12
    const minute = now.getMinutes().toString().padStart(2, '0')
    return `${activeFileName.value || '녹음'} ${period} ${hour}:${minute}`
  }

  const cloneTranscriptions = () => {
    return JSON.parse(JSON.stringify(transcriptions.value || []))
  }

  const handleStartRecording = (mode = 'lecture') => {
    return startRecording(mode, activeFileId.value)
  }

  const handleStopRecording = async () => {
    const shouldSaveRecording = isRecording.value
    const recordingSnapshot = cloneTranscriptions()
    const durationText = recordingTimeText.value
    const mode = recordingMode.value
    const linkedMaterialId = currentPreviewMaterial.value?.id || null
    const linkedMaterialName = currentPreviewMaterial.value?.name || ''

    stopActiveRecording()

    if (!shouldSaveRecording || recordingSnapshot.length === 0) return

    const targetFileId = activeFileId.value
    if (!targetFileId) return

    const recording = {
      id: createLocalId('recording'),
      title: formatRecordingTitle(currentPreviewMaterial.value),
      endedAt: new Date().toISOString(),
      durationText,
      recordingMode: mode,
      materialIds: linkedMaterialId ? [linkedMaterialId] : [],
      materialNames: linkedMaterialName ? [linkedMaterialName] : [],
      audioUrl: null,
      transcriptions: recordingSnapshot
    }

    fileTree.value = updateNodeById(
      normalizeFileTree(fileTree.value),
      targetFileId,
      (node) => addRecordingToCurrentWeek(node, recording)
    )

    const updatedNode = findNodeById(fileTree.value, targetFileId)
    if (isWorkspaceUuid(targetFileId) && Array.isArray(updatedNode?.weeks)) {
      try {
        await saveSessionResources(targetFileId, updatedNode.weeks)
      } catch (error) {
        console.error('[workspace] session resources save failed:', error)
      }
    }
  }

  // 앱이 내려갈 때 마이크/WebSocket 등 녹음 리소스를 정리합니다.
  onUnmounted(() => {
    stopActiveRecording()
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
    currentRecordings,
    currentPreviewMaterial,
    isRightSidebarVisible,
    summaryNotes,
    aiInput,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleAiInputUpdate,
    handleFileSelect,
    startRecording: handleStartRecording,
    pauseRecording,
    resumeRecording,
    stopRecording: handleStopRecording,
    handleRightSidebarToggle,
    handleAddToNote,
    handleAskAi,
    handleUploadLectureMaterials,
    handleClosePreviewMaterial,
    handleOpenStoredMaterial
  }
}
