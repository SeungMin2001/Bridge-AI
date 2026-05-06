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
import { useScheduleState } from './useScheduleState'
import { useSummaryState } from './useSummaryState'

// 앱 전체에서 공유하는 상태 모듈들을 하나로 묶어 App.vue에 전달합니다.
export function useAppState() {
  const isRightSidebarVisible = ref(true)
  const scheduleExtractionNotice = ref(null)

  // 파일 트리, 즐겨찾기, 현재 선택 파일 상태입니다.
  const {
    fileTree,
    favorites,
    recentFiles,
    activeFileName,
    activeFileId,
    activeFileType,
    currentAttachments,
    currentRecordings,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleFileSelect: originHandleFileSelect
  } = useFileTreeState()

  // 녹음 상태, 전사 목록, 녹음 제어 함수입니다.
  const {
    transcriptions,
    isRecording,
    isRecordingPaused,
    recordingMode,
    recordingTimeText,
    activeRecordingId,
    activeRecordingStartedAt,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording: stopActiveRecording
  } = useRecordingState()

  const {
    hydrateSchedules
  } = useScheduleState()

  const {
    summaryState,
    clearSummaryState,
    startLiveSummary,
    loadSummariesForSession,
    generateSummariesForSession
  } = useSummaryState()

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

  const handleFileSelect = (id, node) => {
    // 신창영 : 파일을 새로 선택할 때마다 이전 실시간 전사 화면을 초기화
    if (activeFileId.value !== id) {
      transcriptions.value = []
    }
    originHandleFileSelect(id, node)
    if (isWorkspaceUuid(id) && node?.type === 'file') {
      loadSummariesForSession(id)
    } else {
      clearSummaryState()
    }
  }

  const handleOpenRecording = ({ sessionId = '', recordingId = '' } = {}) => {
    if (!isWorkspaceUuid(sessionId)) return
    loadSummariesForSession(sessionId, recordingId)
  }

  const cloneTranscriptions = () => {
    return JSON.parse(JSON.stringify(transcriptions.value || []))
  }

  const handleStartRecording = (mode = 'lecture') => {
    const recordingId = createLocalId('recording')
    if (isWorkspaceUuid(activeFileId.value)) {
      startLiveSummary(activeFileId.value, recordingId)
    } else {
      clearSummaryState()
    }
    return startRecording(mode, activeFileId.value, recordingId)
  }

  const showScheduleExtractionNotice = (sessionId, notifications = []) => {
    const items = notifications
      .filter((item) => item?.schedule_id && item?.title)
      .map((item) => ({
        id: item.schedule_id,
        title: item.title,
        dueDate: item.due_date || '',
        eventType: item.event_type || '',
        description: item.description || '',
        sourceText: item.source_text || ''
      }))

    if (items.length === 0) {
      scheduleExtractionNotice.value = null
      return
    }

    scheduleExtractionNotice.value = {
      id: `${sessionId}-${Date.now()}`,
      sessionId,
      count: items.length,
      items
    }
  }

  const dismissScheduleExtractionNotice = () => {
    scheduleExtractionNotice.value = null
  }

  const extractSchedulesForSession = async (sessionId, recordingId = '') => {
    if (!isWorkspaceUuid(sessionId)) return

    const response = await fetch('/schedule/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        recording_id: recordingId || null
      })
    })

    if (!response.ok) {
      throw new Error(`schedule extract failed: ${response.status}`)
    }

    const result = await response.json()
    await hydrateSchedules({ force: true })
    showScheduleExtractionNotice(sessionId, result?.notifications || [])
  }

  const handleStopRecording = async () => {
    const shouldSaveRecording = isRecording.value
    const recordingSnapshot = cloneTranscriptions()
    const durationText = recordingTimeText.value
    const mode = recordingMode.value
    const recordingId = activeRecordingId.value || createLocalId('recording')
    const startedAt = activeRecordingStartedAt.value || new Date().toISOString()
    const linkedMaterialId = currentPreviewMaterial.value?.id || null
    const linkedMaterialName = currentPreviewMaterial.value?.name || ''

    stopActiveRecording()

    if (!shouldSaveRecording || recordingSnapshot.length === 0) return

    const targetFileId = activeFileId.value
    if (!targetFileId) return

    const recording = {
      id: recordingId,
      recordingId,
      title: formatRecordingTitle(currentPreviewMaterial.value),
      startedAt,
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

      try {
        await extractSchedulesForSession(targetFileId, recordingId)
      } catch (error) {
        console.error('[schedule] extract after recording failed:', error)
      }

      try {
        await generateSummariesForSession(targetFileId, recordingSnapshot, mode, recordingId)
      } catch (error) {
        console.error('[summary] generate after recording failed:', error)
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
    recentFiles,
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
    scheduleExtractionNotice,
    summaryState,
    summaryNotes,
    aiInput,
    dismissScheduleExtractionNotice,
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
    handleOpenStoredMaterial,
    handleOpenRecording
  }
}
