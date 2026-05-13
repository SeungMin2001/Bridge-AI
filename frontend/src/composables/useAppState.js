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
import { useChat } from './useChat'
import { useScheduleState } from './useScheduleState'
import { useSummaryState } from './useSummaryState'

// 앱 전체에서 공유하는 상태 모듈들을 하나로 묶어 App.vue에 전달합니다.
export function useAppState() {
  const isRightSidebarVisible = ref(true)
  const scheduleExtractionNotice = ref(null)
  const { clearHistory, closeCitePopover } = useChat()
  // 녹음 중에는 현재 전사 스냅샷을 8초마다 요약 API로 넘깁니다.
  const LIVE_SUMMARY_REFRESH_MS = 8000
  let liveSummaryTimer = null
  let liveSummaryInFlight = false

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
    recordingAudioLevel,
    activeRecordingId,
    activeRecordingStartedAt,
    diarizationEnabled,
    diarizationStatus,
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
    generateSummariesForSession,
    generateMaterialSummaryForSource,
    deleteSummary
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
    const isDifferentFile = activeFileId.value !== id
    if (isDifferentFile) {
      transcriptions.value = []
      clearHistory()
      closeCitePopover()
      handleAiInputUpdate('')
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

  // 파일 전환, 녹음 종료, 컴포넌트 해제 시 실시간 요약 타이머를 정리합니다.
  const stopLiveSummaryRefresh = () => {
    if (liveSummaryTimer) {
      clearInterval(liveSummaryTimer)
      liveSummaryTimer = null
    }
  }

  // STT 화면에 쌓인 현재 전사 스냅샷을 8초마다 화자별 요약 API로 넘깁니다.
  const refreshLiveSummary = async (sessionId, recordingId, mode, options = {}) => {
    if (!isWorkspaceUuid(sessionId) || liveSummaryInFlight) return
    const snapshot = cloneTranscriptions()
    if (!snapshot.length) return

    liveSummaryInFlight = true
    try {
      await generateSummariesForSession(
        sessionId,
        snapshot,
        mode,
        recordingId,
        {
          live: options.final !== true,
          diarizationEnabled: options.diarizationEnabled === true
        },
      )
    } catch (error) {
      console.warn('[summary] live refresh failed:', error)
    } finally {
      liveSummaryInFlight = false
    }
  }

  const startLiveSummaryRefresh = (sessionId, recordingId, mode, options = {}) => {
    stopLiveSummaryRefresh()
    if (!isWorkspaceUuid(sessionId)) return
    const shouldDiarize = options.diarizationEnabled === true

    // 요약 요청이 아직 끝나지 않았으면 다음 주기를 건너뛰어 LLM 요청이 밀리지 않게 합니다.
    liveSummaryTimer = setInterval(() => {
      if (!isRecording.value) {
        stopLiveSummaryRefresh()
        return
      }
      refreshLiveSummary(sessionId, recordingId, mode, {
        final: false,
        diarizationEnabled: shouldDiarize
      })
    }, LIVE_SUMMARY_REFRESH_MS)
  }

  const handleStartRecording = (payload = 'lecture') => {
    const options = typeof payload === 'object' && payload !== null ? payload : { mode: payload }
    const mode = options.mode || 'lecture'
    const shouldDiarize = options.diarizationEnabled === true
    const recordingId = createLocalId('recording')
    stopLiveSummaryRefresh()
    if (isWorkspaceUuid(activeFileId.value)) {
      startLiveSummary(activeFileId.value, recordingId, { diarizationEnabled: shouldDiarize })
      startLiveSummaryRefresh(activeFileId.value, recordingId, mode, { diarizationEnabled: shouldDiarize })
    } else {
      clearSummaryState()
    }
    return startRecording(mode, activeFileId.value, recordingId, { diarizationEnabled: shouldDiarize })
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
    const initialRecordingSnapshot = cloneTranscriptions()
    const durationText = recordingTimeText.value
    const mode = recordingMode.value
    const shouldDiarize = diarizationEnabled.value === true
    const recordingId = activeRecordingId.value || createLocalId('recording')
    const startedAt = activeRecordingStartedAt.value || new Date().toISOString()
    const linkedMaterialId = currentPreviewMaterial.value?.id || null
    const linkedMaterialName = currentPreviewMaterial.value?.name || ''

    stopLiveSummaryRefresh()
    const stoppedRecording = await stopActiveRecording({ finalize: shouldDiarize })
    // 신창영: 수정 이유 - 녹음 종료 후 전체 오디오 화자분리로 보정된 전사 목록을 최종 저장/요약에 사용합니다.
    const recordingSnapshot = stoppedRecording?.transcriptions?.length
      ? stoppedRecording.transcriptions
      : initialRecordingSnapshot

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
      diarizationEnabled: shouldDiarize,
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
        await generateSummariesForSession(targetFileId, recordingSnapshot, mode, recordingId, {
          live: false,
          diarizationEnabled: shouldDiarize
        })
      } catch (error) {
        console.error('[summary] generate after recording failed:', error)
      }

    }
  }

  // 앱이 내려갈 때 마이크/WebSocket 등 녹음 리소스를 정리합니다.
  onUnmounted(() => {
    stopLiveSummaryRefresh()
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
    startRecording: handleStartRecording,
    pauseRecording,
    resumeRecording,
    stopRecording: handleStopRecording,
    generateMaterialSummaryForSource,
    deleteSummary,
    handleRightSidebarToggle,
    handleAddToNote,
    handleAskAi,
    handleUploadLectureMaterials,
    handleClosePreviewMaterial,
    handleOpenStoredMaterial,
    handleOpenRecording
  }
}
