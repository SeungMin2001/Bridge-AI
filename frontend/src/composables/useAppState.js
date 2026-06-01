import { onUnmounted, ref, watch } from 'vue'
import { isWorkspaceUuid, saveSessionResources, uploadWorkspaceRecording } from '../api/workspaceApi.js'
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
  const LIVE_SUMMARY_REFRESH_MS = Number(import.meta.env.VITE_LIVE_SUMMARY_REFRESH_MS || 20000)
  const LIVE_SCHEDULE_EXTRACT_DELAY_MS = Number(import.meta.env.VITE_LIVE_SCHEDULE_EXTRACT_DELAY_MS || 2500)
  const SCHEDULE_EVENT_PATTERN = /(중간고사|기말고사|쪽지시험|시험|고사|퀴즈|과제|제출|마감|보고서|레포트|리포트|발표|프로젝트|팀플|보강|휴강|실습|특강|수업|일정|기한|데드라인)/
  const SCHEDULE_DATE_PATTERN = /(\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}[./]\d{1,2}|오늘|내일|모레|다음\s*달\s*\d{1,2}\s*일|다다음\s*주|다음\s*주|이번\s*주|다다음주|다음주|이번주|월요일|화요일|수요일|목요일|금요일|토요일|일요일|오전\s*(\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시|오후\s*(\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시|\d{1,2}\s*시|\d{1,2}\s*분|까지|전까지|마감)/
  let liveSummaryTimer = null
  let liveSummaryInFlight = false
  let liveScheduleTimer = null
  let liveScheduleInFlight = false
  const liveScheduleRequestKeys = new Set()

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
    hydrateSchedules,
    confirmAndSyncToNotion,
    ignoreSchedule
  } = useScheduleState()

  const {
    summaryState,
    clearSummaryState,
    startLiveSummary,
    startFinalRecordingSummary,
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

  const getFileStem = (filename = '') => (
    String(filename || '')
      .replace(/\.[^.]+$/, '')
      .replace(/\s+/g, ' ')
      .trim()
  )

  const getAudioDuration = (file) => new Promise((resolve) => {
    if (!file || typeof document === 'undefined') {
      resolve(0)
      return
    }

    const audio = document.createElement('audio')
    const objectUrl = URL.createObjectURL(file)
    const cleanup = () => {
      URL.revokeObjectURL(objectUrl)
      audio.removeAttribute('src')
      audio.load()
    }

    const timer = window.setTimeout(() => {
      cleanup()
      resolve(0)
    }, 5000)

    audio.preload = 'metadata'
    audio.onloadedmetadata = () => {
      window.clearTimeout(timer)
      const duration = Number.isFinite(audio.duration) ? audio.duration : 0
      cleanup()
      resolve(duration)
    }
    audio.onerror = () => {
      window.clearTimeout(timer)
      cleanup()
      resolve(0)
    }
    audio.src = objectUrl
  })

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

  const generateRecordingSummaryForSource = async ({
    sessionId = activeFileId.value,
    recordingId = '',
    recording = null,
    recordings = [],
    summarySentences = 8
  } = {}) => {
    const targetSessionId = sessionId || activeFileId.value
    if (!isWorkspaceUuid(targetSessionId)) return

    const selectedRecordings = Array.isArray(recordings) && recordings.length
      ? recordings
      : (recording ? [recording] : [])
    const recordingSnapshot = selectedRecordings.flatMap((item) => (
      Array.isArray(item?.transcriptions) ? item.transcriptions : []
    ))
    if (!recordingSnapshot.length) return

    const targetRecordingId = recordingId
      || selectedRecordings.map((item) => item?.id || item?.recordingId).filter(Boolean).join('+')
      || `combined-recording-${Date.now()}`
    const mode = selectedRecordings[0]?.recordingMode || recordingMode.value || 'lecture'
    const shouldDiarize = selectedRecordings.some((item) => item?.diarizationEnabled === true)

    await generateSummariesForSession(targetSessionId, recordingSnapshot, mode, targetRecordingId, {
      live: false,
      diarizationEnabled: shouldDiarize,
      summarySentences
    })
  }

  const handleUploadRecordingFile = async (files = []) => {
    const file = Array.from(files || [])[0]
    const targetFileId = activeFileId.value
    if (!file || !targetFileId) return

    if (!file.type?.startsWith('audio/') && !/\.(aac|flac|m4a|mp3|ogg|opus|wav|webm)$/i.test(file.name)) {
      console.error('[workspace] recording upload failed: invalid audio file')
      return
    }

    const durationSeconds = await getAudioDuration(file)
    let recording = null
    let nextTree = normalizeFileTree(fileTree.value)

    if (isWorkspaceUuid(targetFileId)) {
      const result = await uploadWorkspaceRecording(targetFileId, file, {
        title: getFileStem(file.name),
        durationSeconds
      })
      recording = result.recording
      nextTree = result.node
        ? updateNodeById(nextTree, targetFileId, () => result.node)
        : updateNodeById(nextTree, targetFileId, (node) => addRecordingToCurrentWeek(node, recording))
    } else {
      const recordingId = createLocalId('recording')
      const uploadedAt = new Date().toISOString()
      recording = {
        id: recordingId,
        recordingId,
        title: getFileStem(file.name) || '업로드한 음성파일',
        startedAt: uploadedAt,
        endedAt: uploadedAt,
        durationText: '',
        durationSeconds,
        recordingMode: 'uploaded',
        diarizationEnabled: false,
        audioUrl: URL.createObjectURL(file),
        originalName: file.name,
        size: file.size,
        type: file.type || 'audio/*',
        uploadedAt,
        transcriptionStatus: 'not_started',
        transcriptions: []
      }
      nextTree = updateNodeById(nextTree, targetFileId, (node) => addRecordingToCurrentWeek(node, recording))
    }

    fileTree.value = nextTree
    const updatedNode = findNodeById(fileTree.value, targetFileId)
    if (isWorkspaceUuid(targetFileId) && Array.isArray(updatedNode?.weeks) && !recording?.storedName) {
      try {
        await saveSessionResources(targetFileId, updatedNode.weeks)
      } catch (error) {
        console.error('[workspace] session resources save failed:', error)
      }
    }

    if (recording) {
      handleOpenRecording({
        sessionId: targetFileId,
        recordingId: recording.id || recording.recordingId || ''
      })
    }
  }

  const cloneTranscriptions = () => {
    return JSON.parse(JSON.stringify(transcriptions.value || []))
  }

  const getScheduleCandidateSignature = (items = []) => {
    const text = items
      .flatMap((item) => {
        const segments = Array.isArray(item?.segments) ? item.segments : []
        return segments.length
          ? segments.map((segment) => segment?.text || '')
          : [item?.text || '']
      })
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim()

    if (!text) return ''
    const sentences = text
      .split(/[.!?。！？\n]+/)
      .map((sentence) => sentence.trim())
      .filter(Boolean)

    return sentences
      .filter((sentence) => SCHEDULE_EVENT_PATTERN.test(sentence) && SCHEDULE_DATE_PATTERN.test(sentence))
      .slice(-5)
      .join(' | ')
      .slice(-1200)
  }

  const hasSavedTranscriptSegment = (items = []) => (
    items.some((item) => (
      Array.isArray(item?.segments) &&
      item.segments.some((segment) => segment?.transcript_id || segment?.transcriptId)
    ))
  )

  // 파일 전환, 녹음 종료, 컴포넌트 해제 시 실시간 요약 타이머를 정리합니다.
  const stopLiveSummaryRefresh = () => {
    if (liveSummaryTimer) {
      clearInterval(liveSummaryTimer)
      liveSummaryTimer = null
    }
  }

  const stopLiveScheduleExtraction = () => {
    if (liveScheduleTimer) {
      clearTimeout(liveScheduleTimer)
      liveScheduleTimer = null
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

  const scheduleLiveScheduleExtraction = () => {
    if (!isRecording.value || !isWorkspaceUuid(activeFileId.value) || !activeRecordingId.value) return
    const snapshot = cloneTranscriptions()
    if (!hasSavedTranscriptSegment(snapshot)) return

    const signature = getScheduleCandidateSignature(snapshot)
    if (!signature) return

    const requestKey = `${activeFileId.value}:${activeRecordingId.value}:${signature}`
    if (liveScheduleRequestKeys.has(requestKey) || liveScheduleInFlight) return

    stopLiveScheduleExtraction()
    liveScheduleTimer = setTimeout(async () => {
      if (liveScheduleRequestKeys.has(requestKey) || liveScheduleInFlight) return
      liveScheduleInFlight = true
      try {
        await extractSchedulesForSession(activeFileId.value, activeRecordingId.value)
        liveScheduleRequestKeys.add(requestKey)
      } catch (error) {
        console.warn('[schedule] live extract failed:', error)
      } finally {
        liveScheduleInFlight = false
      }
    }, LIVE_SCHEDULE_EXTRACT_DELAY_MS)
  }

  watch(
    [
      isRecording,
      activeFileId,
      activeRecordingId,
      () => transcriptions.value.map((item) => (
        `${item?.text || ''} ${Array.isArray(item?.segments)
          ? item.segments.map((segment) => `${segment?.text || ''}:${segment?.transcript_id || segment?.transcriptId || ''}`).join(' ')
          : ''}`
      )).join('\n')
    ],
    scheduleLiveScheduleExtraction,
  )

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
    stopLiveScheduleExtraction()
    if (shouldSaveRecording && isWorkspaceUuid(activeFileId.value)) {
      startFinalRecordingSummary(activeFileId.value, recordingId, { diarizationEnabled: shouldDiarize })
    }
    const stoppedRecording = await stopActiveRecording({ finalize: true })
    const finalizeResult = stoppedRecording?.finalizeResult || {}
    // 신창영: 수정 이유 - 녹음 종료 후 전체 오디오 화자분리로 보정된 전사 목록을 최종 저장/요약에 사용합니다.
    const recordingSnapshot = stoppedRecording?.transcriptions?.length
      ? stoppedRecording.transcriptions
      : initialRecordingSnapshot

    if (!shouldSaveRecording || (recordingSnapshot.length === 0 && !finalizeResult.audioUrl)) {
      if (isWorkspaceUuid(activeFileId.value)) {
        await loadSummariesForSession(activeFileId.value, recordingId, {
          silent: true,
          diarizationEnabled: shouldDiarize
        })
      }
      return
    }

    const targetFileId = activeFileId.value
    if (!targetFileId) {
      clearSummaryState()
      return
    }

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
      audioUrl: finalizeResult.audioUrl || null,
      storedName: finalizeResult.storedName || '',
      size: finalizeResult.audioSize || 0,
      type: finalizeResult.audioType || '',
      durationSeconds: finalizeResult.durationSeconds ?? null,
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

      if (recordingSnapshot.length > 0) {
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
  }

  const updateScheduleNotionId = async (scheduleId, notionPageId) => {
    try {
      const response = await fetch(`/schedule/${scheduleId}/notion`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notion_page_id: notionPageId })
      })
      if (!response.ok) {
        throw new Error(`notion id update failed: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('[schedule] update notion id failed:', error)
      throw error
    }
  }

  // 앱이 내려갈 때 마이크/WebSocket 등 녹음 리소스를 정리합니다.
  onUnmounted(() => {
    stopLiveSummaryRefresh()
    stopLiveScheduleExtraction()
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
    extractSchedulesForSession,
    updateScheduleNotionId,
    confirmAndSyncToNotion,
    ignoreSchedule,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleAiInputUpdate,
    handleFileSelect,
    startRecording: handleStartRecording,
    pauseRecording,
    resumeRecording,
    stopRecording: handleStopRecording,
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
  }
}
