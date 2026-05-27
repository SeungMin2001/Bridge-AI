import { ref } from 'vue'
import { isWorkspaceUuid } from '../api/workspaceApi.js'

const SUMMARY_API_BASE = '/summary'
const MATERIAL_SUMMARY_ENDPOINT = '/material/generate'

// 개발 중 브라우저 콘솔에서 STT -> speaker_id -> 요약 요청 흐름을 추적하기 위한 로그입니다.
const logSpeakerFlow = (step, payload = {}) => {
  if (!import.meta.env.DEV) return
  console.debug(`[speaker-flow] ${step}`, payload)
}

// pyannote가 내려주는 SPEAKER_00 형태의 내부 ID를 화면용 라벨로 바꿉니다.
const formatSpeakerLabel = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  const match = normalized.match(/^SPEAKER_(\d+)$/i)
  if (match) return `화자 ${Number.parseInt(match[1], 10) + 1}`
  if (!normalized || normalized === 'UNKNOWN') return '화자 미상'
  return normalized
}

// 화자분리가 실패한 짧은 발화는 UNKNOWN으로 들어올 수 있어 요약 대상에서 제외합니다.
const isUnknownSpeaker = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  return !normalized || normalized === 'UNKNOWN' || normalized === '화자 미상'
}

// DB 조회 순서와 상관없이 화자 1, 화자 2 순서로 카드가 보이도록 정렬 기준을 만듭니다.
const getSpeakerSortOrder = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  const speakerMatch = normalized.match(/^SPEAKER_(\d+)$/i)
  if (speakerMatch) return Number.parseInt(speakerMatch[1], 10)

  const labelMatch = normalized.match(/^화자\s*(\d+)$/)
  if (labelMatch) return Number.parseInt(labelMatch[1], 10) - 1

  return Number.MAX_SAFE_INTEGER
}

const withQuery = (endpoint, params = {}) => {
  const searchParams = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      searchParams.set(key, value)
    }
  })
  const query = searchParams.toString()
  return query ? `${endpoint}?${query}` : endpoint
}

const createEmptySummaryState = (sessionId = '', recordingId = '', status = 'idle', options = {}) => ({
  sessionId,
  recordingId,
  status,
  diarizationEnabled: options.diarizationEnabled !== false,
  speakerSummaries: [],
  sessionSummary: null,
  recordingSummaries: [],
  materialSummaries: [],
  materialStatus: 'idle',
  materialError: '',
  error: ''
})

const requestSummaryJson = async (endpoint, options = {}) => {
  const response = await fetch(`${SUMMARY_API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  })

  if (!response.ok) {
    const raw = await response.text()
    let detail = raw
    try {
      const parsed = raw ? JSON.parse(raw) : {}
      detail = parsed.detail || parsed.error || raw
    } catch {
      detail = raw
    }
    throw new Error(detail || `Summary API request failed: ${response.status}`)
  }

  return response.json()
}

const postSummaryJson = (endpoint, payload) => requestSummaryJson(endpoint, {
  method: 'POST',
  body: JSON.stringify(payload)
})

const getMaterialSummaryTopK = (summaryLevel = 'standard', summarySentences = 8) => {
  const sentences = Math.max(1, Number(summarySentences) || 8)
  const level = String(summaryLevel || 'standard').toLowerCase()
  if (level === 'detailed' || level === 'page') {
    return Math.min(14, Math.max(12, sentences * 2))
  }
  return Math.min(12, Math.max(10, sentences * 2))
}

const getTranscriptText = (transcription = {}) => {
  if (Array.isArray(transcription.segments) && transcription.segments.length) {
    return transcription.segments
      .map((segment) => segment?.text || '')
      .filter(Boolean)
      .join(' ')
      .trim()
  }
  return String(transcription.text || '').trim()
}

const buildSessionText = (recordingSnapshot = []) => (
  recordingSnapshot
    .map((transcription) => getTranscriptText(transcription))
    .filter(Boolean)
    .join('\n')
    .trim()
)

const getTranscriptionTimeBounds = (transcription = {}) => {
  const starts = []
  const ends = []

  if (Number.isFinite(Number(transcription.start))) starts.push(Number(transcription.start))
  if (Number.isFinite(Number(transcription.startTime))) starts.push(Number(transcription.startTime))
  if (Number.isFinite(Number(transcription.end))) ends.push(Number(transcription.end))
  if (Number.isFinite(Number(transcription.endTime))) ends.push(Number(transcription.endTime))

  if (Array.isArray(transcription.segments)) {
    transcription.segments.forEach((segment) => {
      if (Number.isFinite(Number(segment?.start))) starts.push(Number(segment.start))
      if (Number.isFinite(Number(segment?.start_time))) starts.push(Number(segment.start_time))
      if (Number.isFinite(Number(segment?.end))) ends.push(Number(segment.end))
      if (Number.isFinite(Number(segment?.end_time))) ends.push(Number(segment.end_time))
    })
  }

  return {
    start: starts.length ? Math.min(...starts) : null,
    end: ends.length ? Math.max(...ends) : null
  }
}

// 현재까지 쌓인 전사문을 speaker_id별로 묶어 화자별 요약 API payload로 변환합니다.
const buildSpeakerPayloads = (sessionId, recordingSnapshot = [], recordingMode = 'lecture', recordingId = '') => {
  const speakerMap = new Map()

  recordingSnapshot.forEach((transcription) => {
    const segmentItems = Array.isArray(transcription.segments) && transcription.segments.length
      ? transcription.segments
      : [transcription]

    segmentItems.forEach((segment) => {
      const text = String(segment?.text || getTranscriptText(transcription) || '').trim()
      if (!text) return

      // 신창영: 수정 이유 - 녹음 종료 후 전체 화자분리 보정은 segment 단위로 들어오므로, 최종 요약도 segment의 speakerId를 우선 사용합니다.
      const speakerId = segment?.speakerId
        || segment?.speaker
        || transcription.speakerId
        || transcription.speaker
        || ''
      if (isUnknownSpeaker(speakerId)) return

      const speakerLabel = segment?.speaker || transcription.speaker || speakerId
      const start = segment?.start != null && Number.isFinite(Number(segment.start))
        ? Number(segment.start)
        : getTranscriptionTimeBounds(transcription).start
      const end = segment?.end != null && Number.isFinite(Number(segment.end))
        ? Number(segment.end)
        : getTranscriptionTimeBounds(transcription).end

      if (!speakerMap.has(speakerId)) {
        speakerMap.set(speakerId, {
          session_id: sessionId,
          recording_id: recordingId || transcription.recordingId || '',
          speaker_id: speakerLabel,
          speaker_texts: [],
          source_start_time: null,
          source_end_time: null
        })
      }

      const item = speakerMap.get(speakerId)
      item.speaker_texts.push(text)
      if (start !== null) {
        item.source_start_time = item.source_start_time === null
          ? start
          : Math.min(item.source_start_time, start)
      }
      if (end !== null) {
        item.source_end_time = item.source_end_time === null
          ? end
          : Math.max(item.source_end_time, end)
      }
    })
  })

  return Array.from(speakerMap.values())
    .map((item) => ({
      session_id: item.session_id,
      recording_id: item.recording_id || null,
      speaker_id: item.speaker_id,
      speaker_text: item.speaker_texts.join('\n'),
      summary_sentences: 3,
      source_start_time: item.source_start_time,
      source_end_time: item.source_end_time
    }))
    .filter((item) => item.speaker_text.trim().length >= 10)
}

const isMaterialSummaryRow = (item = {}) => (
  item?.speaker_id === 'MATERIAL' || String(item?.recording_id || '').startsWith('material:')
)

const parseMaterialSource = (sourceText = '') => {
  if (!sourceText) return {}
  try {
    return JSON.parse(sourceText)
  } catch {
    return {}
  }
}

const normalizeMaterialSummary = (item = {}) => {
  const source = parseMaterialSource(item.source_text || '')
  const sourceMaterials = Array.isArray(source.materials) ? source.materials : []
  const firstMaterial = sourceMaterials[0] || {}

  return {
    id: item.summary_id,
    key: item.summary_id || item.recording_id || firstMaterial.storedName || firstMaterial.name,
    summary: item.session_summary || item.material_summary || '',
    createdAt: item.created_at || '',
    sourceText: item.source_text || '',
    sourceMaterials,
    summaryLevel: source.summaryLevel || item.summary_level || 'standard',
    title: sourceMaterials.length > 1
      ? `${firstMaterial.name || 'PDF 강의자료'} 외 ${sourceMaterials.length - 1}개`
      : (firstMaterial.name || 'PDF 강의자료')
  }
}

const normalizeSummaryRow = (item = {}) => ({
  id: item.summary_id,
  key: item.summary_id || `${item.speaker_id || 'speaker'}-${item.recording_id || 'session'}`,
  label: formatSpeakerLabel(item.speaker_id),
  speakerId: item.speaker_id || '',
  summary: item.speaker_summary || '',
  latestText: item.source_text || '',
  createdAt: item.created_at || ''
})

const normalizeSessionSummary = (item = {}) => ({
  id: item.summary_id,
  key: item.summary_id || item.recording_id || 'session-summary',
  recordingId: item.recording_id || '',
  summary: item.session_summary || '',
  createdAt: item.created_at || '',
  sourceText: item.source_text || ''
})

// DB에 계속 저장되는 실시간 요약 중 화면에는 화자별 최신 1개만 노출합니다.
const normalizeSummaries = (summaries = [], recordingId = '') => {
  const scopedSummaries = recordingId
    ? summaries.filter((item) => String(item?.recording_id || '') === String(recordingId))
    : summaries

  const speakerGroups = new Map()
  scopedSummaries
    .filter((item) => (
      item?.speaker_summary &&
      !isMaterialSummaryRow(item) &&
      !isUnknownSpeaker(item.speaker_id)
    ))
    .forEach((item) => {
      const key = item.speaker_id || 'UNKNOWN'
      if (!speakerGroups.has(key)) {
        speakerGroups.set(key, [])
      }
      speakerGroups.get(key).push(item)
    })

  const speakerSummaries = Array.from(speakerGroups.entries())
    .sort(([leftSpeakerId], [rightSpeakerId]) => (
      getSpeakerSortOrder(leftSpeakerId) - getSpeakerSortOrder(rightSpeakerId)
    ))
    .map(([speakerId, items]) => {
      const [latest] = items.sort((left, right) => (
        new Date(right.created_at || 0).getTime() - new Date(left.created_at || 0).getTime()
      ))
      return {
        ...normalizeSummaryRow(latest),
        key: speakerId || latest?.summary_id || 'UNKNOWN'
      }
    })

  const sessionSummary = scopedSummaries.find((item) => item?.session_summary && !isMaterialSummaryRow(item)) || null
  const recordingSummaryGroups = new Map()
  summaries
    .filter((item) => item?.session_summary && !isMaterialSummaryRow(item))
    .forEach((item) => {
      const key = item.recording_id || 'session'
      if (!recordingSummaryGroups.has(key)) {
        recordingSummaryGroups.set(key, [])
      }
      recordingSummaryGroups.get(key).push(item)
    })

  const recordingSummaries = Array.from(recordingSummaryGroups.values())
    .map((items) => {
      const [latest] = items.sort((left, right) => (
        new Date(right.created_at || 0).getTime() - new Date(left.created_at || 0).getTime()
      ))
      return normalizeSessionSummary(latest)
    })

  const materialSummaries = summaries
    .filter((item) => item?.session_summary && isMaterialSummaryRow(item))
    .map(normalizeMaterialSummary)

  return {
    speakerSummaries,
    sessionSummary: sessionSummary ? normalizeSessionSummary(sessionSummary) : null,
    recordingSummaries,
    materialSummaries
  }
}

const normalizeMaterialForRequest = (material = {}) => ({
  id: material.id || '',
  name: material.name || material.title || 'PDF 강의자료',
  storedName: material.storedName || '',
  url: material.url || '',
  type: material.type || material.fileType || material.mimeType || ''
})

export function useSummaryState() {
  const summaryState = ref(createEmptySummaryState())

  const clearSummaryState = () => {
    summaryState.value = createEmptySummaryState()
  }

  const startLiveSummary = (sessionId = '', recordingId = '', options = {}) => {
    summaryState.value = createEmptySummaryState(sessionId, recordingId, 'live', {
      diarizationEnabled: options.diarizationEnabled !== false
    })
    return summaryState.value
  }

  const setSummaryState = (nextState) => {
    summaryState.value = {
      ...summaryState.value,
      ...nextState
    }
  }

  const startFinalRecordingSummary = (sessionId = '', recordingId = '', options = {}) => {
    if (!isWorkspaceUuid(sessionId)) return summaryState.value
    setSummaryState({
      sessionId,
      recordingId,
      diarizationEnabled: options.diarizationEnabled !== false,
      status: 'generating',
      error: ''
    })
    return summaryState.value
  }

  const loadSummariesForSession = async (sessionId, recordingId = '', options = {}) => {
    if (!isWorkspaceUuid(sessionId)) {
      summaryState.value = createEmptySummaryState()
      return summaryState.value
    }

    const diarizationEnabled = options.diarizationEnabled ?? summaryState.value.diarizationEnabled ?? true

    if (!options.silent) {
      setSummaryState({ sessionId, recordingId, diarizationEnabled, status: 'loading', error: '' })
    } else {
      setSummaryState({ sessionId, recordingId, diarizationEnabled, error: '' })
    }

    try {
      const summaryResult = await requestSummaryJson(withQuery(`/session/${sessionId}`, { recording_id: recordingId }))
      const normalized = normalizeSummaries(summaryResult.summaries || [], recordingId)
      summaryState.value = {
        ...createEmptySummaryState(sessionId, recordingId, 'done', { diarizationEnabled }),
        status: 'done',
        speakerSummaries: normalized.speakerSummaries,
        sessionSummary: normalized.sessionSummary,
        recordingSummaries: normalized.recordingSummaries,
        materialSummaries: normalized.materialSummaries
      }
      return summaryState.value
    } catch (error) {
      console.warn('[summary] load failed:', error)
      summaryState.value = {
        ...createEmptySummaryState(sessionId, recordingId, 'error', { diarizationEnabled }),
        status: 'error',
        error: error?.message || '요약을 불러오지 못했습니다.'
      }
      return summaryState.value
    }
  }

  const generateSummariesForSession = async (
    sessionId,
    recordingSnapshot = [],
    recordingMode = 'lecture',
    recordingId = '',
    options = {},
  ) => {
    if (!isWorkspaceUuid(sessionId)) return

    const isLiveUpdate = options.live === true
    const shouldDiarize = options.diarizationEnabled !== false
    const sessionText = buildSessionText(recordingSnapshot)

    if (!shouldDiarize) {
      if (sessionText.length < 10) {
        if (!isLiveUpdate) {
          summaryState.value = {
            ...createEmptySummaryState(sessionId, recordingId, 'error', { diarizationEnabled: false }),
            error: '요약할 전사문이 부족합니다.'
          }
        }
        return
      }

      setSummaryState({
        sessionId,
        recordingId,
        diarizationEnabled: false,
        status: isLiveUpdate ? 'updating' : 'generating',
        error: ''
      })

      try {
        await postSummaryJson('/session/text/generate', {
          session_id: sessionId,
          recording_id: recordingId || null,
          session_text: sessionText,
          summary_sentences: 3
        })
      } catch (error) {
        console.warn('[summary] session text generation failed:', error)
        if (!isLiveUpdate) {
          setSummaryState({
            sessionId,
            recordingId,
            diarizationEnabled: false,
            status: 'error',
            error: error?.message || '전체 요약 생성에 실패했습니다.'
          })
        }
        return
      }

      await loadSummariesForSession(sessionId, recordingId, {
        silent: isLiveUpdate,
        diarizationEnabled: false
      })
      return
    }

    setSummaryState({
      sessionId,
      recordingId,
      diarizationEnabled: true,
      status: isLiveUpdate ? 'updating' : 'generating',
      error: ''
    })

    if (!isLiveUpdate && sessionText.length >= 10) {
      try {
        // 신창영: 수정 이유 - 화자분리 녹음도 종료 후에는 전체 녹음 요약과 화자별 요약을 둘 다 생성합니다.
        await postSummaryJson('/session/text/generate', {
          session_id: sessionId,
          recording_id: recordingId || null,
          session_text: sessionText,
          summary_sentences: 3
        })
      } catch (error) {
        console.warn('[summary] diarized session summary generation failed:', error)
      }
    }

    const speakerPayloads = buildSpeakerPayloads(sessionId, recordingSnapshot, recordingMode, recordingId)
    logSpeakerFlow('frontend -> backend speaker summary payloads', {
      sessionId,
      recordingId,
      speakers: speakerPayloads.map((payload) => ({
        speakerId: payload.speaker_id,
        textLength: payload.speaker_text.length
      }))
    })
    if (!speakerPayloads.length) {
      if (!isLiveUpdate && sessionText.length >= 10) {
        await loadSummariesForSession(sessionId, recordingId, {
          silent: false,
          diarizationEnabled: true
        })
        return
      }
      if (!isLiveUpdate) {
        summaryState.value = {
          ...createEmptySummaryState(sessionId, recordingId),
          diarizationEnabled: true,
          status: 'error',
          error: '요약할 전사문이 부족합니다.'
        }
      }
      return
    }

    for (const payload of speakerPayloads) {
      try {
        logSpeakerFlow('frontend -> backend /summary/speaker/generate', {
          speakerId: payload.speaker_id,
          recordingId: payload.recording_id,
          textLength: payload.speaker_text.length
        })
        await postSummaryJson('/speaker/generate', payload)
      } catch (error) {
        console.warn('[summary] speaker generation failed:', error)
      }
    }

    await loadSummariesForSession(sessionId, recordingId, {
      silent: isLiveUpdate,
      diarizationEnabled: true
    })
  }

  const generateMaterialSummaryForSource = async ({
    sessionId,
    materials = [],
    summarySentences = 8,
    summaryLevel = 'standard'
  } = {}) => {
    if (!isWorkspaceUuid(sessionId)) return

    const normalizedMaterials = materials
      .map(normalizeMaterialForRequest)
      .filter((material) => material.id || material.storedName || material.name || material.url)

    if (!normalizedMaterials.length) {
      setSummaryState({
        materialStatus: 'error',
        materialError: '요약할 PDF 강의자료를 선택하세요.'
      })
      return
    }

    setSummaryState({
      sessionId,
      materialStatus: 'generating',
      materialError: ''
    })

    try {
      const result = await postSummaryJson(MATERIAL_SUMMARY_ENDPOINT, {
        session_id: sessionId,
        material_ids: normalizedMaterials.map((material) => material.id).filter(Boolean),
        stored_names: normalizedMaterials.map((material) => material.storedName).filter(Boolean),
        summary_sentences: summarySentences,
        summary_level: summaryLevel,
        top_k: getMaterialSummaryTopK(summaryLevel, summarySentences)
      })

      const sourceMaterials = Array.isArray(result.source_materials)
        ? result.source_materials
        : normalizedMaterials
      const firstMaterial = sourceMaterials[0] || {}
      const materialSummary = {
        id: result.summary_id,
        key: result.summary_id || result.recording_id || firstMaterial.storedName || firstMaterial.name,
        title: sourceMaterials.length > 1
          ? `${firstMaterial.name || 'PDF 강의자료'} 외 ${sourceMaterials.length - 1}개`
          : (firstMaterial.name || 'PDF 강의자료'),
        summary: result.material_summary || result.session_summary || '',
        createdAt: new Date().toISOString(),
        sourceMaterials,
        summaryLevel: result.summary_level || summaryLevel,
        sourceText: ''
      }

      setSummaryState({
        materialStatus: 'done',
        materialError: '',
        materialSummaries: [
          materialSummary,
          ...(summaryState.value.materialSummaries || []).filter((item) => item.id !== materialSummary.id)
        ]
      })
    } catch (error) {
      console.warn('[summary] material generation failed:', error)
      setSummaryState({
        materialStatus: 'error',
        materialError: error?.message || '파일 요약 생성에 실패했습니다.'
      })
    }
  }

  const deleteSummary = async (summaryId) => {
    const targetId = String(summaryId || '').trim()
    if (!targetId) return

    try {
      await requestSummaryJson(`/${encodeURIComponent(targetId)}`, {
        method: 'DELETE'
      })

      const currentState = summaryState.value
      setSummaryState({
        error: '',
        materialError: '',
        sessionSummary: currentState.sessionSummary?.id === targetId
          ? null
          : currentState.sessionSummary,
        recordingSummaries: (currentState.recordingSummaries || []).filter((item) => item.id !== targetId),
        speakerSummaries: (currentState.speakerSummaries || [])
          .filter((item) => item.id !== targetId),
        materialSummaries: (currentState.materialSummaries || []).filter((item) => item.id !== targetId)
      })
    } catch (error) {
      console.warn('[summary] delete failed:', error)
      setSummaryState({
        error: error?.message || '요약 삭제에 실패했습니다.',
        materialError: error?.message || '요약 삭제에 실패했습니다.'
      })
    }
  }

  return {
    summaryState,
    clearSummaryState,
    startLiveSummary,
    startFinalRecordingSummary,
    loadSummariesForSession,
    generateSummariesForSession,
    generateMaterialSummaryForSource,
    deleteSummary
  }
}
