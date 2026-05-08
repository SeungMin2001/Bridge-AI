import { ref } from 'vue'
import { isWorkspaceUuid } from '../api/workspaceApi.js'

const SUMMARY_API_BASE = '/summary'
const MATERIAL_SUMMARY_ENDPOINT = '/test/material/textrank'

const createEmptySummaryState = (sessionId = '', recordingId = '', status = 'idle') => ({
  sessionId,
  recordingId,
  status,
  speakerSummaries: [],
  sessionSummary: null,
  materialSummaries: [],
  materialStatus: 'idle',
  materialError: '',
  keywords: [],
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

const buildSpeakerPayloads = (sessionId, recordingSnapshot = [], recordingMode = 'lecture', recordingId = '') => {
  const speakerMap = new Map()

  recordingSnapshot.forEach((transcription, index) => {
    const text = getTranscriptText(transcription)
    if (!text) return

    const speakerId = transcription.speakerId
      || transcription.speaker
      || (recordingMode === 'meeting' ? `unknown-speaker-${index + 1}` : '나')
    const speakerLabel = transcription.speaker || speakerId

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

const normalizeSummaries = (summaries = [], recordingId = '') => {
  const scopedSummaries = recordingId
    ? summaries.filter((item) => String(item?.recording_id || '') === String(recordingId))
    : summaries

  const speakerSummaries = scopedSummaries
    .filter((item) => item?.speaker_summary && !isMaterialSummaryRow(item))
    .map((item) => ({
      id: item.summary_id,
      key: item.summary_id || `${item.speaker_id || 'speaker'}-${item.recording_id || 'session'}`,
      label: item.speaker_id || '화자',
      summary: item.speaker_summary,
      latestText: item.source_text || '',
      createdAt: item.created_at || ''
    }))

  const sessionSummary = scopedSummaries.find((item) => item?.session_summary && !isMaterialSummaryRow(item)) || null
  const materialSummaries = summaries
    .filter((item) => item?.session_summary && isMaterialSummaryRow(item))
    .map(normalizeMaterialSummary)

  return {
    speakerSummaries,
    sessionSummary: sessionSummary
      ? {
          id: sessionSummary.summary_id,
          recordingId: sessionSummary.recording_id || '',
          summary: sessionSummary.session_summary,
          createdAt: sessionSummary.created_at || '',
          sourceText: sessionSummary.source_text || ''
        }
      : null,
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

  const startLiveSummary = (sessionId = '', recordingId = '') => {
    summaryState.value = createEmptySummaryState(sessionId, recordingId, 'live')
    return summaryState.value
  }

  const setSummaryState = (nextState) => {
    summaryState.value = {
      ...summaryState.value,
      ...nextState
    }
  }

  const loadSummariesForSession = async (sessionId, recordingId = '') => {
    if (!isWorkspaceUuid(sessionId)) {
      summaryState.value = createEmptySummaryState()
      return summaryState.value
    }

    setSummaryState({ sessionId, recordingId, status: 'loading', error: '' })

    try {
      const [summaryResult, keywordResult] = await Promise.all([
        requestSummaryJson(`/session/${sessionId}`),
        requestSummaryJson(`/keywords/session/${sessionId}`).catch(() => ({ keywords: [] }))
      ])
      const normalized = normalizeSummaries(summaryResult.summaries || [], recordingId)
      summaryState.value = {
        ...createEmptySummaryState(sessionId, recordingId),
        status: 'done',
        speakerSummaries: normalized.speakerSummaries,
        sessionSummary: normalized.sessionSummary,
        materialSummaries: normalized.materialSummaries,
        keywords: keywordResult.keywords || []
      }
      return summaryState.value
    } catch (error) {
      console.warn('[summary] load failed:', error)
      summaryState.value = {
        ...createEmptySummaryState(sessionId, recordingId),
        status: 'error',
        error: error?.message || '요약을 불러오지 못했습니다.'
      }
      return summaryState.value
    }
  }

  const generateSummariesForSession = async (sessionId, recordingSnapshot = [], recordingMode = 'lecture', recordingId = '') => {
    if (!isWorkspaceUuid(sessionId)) return

    const speakerPayloads = buildSpeakerPayloads(sessionId, recordingSnapshot, recordingMode, recordingId)
    if (!speakerPayloads.length) {
      summaryState.value = {
        ...createEmptySummaryState(sessionId, recordingId),
        status: 'error',
        error: '요약할 전사문이 부족합니다.'
      }
      return
    }

    setSummaryState({ sessionId, recordingId, status: 'generating', error: '' })

    let generatedKeywords = []
    try {
      const keywordResult = await postSummaryJson('/keywords/generate', {
        session_id: sessionId,
        recording_id: recordingId || null,
        top_k: 12,
        window_size: 4
      })
      generatedKeywords = keywordResult.keywords || []
    } catch (error) {
      console.warn('[summary] keyword generation failed:', error)
    }

    const speakerResults = []
    for (const payload of speakerPayloads) {
      try {
        speakerResults.push(await postSummaryJson('/speaker/generate', payload))
      } catch (error) {
        console.warn('[summary] speaker generation failed:', error)
      }
    }

    if (generatedKeywords.length && speakerResults.length) {
      try {
        await postSummaryJson('/session/generate', {
          session_id: sessionId,
          recording_id: recordingId || null,
          summary_sentences: 3
        })
      } catch (error) {
        console.warn('[summary] session generation failed:', error)
      }
    }

    await loadSummariesForSession(sessionId, recordingId)
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
        top_k: summarySentences
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
        speakerSummaries: (currentState.speakerSummaries || []).filter((item) => item.id !== targetId),
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
    loadSummariesForSession,
    generateSummariesForSession,
    generateMaterialSummaryForSource,
    deleteSummary
  }
}
