<script setup>
import { computed, ref, watch } from 'vue'
import { marked } from 'marked'
import LoadingHourglass from '../../ui/LoadingHourglass.vue'

marked.setOptions({ breaks: true, gfm: true })

const props = defineProps({
  tabAnim: { type: String, default: 'tab-slide-right' },
  activeFileId: { type: String, default: '' },
  currentAttachments: { type: Array, default: () => [] },
  currentRecordings: { type: Array, default: () => [] },
  folderFiles: { type: Array, default: () => [] },
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingTimeText: { type: String, default: '00:00:00' },
  recordingMode: { type: String, default: 'lecture' },
  diarizationEnabled: { type: Boolean, default: true },
  transcriptions: { type: Array, default: () => [] },
  summaryState: { type: Object, default: () => ({}) },
  summarySource: { type: Object, default: null }
})

const emit = defineEmits([
  'deleteSummary',
  'generateMaterialSummary',
  'generateRecordingSummary',
  'askAi',
  'addToNote'
])

const summaryModes = [
  { key: 'basic', label: '요약', icon: 'summarize' },
  { key: 'live', label: '실시간 요약', icon: 'graphic_eq' }
]

const summaryMode = ref('basic')
const pendingMaterialSummaryKey = ref('')
const pendingRecordingSummaryKey = ref('')
const collapsedBasicSummaryKeys = ref(new Set())
const collapsedLiveFlowKeys = ref(new Set())
const isSummarySourcePickerOpen = ref(false)
const summarySourcePickerAnchor = ref({ left: 0, top: 0 })
const hasPickedSummarySources = ref(false)
const localSummarySourceIds = ref([])

const setSummaryMode = (mode) => {
  summaryMode.value = mode
}

const renderMarkdown = (text = '') => marked.parse(String(text || '').trim())

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

const formatSummaryTime = (value = '') => {
  if (!value) return ''
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return ''
  return parsed.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
}

const formatRecordingSeconds = (value = 0) => {
  const seconds = Math.max(0, Math.floor(Number(value) || 0))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainSeconds = String(seconds % 60).padStart(2, '0')
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${remainSeconds}`
  return `${String(minutes).padStart(2, '0')}:${remainSeconds}`
}

const formatRecordingOffset = (value = '') => {
  if (Number.isFinite(Number(value)) && value !== '') return formatRecordingSeconds(Number(value))

  const raw = String(value || '').trim()
  if (!raw) return ''

  const parts = raw.split(':').map((part) => Number.parseInt(part, 10))
  if (parts.some((part) => Number.isNaN(part))) return raw
  if (parts.length === 3) return formatRecordingSeconds((parts[0] * 3600) + (parts[1] * 60) + parts[2])
  if (parts.length === 2) return formatRecordingSeconds((parts[0] * 60) + parts[1])
  return raw
}

const sourceTokens = (source = {}) => (
  [
    source.id,
    source.materialId,
    source.recordingId,
    source.storedName,
    source.name,
    source.title,
    source.originalName,
    source.url
  ]
    .map((value) => String(value || '').trim())
    .filter(Boolean)
)

const getSourceId = (source = {}, fallback = '') => sourceTokens(source)[0] || String(fallback)
const getSourceTitle = (source = {}, fallback = '파일') => (
  String(source.title || source.name || source.originalName || source.storedName || fallback).trim()
)

const isPdfMaterial = (material = {}) => {
  const type = String(material.type || material.fileType || material.mimeType || '').toLowerCase()
  const name = String(material.name || material.title || material.storedName || '').toLowerCase()
  return type.includes('pdf') || name.endsWith('.pdf')
}

const getMaterialSetKey = (materials = []) => (
  materials
    .map((material, index) => sourceTokens(material)[0] || getSourceTitle(material, index))
    .filter(Boolean)
    .sort()
    .join('|')
)

const getRecordingSetKey = (recordings = []) => (
  recordings
    .map((recording, index) => getSourceId(recording, index))
    .filter(Boolean)
    .sort()
    .join('+')
)

const getLatestByDate = (items = []) => (
  [...items].sort((left, right) => (
    new Date(right.createdAt || 0).getTime() - new Date(left.createdAt || 0).getTime()
  ))[0] || null
)

const materialSummaries = computed(() => (
  Array.isArray(props.summaryState?.materialSummaries)
    ? props.summaryState.materialSummaries
    : []
))

const recordingSummaries = computed(() => (
  Array.isArray(props.summaryState?.recordingSummaries)
    ? props.summaryState.recordingSummaries
    : []
))

const materialMatches = (material = {}, sourceMaterial = {}) => {
  const tokens = new Set(sourceTokens(material))
  return sourceTokens(sourceMaterial).some((token) => tokens.has(token))
}

const getMaterialSummaryForSource = (material = {}) => (
  getLatestByDate(materialSummaries.value.filter((summary) => (
    Array.isArray(summary.sourceMaterials) &&
    summary.sourceMaterials.length === 1 &&
    summary.sourceMaterials.some((sourceMaterial) => materialMatches(material, sourceMaterial))
  )))
)

const getRecordingSummaryForSource = (recording = {}) => {
  const ids = new Set(sourceTokens(recording))
  return getLatestByDate(recordingSummaries.value.filter((summary) => (
    summary.recordingId && ids.has(String(summary.recordingId))
  )))
}

const getRecordingTranscriptions = (recording = {}) => (
  Array.isArray(recording.transcriptions) ? recording.transcriptions : []
)

const hasRecordingTranscript = (recording = {}) => (
  getRecordingTranscriptions(recording).some((transcription) => getTranscriptText(transcription))
)

const materialSourceItems = computed(() => (
  Array.isArray(props.currentAttachments)
    ? props.currentAttachments.map((material, index) => {
        const supported = isPdfMaterial(material)
        return {
          key: `material-${getSourceId(material, index)}`,
          type: 'material',
          icon: supported ? 'picture_as_pdf' : 'description',
          title: getSourceTitle(material, `강의자료 ${index + 1}`),
          source: material,
          supported,
          summary: getMaterialSummaryForSource(material)
        }
      })
    : []
))

const recordingSourceItems = computed(() => (
  Array.isArray(props.currentRecordings)
    ? props.currentRecordings.map((recording, index) => {
        const transcriptions = getRecordingTranscriptions(recording)
        return {
          key: `recording-${getSourceId(recording, index)}`,
          type: 'recording',
          icon: 'graphic_eq',
          title: getSourceTitle(recording, `녹음본 ${index + 1}`),
          source: recording,
          recordingId: recording.id || recording.recordingId || '',
          supported: hasRecordingTranscript(recording),
          summary: getRecordingSummaryForSource(recording)
        }
      })
    : []
))

const basicFileItems = computed(() => [
  ...materialSourceItems.value,
  ...recordingSourceItems.value
])

const isMaterialGenerating = computed(() => props.summaryState?.materialStatus === 'generating')
const isRecordingGenerating = computed(() => props.summaryState?.status === 'generating')
const materialError = computed(() => props.summaryState?.materialError || '')

const isBasicItemGenerating = (item) => (
  item.type === 'material'
    ? pendingMaterialSummaryKey.value === item.key && isMaterialGenerating.value
    : pendingRecordingSummaryKey.value === item.key && isRecordingGenerating.value
)

const isBasicSummaryExpanded = (item) => (
  Boolean(item?.summary) && !collapsedBasicSummaryKeys.value.has(item.key)
)

const toggleBasicSummary = (item) => {
  if (!item?.summary) return
  const next = new Set(collapsedBasicSummaryKeys.value)
  if (next.has(item.key)) {
    next.delete(item.key)
  } else {
    next.add(item.key)
  }
  collapsedBasicSummaryKeys.value = next
}

const handleGenerateBasicSummary = (item) => {
  if (!item?.supported || !props.activeFileId) return
  if (item.type === 'material') {
    if (isMaterialGenerating.value) return
    pendingMaterialSummaryKey.value = item.key
    emit('generateMaterialSummary', {
      sessionId: props.activeFileId,
      materials: [item.source],
      summaryLevel: 'detailed',
      summarySentences: 12
    })
    return
  }

  if (isRecordingGenerating.value) return
  pendingRecordingSummaryKey.value = item.key
  emit('generateRecordingSummary', {
    sessionId: props.activeFileId,
    recordingId: item.recordingId,
    recording: item.source,
    summarySentences: 10
  })
}

const handleDeleteSummary = (summaryId) => {
  if (!summaryId) return
  if (!confirm('이 요약을 삭제할까요?')) return
  emit('deleteSummary', summaryId)
}

const getSummarySourceStableId = (source = {}, type = 'source', index = 0) => (
  getSourceId(source, `${type}-${index}`)
)

const materialFromSource = (source = {}) => source.material || (source.type === 'material' ? source : null)
const recordingFromSource = (source = {}) => source.recording || (source.type === 'recording' ? source : null)

const getFileMaterials = (file = {}) => {
  const weekMaterials = Array.isArray(file.weeks)
    ? file.weeks.flatMap((week) => Array.isArray(week?.materials) ? week.materials : [])
    : []
  return weekMaterials.length ? weekMaterials : (Array.isArray(file.attachments) ? file.attachments : [])
}

const getFileRecordings = (file = {}) => {
  const weekRecordings = Array.isArray(file.weeks)
    ? file.weeks.flatMap((week) => Array.isArray(week?.recordings) ? week.recordings : [])
    : []
  return weekRecordings.length ? weekRecordings : (Array.isArray(file.recordings) ? file.recordings : [])
}

const getSummarySourceTitle = (sourceTitle = '', file = null) => {
  const title = String(sourceTitle || '').trim() || '소스'
  const fileTitle = String(file?.name || '').trim()
  return fileTitle ? `${fileTitle} · ${title}` : title
}

const availableSummarySourceItems = computed(() => {
  const sources = []
  const seen = new Set()
  const addSource = (source) => {
    if (!source?.uid || seen.has(source.uid)) return
    seen.add(source.uid)
    sources.push(source)
  }

  const folderFiles = Array.isArray(props.folderFiles) && props.folderFiles.length
    ? props.folderFiles
    : [{
        id: props.activeFileId || 'current',
        name: '',
        attachments: props.currentAttachments,
        recordings: props.currentRecordings
      }]

  folderFiles.forEach((file, fileIndex) => {
    const fileId = file?.id || `file-${fileIndex}`
    getFileMaterials(file).forEach((material, index) => {
      const id = getSummarySourceStableId(material, 'material', index)
      const pdf = isPdfMaterial(material)
      const title = getSourceTitle(material, `강의자료 ${index + 1}`)
      addSource({
        uid: `material:${fileId}:${id}`,
        id,
        fileId,
        fileTitle: file?.name || '',
        type: 'material',
        title: getSummarySourceTitle(title, file),
        icon: pdf ? 'picture_as_pdf' : 'description',
        material,
        disabled: !pdf,
        disabledReason: 'PDF 자료만 요약할 수 있습니다.'
      })
    })

    getFileRecordings(file).forEach((recording, index) => {
      const id = getSummarySourceStableId(recording, 'recording', index)
      const transcriptCount = getRecordingTranscriptions(recording).length
      const title = getSourceTitle(recording, `녹음본 ${index + 1}`)
      addSource({
        uid: `recording:${fileId}:${id}`,
        id,
        fileId,
        fileTitle: file?.name || '',
        type: 'recording',
        title: getSummarySourceTitle(title, file),
        icon: 'graphic_eq',
        recording,
        recordingId: recording?.id || recording?.recordingId || id,
        transcriptCount,
        disabled: !hasRecordingTranscript(recording),
        disabledReason: file?.resourcesLoaded === false ? '전사문을 불러오는 중입니다.' : '전사된 녹음본만 요약할 수 있습니다.'
      })
    })
  })

  return sources
})

const selectedSourceItems = computed(() => (
  hasPickedSummarySources.value
    ? availableSummarySourceItems.value.filter((source) => (
        localSummarySourceIds.value.includes(source.uid) && !source.disabled
      ))
    : (Array.isArray(props.summarySource?.sources) ? props.summarySource.sources : [])
))

const selectedCombinedMaterials = computed(() => (
  selectedSourceItems.value
    .map((source) => materialFromSource(source))
    .filter((material) => material && isPdfMaterial(material))
))

const selectedCombinedRecordings = computed(() => (
  selectedSourceItems.value
    .map((source) => recordingFromSource(source))
    .filter((recording) => recording && hasRecordingTranscript(recording))
))

const selectedCombinedSourceCount = computed(() => selectedSourceItems.value.length)
const visibleSelectedSources = computed(() => selectedSourceItems.value.slice(0, 5))
const hiddenSelectedSourceCount = computed(() => Math.max(0, selectedCombinedSourceCount.value - visibleSelectedSources.value.length))
const combinedRecordingId = computed(() => getRecordingSetKey(selectedCombinedRecordings.value))
const hasSelectedSummarySource = computed(() => selectedCombinedSourceCount.value > 0)
const summarySourceMeta = computed(() => {
  if (!hasSelectedSummarySource.value) return '소스 추가를 눌러 요약할 자료를 선택하세요.'
  const scopeParts = []
  if (selectedCombinedMaterials.value.length) scopeParts.push(`PDF ${selectedCombinedMaterials.value.length}개`)
  if (selectedCombinedRecordings.value.length) scopeParts.push(`전사 ${selectedCombinedRecordings.value.length}개`)
  if (!scopeParts.length) return '요약할 수 있는 PDF 또는 전사 소스가 없습니다.'
  return `선택자료 ${selectedCombinedSourceCount.value}개 · ${scopeParts.join(' · ')} 연결 됨`
})

const combinedMaterialSummary = computed(() => {
  if (!selectedCombinedMaterials.value.length) return null
  const targetKey = getMaterialSetKey(selectedCombinedMaterials.value)
  return getLatestByDate(materialSummaries.value.filter((summary) => (
    Array.isArray(summary.sourceMaterials) &&
    summary.sourceMaterials.length === selectedCombinedMaterials.value.length &&
    getMaterialSetKey(summary.sourceMaterials) === targetKey
  )))
})

const combinedRecordingSummary = computed(() => {
  if (!combinedRecordingId.value) return null
  return recordingSummaries.value.find((summary) => summary.recordingId === combinedRecordingId.value) || null
})

const canGenerateCombinedSummary = computed(() => (
  Boolean(props.activeFileId) &&
  (selectedCombinedMaterials.value.length > 0 || selectedCombinedRecordings.value.length > 0) &&
  !isMaterialGenerating.value &&
  !isRecordingGenerating.value
))

const isCombinedGenerating = computed(() => (
  (pendingMaterialSummaryKey.value === 'combined' && isMaterialGenerating.value) ||
  (pendingRecordingSummaryKey.value === 'combined' && isRecordingGenerating.value)
))

const handleGenerateCombinedSummary = () => {
  if (!canGenerateCombinedSummary.value) return

  if (selectedCombinedMaterials.value.length) {
    pendingMaterialSummaryKey.value = 'combined'
    emit('generateMaterialSummary', {
      sessionId: props.activeFileId,
      materials: selectedCombinedMaterials.value,
      summaryLevel: 'detailed',
      summarySentences: 12
    })
  }

  if (selectedCombinedRecordings.value.length) {
    pendingRecordingSummaryKey.value = 'combined'
    emit('generateRecordingSummary', {
      sessionId: props.activeFileId,
      recordingId: combinedRecordingId.value,
      recordings: selectedCombinedRecordings.value,
      summarySentences: 10
    })
  }
}

const getSummarySourceIcon = (source = {}) => {
  if (source.type === 'recording') return 'graphic_eq'
  if (isPdfMaterial(materialFromSource(source))) return 'picture_as_pdf'
  return 'draft'
}

const getSummaryAvailableSourceMeta = (source = {}) => {
  if (source.disabled) return source.disabledReason || '요약에 사용할 수 없습니다.'
  if (source.type === 'recording') return `전사 ${source.transcriptCount || getRecordingTranscriptions(source.recording).length}개`
  if (isPdfMaterial(materialFromSource(source))) return 'PDF 자료'
  return '강의자료'
}

const getSummarySourceIdentity = (source = {}) => (
  `${source.type || 'source'}:${source.id || source.materialId || source.recordingId || source.title || ''}`
)

const syncLocalSummarySourcesFromExternal = () => {
  const selectedIdentities = new Set(selectedSourceItems.value.map(getSummarySourceIdentity))
  localSummarySourceIds.value = availableSummarySourceItems.value
    .filter((source) => selectedIdentities.has(getSummarySourceIdentity(source)) && !source.disabled)
    .map((source) => source.uid)
}

const isSummarySourceSelected = (source = {}) => localSummarySourceIds.value.includes(source.uid)
const selectableSummarySourceIds = computed(() => (
  availableSummarySourceItems.value
    .filter((source) => !source.disabled)
    .map((source) => source.uid)
))
const areAllSummarySourcesSelected = computed(() => (
  selectableSummarySourceIds.value.length > 0 &&
  selectableSummarySourceIds.value.every((uid) => localSummarySourceIds.value.includes(uid))
))

const updateSummarySourcePickerAnchor = (target) => {
  const rect = target?.getBoundingClientRect?.()
  if (!rect) return
  const popoverWidth = Math.min(360, Math.max(280, window.innerWidth - 48))
  const gap = 10
  summarySourcePickerAnchor.value = {
    left: Math.min(rect.right + gap, Math.max(16, window.innerWidth - popoverWidth - 16)),
    top: Math.min(Math.max(16, rect.top), Math.max(16, window.innerHeight - 536))
  }
}

const toggleSummarySourcePicker = (event) => {
  if (!isSummarySourcePickerOpen.value && !hasPickedSummarySources.value) {
    syncLocalSummarySourcesFromExternal()
  }
  if (!isSummarySourcePickerOpen.value) {
    updateSummarySourcePickerAnchor(event?.currentTarget)
  }
  isSummarySourcePickerOpen.value = !isSummarySourcePickerOpen.value
}

const closeSummarySourcePicker = () => {
  isSummarySourcePickerOpen.value = false
}

const handleSummarySourcePickerBackdrop = () => {
  closeSummarySourcePicker()
}

const toggleSummarySourceSelection = (source = {}) => {
  if (!source || source.disabled) return
  hasPickedSummarySources.value = true
  const selected = new Set(localSummarySourceIds.value)
  if (selected.has(source.uid)) selected.delete(source.uid)
  else selected.add(source.uid)
  localSummarySourceIds.value = Array.from(selected)
}

const clearSummarySourceSelection = () => {
  hasPickedSummarySources.value = true
  localSummarySourceIds.value = []
}

const selectAllSummaryFolderSources = () => {
  hasPickedSummarySources.value = true
  localSummarySourceIds.value = areAllSummarySourcesSelected.value
    ? []
    : selectableSummarySourceIds.value
}

const formatSpeakerLabel = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  const match = normalized.match(/^SPEAKER_(\d+)$/i)
  if (match) return `화자 ${Number.parseInt(match[1], 10) + 1}`
  if (!normalized || normalized === 'UNKNOWN') return '화자 미상'
  return normalized
}

const isUnknownSpeaker = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  return !normalized || normalized === 'UNKNOWN' || normalized === '화자 미상'
}

const getSpeakerAccent = (speakerLabel = '') => {
  if (speakerLabel === '나' || speakerLabel === '화자 1') return 'blue'
  if (speakerLabel === '화자 B' || speakerLabel === '화자 2') return 'rose'
  if (speakerLabel === '화자 C' || speakerLabel === '화자 3') return 'green'
  return 'amber'
}

const getSpeakerAvatarSrc = (speakerLabel = '') => {
  if (speakerLabel === '화자 B' || speakerLabel === '화자 2') return '/images/man1.png'
  if (speakerLabel === '화자 3') return '/images/woman2.png'
  if (speakerLabel === '화자 4') return '/images/man2.png'
  return '/images/woman1.png'
}

const speakerOrderTranscriptions = computed(() => {
  const liveTranscriptions = Array.isArray(props.transcriptions)
    ? props.transcriptions.filter((transcription) => getTranscriptText(transcription))
    : []
  if (liveTranscriptions.length) return liveTranscriptions

  const recordingId = props.summaryState?.recordingId || ''
  const recording = props.currentRecordings.find((item) => (
    recordingId && (item?.id === recordingId || item?.recordingId === recordingId)
  ))
  return Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
})

const keywordStopwords = new Set([
  '그리고', '그러나', '그래서', '하지만', '또는', '있는', '없는', '합니다', '했습니다',
  '있습니다', '됩니다', '대한', '대해', '너무', '그냥', '현재', '이제', '오늘',
  '수업', '강의', '내용', '부분', '요약', '녹음', '전사', '정리', '이번', '다음',
  '이전', '우리', '제가', '저는', '하는', '하면', '된다', '된다면', '것은', '것이',
  '것을', '것도', '이런', '저런', '그런', '있고', '있다', '없다', '때문'
])

const normalizeKeywordToken = (token = '') => (
  String(token || '')
    .trim()
    .replace(/(입니다|합니다|했습니다|됩니다|되었다|하였다|했다|하고|라는|이라고|께서|에서는|에서|으로|로써|부터|까지|에게|한테|보다|처럼|만큼|마다|이나|거나|와|과|을|를|은|는|이|가|의|도|만|로|에)$/u, '')
)

const extractKeywords = (text = '', limit = 8) => {
  const keywordMap = new Map()
  String(text || '')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .map(normalizeKeywordToken)
    .filter((token) => token.length >= 2 && !keywordStopwords.has(token))
    .forEach((token) => {
      if (!keywordMap.has(token)) {
        keywordMap.set(token, { keyword: token, count: 0, firstIndex: keywordMap.size })
      }
      keywordMap.get(token).count += 1
    })

  return Array.from(keywordMap.values())
    .sort((left, right) => (
      right.count - left.count ||
      right.keyword.length - left.keyword.length ||
      left.firstIndex - right.firstIndex
    ))
    .slice(0, limit)
    .map((item) => item.keyword)
}

const getSegmentTimeValue = (segment = {}, transcription = {}) => {
  const candidates = [
    segment.start,
    segment.startTime,
    segment.start_time,
    transcription.start,
    transcription.startTime,
    transcription.start_time
  ]
  const numeric = candidates.find((candidate) => Number.isFinite(Number(candidate)))
  if (numeric !== undefined) return formatRecordingOffset(numeric)
  return formatRecordingOffset(segment.time || transcription.time || props.recordingTimeText)
}

const liveFlowEntries = computed(() => {
  const entries = []

  speakerOrderTranscriptions.value.forEach((transcription, transcriptionIndex) => {
    const segments = Array.isArray(transcription?.segments) && transcription.segments.length
      ? transcription.segments
      : [transcription]

    segments.forEach((segment, segmentIndex) => {
      const text = String(segment?.text || getTranscriptText(transcription) || '').replace(/\s+/g, ' ').trim()
      if (!text) return

      const speakerId = String(
        segment?.speakerId ||
        segment?.speaker_id ||
        transcription?.speakerId ||
        transcription?.speaker_id ||
        ''
      ).trim()
      const speaker = String(segment?.speaker || transcription?.speaker || '').trim()

      entries.push({
        key: `${transcriptionIndex}-${segmentIndex}-${text.slice(0, 16)}`,
        text,
        speakerId,
        speaker,
        displaySpeaker: getDisplaySpeakerLabel(speakerId, speaker || formatSpeakerLabel(speakerId)),
        time: getSegmentTimeValue(segment, transcription)
      })
    })
  })

  return entries
})

const getRecentUniqueFlowItems = (entries = [], limit = 5) => {
  const seen = new Set()
  return [...entries]
    .reverse()
    .filter((entry) => {
      const key = entry.text.replace(/\s+/g, ' ').trim()
      if (!key || seen.has(key)) return false
      seen.add(key)
      return true
    })
    .slice(0, limit)
    .reverse()
}

const isLiveFlowExpanded = (key = '') => !collapsedLiveFlowKeys.value.has(key)

const toggleLiveFlow = (key = '') => {
  const next = new Set(collapsedLiveFlowKeys.value)
  if (next.has(key)) {
    next.delete(key)
  } else {
    next.add(key)
  }
  collapsedLiveFlowKeys.value = next
}

const sessionFlowItems = computed(() => getRecentUniqueFlowItems(liveFlowEntries.value))

const sessionKeywordItems = computed(() => extractKeywords([
  sessionSummary.value?.summary || '',
  ...sessionFlowItems.value.map((item) => item.text)
].join(' ')))

const getSpeakerFlowItemsForSpeaker = (speaker = {}) => {
  const speakerTokens = new Set([
    speaker.speakerId,
    speaker.key,
    speaker.label
  ].map((value) => String(value || '').trim()).filter(Boolean))

  const entries = liveFlowEntries.value.filter((entry) => (
    speakerTokens.has(entry.speakerId) ||
    speakerTokens.has(entry.speaker) ||
    speakerTokens.has(entry.displaySpeaker)
  ))

  return getRecentUniqueFlowItems(entries)
}

const getSpeakerKeywordItemsForSpeaker = (speaker = {}, flowItems = []) => extractKeywords([
  speaker.summary || '',
  speaker.latestText || '',
  ...flowItems.map((item) => item.text)
].join(' '))

const speakerDisplayMap = computed(() => {
  const speakerMap = new Map()
  speakerOrderTranscriptions.value.forEach((transcription) => {
    const speakerId = String(transcription?.speakerId || transcription?.speaker || '').trim()
    if (isUnknownSpeaker(speakerId) || speakerMap.has(speakerId)) return
    speakerMap.set(speakerId, {
      label: `화자 ${speakerMap.size + 1}`,
      order: speakerMap.size
    })
  })
  return speakerMap
})

const getDisplaySpeakerLabel = (speakerId = '', fallbackLabel = '') => {
  const normalized = String(speakerId || '').trim()
  return speakerDisplayMap.value.get(normalized)?.label || fallbackLabel || formatSpeakerLabel(normalized)
}

const getDisplaySpeakerOrder = (speakerId = '', fallbackIndex = 0) => {
  const normalized = String(speakerId || '').trim()
  const mappedOrder = speakerDisplayMap.value.get(normalized)?.order
  if (Number.isFinite(mappedOrder)) return mappedOrder
  const match = normalized.match(/^SPEAKER_(\d+)$/i)
  if (match) return Number.parseInt(match[1], 10)
  return fallbackIndex
}

const buildMockSpeakerSummary = (utterances) => {
  const texts = utterances.map((utterance) => utterance.text).filter(Boolean)
  if (texts.length <= 1) return texts[0] || ''
  return texts.slice(-2).join(' ')
}

const speakerSummaryItems = computed(() => {
  const speakerMap = new Map()
  props.transcriptions.forEach((transcription, index) => {
    const text = getTranscriptText(transcription)
    if (!text) return
    const rawSpeakerId = transcription.speakerId || transcription.speaker || ''
    if (isUnknownSpeaker(rawSpeakerId)) return

    const speakerLabel = getDisplaySpeakerLabel(rawSpeakerId, transcription.speaker || formatSpeakerLabel(rawSpeakerId))
    const speakerKey = rawSpeakerId || `${props.recordingMode === 'meeting' ? 'unknown-speaker' : 'me'}-${index}`
    if (!speakerMap.has(speakerKey)) {
      speakerMap.set(speakerKey, {
        key: speakerKey,
        speakerId: rawSpeakerId,
        label: speakerLabel,
        firstIndex: index,
        utterances: []
      })
    }
    speakerMap.get(speakerKey).utterances.push({ text, time: transcription.time || '', order: index })
  })

  return Array.from(speakerMap.values()).map((speaker) => {
    const latestUtterance = speaker.utterances[speaker.utterances.length - 1]
    const accent = getSpeakerAccent(speaker.label)
    return {
      ...speaker,
      accent: {
        avatar: `speaker-summary-avatar-${accent}`,
        dot: `speaker-summary-dot-${accent}`
      },
      avatarSrc: getSpeakerAvatarSrc(speaker.label),
      utteranceCount: speaker.utterances.length,
      summary: buildMockSpeakerSummary(speaker.utterances),
      latestText: latestUtterance?.text || '',
      lastUpdatedAt: latestUtterance?.time || ''
    }
  }).sort((left, right) => left.firstIndex - right.firstIndex)
})

const backendSpeakerSummaryItems = computed(() => {
  const items = Array.isArray(props.summaryState?.speakerSummaries)
    ? props.summaryState.speakerSummaries
    : []

  return items.map((item, index) => {
    const speakerId = item.speakerId || item.label || ''
    const label = getDisplaySpeakerLabel(speakerId, item.label || '화자')
    const accent = getSpeakerAccent(label)
    return {
      id: item.id,
      key: item.key || item.id || `backend-speaker-${index}`,
      label,
      speakerId,
      firstIndex: getDisplaySpeakerOrder(speakerId, index),
      accent: {
        avatar: `speaker-summary-avatar-${accent}`,
        dot: `speaker-summary-dot-${accent}`
      },
      avatarSrc: getSpeakerAvatarSrc(label),
      utteranceCount: null,
      summary: item.summary || '',
      latestText: item.latestText || '',
      lastUpdatedAt: formatSummaryTime(item.createdAt)
    }
  }).sort((left, right) => left.firstIndex - right.firstIndex)
})

const isSpeakerSummaryEnabled = computed(() => (
  props.summaryState?.diarizationEnabled ?? props.diarizationEnabled
) !== false)
const displayedSpeakerSummaryItems = computed(() => {
  if (!isSpeakerSummaryEnabled.value) return []
  return backendSpeakerSummaryItems.value.length ? backendSpeakerSummaryItems.value : speakerSummaryItems.value
})
const displayedSpeakerInsightItems = computed(() => (
  displayedSpeakerSummaryItems.value.map((speaker) => {
    const flowItems = getSpeakerFlowItemsForSpeaker(speaker)
    return {
      ...speaker,
      flowItems,
      keywords: getSpeakerKeywordItemsForSpeaker(speaker, flowItems)
    }
  })
))
const sessionSummary = computed(() => props.summaryState?.sessionSummary || null)
const sessionSummaryTitle = computed(() => {
  const recordingId = sessionSummary.value?.recordingId || props.summaryState?.recordingId || ''
  const recording = props.currentRecordings.find((item) => (
    recordingId && (item?.id === recordingId || item?.recordingId === recordingId)
  ))
  const recordingTitle = recording?.title || recording?.name || ''
  return recordingTitle ? `${recordingTitle} 요약` : '녹음본 요약'
})
const isSummaryLoading = computed(() => props.summaryState?.status === 'loading')
const isFinalRecordingSummaryGenerating = computed(() => (
  !props.isRecording && props.summaryState?.status === 'generating'
))
const isSummaryUpdating = computed(() => props.summaryState?.status === 'updating')
const isLiveSummary = computed(() => props.summaryState?.status === 'live')
const hasSpeakerSummaries = computed(() => displayedSpeakerSummaryItems.value.length > 0)
const hasLiveSummaryContent = computed(() => Boolean(
  sessionSummary.value ||
  hasSpeakerSummaries.value ||
  isFinalRecordingSummaryGenerating.value
))
const isWaitingForFirstLiveSummary = computed(() => (
  props.isRecording &&
  !hasSpeakerSummaries.value &&
  !sessionSummary.value &&
  !isFinalRecordingSummaryGenerating.value
))
const liveEmptyMessage = computed(() => (
  props.summaryState?.error ||
  (props.isRecording || isLiveSummary.value || isSummaryUpdating.value || isFinalRecordingSummaryGenerating.value
    ? '녹음본을 불러오고 있습니다.'
    : '녹음을 시작하면 실시간 요약이 됩니다.')
))

const sanitizeFileName = (value = 'summary') => (
  String(value || 'summary')
    .trim()
    .replace(/[\\/:*?"<>|]+/g, '_')
    .replace(/\s+/g, '_')
    .slice(0, 80) || 'summary'
)

const escapeHtml = (value = '') => (
  String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
)

const markdownToPlainText = (value = '') => (
  String(value || '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[-*]\s+/gm, '• ')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
)

const downloadSummaryDoc = ({ title = '요약', summary = '', sourceLabel = '' } = {}) => {
  const summaryText = markdownToPlainText(summary)
  if (!summaryText) return

  const safeTitle = title || '요약'
  const createdAt = new Date().toLocaleString('ko-KR')
  const bodyHtml = escapeHtml(summaryText).replace(/\n/g, '<br />')
  const sourceHtml = sourceLabel ? `<p class="meta">대상: ${escapeHtml(sourceLabel)}</p>` : ''
  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(safeTitle)}</title>
  <style>
    body { font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; line-height: 1.65; color: #111827; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    .meta { color: #64748b; font-size: 12px; margin: 0 0 18px; }
    .summary { font-size: 14px; white-space: normal; }
  </style>
</head>
<body>
  <h1>${escapeHtml(safeTitle)}</h1>
  <p class="meta">생성일: ${escapeHtml(createdAt)}</p>
  ${sourceHtml}
  <div class="summary">${bodyHtml}</div>
</body>
</html>`

  const blob = new Blob([html], { type: 'application/msword;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${sanitizeFileName(safeTitle)}.doc`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

watch(
  () => props.summaryState?.materialStatus,
  (status) => {
    if (status !== 'generating') pendingMaterialSummaryKey.value = ''
  }
)

watch(
  () => props.summaryState?.status,
  (status) => {
    if (status !== 'generating') pendingRecordingSummaryKey.value = ''
  }
)

watch(
  () => props.activeFileId,
  () => {
    hasPickedSummarySources.value = false
    localSummarySourceIds.value = []
    closeSummarySourcePicker()
  }
)

watch(
  availableSummarySourceItems,
  (sources) => {
    const availableIds = new Set(sources.filter((source) => !source.disabled).map((source) => source.uid))
    localSummarySourceIds.value = localSummarySourceIds.value.filter((uid) => availableIds.has(uid))
    if (!sources.length) closeSummarySourcePicker()
  },
  { deep: true }
)
</script>

<template>
  <section :class="['tab-content note-canvas flex-1 min-h-0 flex flex-col relative overflow-y-auto overflow-x-hidden custom-scrollbar p-10 pt-4', tabAnim]">
    <div class="summary-panel max-w-5xl mx-auto w-full min-h-full">
      <div class="summary-mode-tabs" aria-label="요약 화면 선택">
        <button
          v-for="mode in summaryModes"
          :key="mode.key"
          type="button"
          :class="{ active: summaryMode === mode.key }"
          @click="setSummaryMode(mode.key)"
        >
          <span>{{ mode.label }}</span>
        </button>
      </div>

      <div v-if="materialError" class="summary-error">
        <span class="material-symbols-outlined">error</span>
        <span>{{ materialError }}</span>
      </div>

      <section v-if="summaryMode === 'basic'" class="summary-view">
        <section class="summary-simple-builder" aria-label="요약 만들기 설정">
          <div class="summary-simple-builder-header">
            <div class="summary-simple-title">
              <h2>요약 만들기</h2>
            </div>
            <div class="summary-source-header-actions">
              <button
                v-if="selectedCombinedSourceCount"
                type="button"
                class="summary-source-clear-button"
                @click="clearSummarySourceSelection"
              >
                초기화
              </button>
              <button type="button" class="summary-source-picker-button" @click="toggleSummarySourcePicker">
                <span class="material-symbols-outlined" aria-hidden="true">add</span>
                <span>소스 추가</span>
              </button>
            </div>
          </div>

          <div class="summary-selected-source-box">
            <div class="summary-selected-source-top">
              <span>{{ summarySourceMeta }}</span>
              <strong>{{ selectedCombinedSourceCount }}개</strong>
            </div>
            <div v-if="selectedCombinedSourceCount" class="summary-selected-source-list">
              <span
                v-for="source in visibleSelectedSources"
                :key="source.id || source.materialId || source.recordingId || source.title"
                class="summary-selected-source-chip"
                :class="{ recording: source.type === 'recording', material: source.type !== 'recording' }"
                :title="source.title"
              >
                <span class="material-symbols-outlined">{{ source.icon || getSummarySourceIcon(source) }}</span>
                <span>{{ source.title }}</span>
              </span>
              <span v-if="hiddenSelectedSourceCount" class="summary-selected-source-more">+{{ hiddenSelectedSourceCount }}</span>
            </div>
          </div>

          <Teleport to="body">
            <div
              v-if="isSummarySourcePickerOpen"
              class="summary-source-picker-layer"
              @click.self="handleSummarySourcePickerBackdrop"
            >
              <div
                class="summary-source-picker-popover"
                role="dialog"
                aria-modal="true"
                aria-label="요약 소스 선택"
                :style="{ left: `${summarySourcePickerAnchor.left}px`, top: `${summarySourcePickerAnchor.top}px` }"
              >
                <div class="summary-source-picker-head">
                  <strong>소스 선택</strong>
                  <button type="button" aria-label="소스 선택 닫기" @click="closeSummarySourcePicker">
                    <span class="material-symbols-outlined">close</span>
                  </button>
                </div>

                <div v-if="availableSummarySourceItems.length" class="summary-source-picker-actions">
                  <button type="button" @click="selectAllSummaryFolderSources">
                    {{ areAllSummarySourcesSelected ? '전체 해제' : '같은 폴더 전체 선택' }}
                  </button>
                </div>

                <div v-if="availableSummarySourceItems.length" class="summary-source-picker-list">
                  <button
                    v-for="source in availableSummarySourceItems"
                    :key="source.uid"
                    type="button"
                    :class="[
                      'summary-source-picker-item',
                      {
                        selected: isSummarySourceSelected(source),
                        disabled: source.disabled,
                        'is-recording': source.type === 'recording'
                      }
                    ]"
                    :disabled="source.disabled"
                    :title="source.disabled ? source.disabledReason : source.title"
                    @click="toggleSummarySourceSelection(source)"
                  >
                    <span class="summary-source-picker-icon material-symbols-outlined">{{ source.icon || getSummarySourceIcon(source) }}</span>
                    <span class="summary-source-picker-copy">
                      <strong>{{ source.title }}</strong>
                      <small>{{ getSummaryAvailableSourceMeta(source) }}</small>
                    </span>
                    <span class="summary-source-picker-check material-symbols-outlined">
                      {{ isSummarySourceSelected(source) ? 'check_circle' : 'radio_button_unchecked' }}
                    </span>
                  </button>
                </div>

                <div v-else class="summary-source-picker-empty">
                  현재 세션에 저장된 소스가 없습니다.
                </div>
              </div>
            </div>
          </Teleport>

          <div class="summary-simple-actions">
            <button
              type="button"
              class="summary-simple-primary"
              :disabled="!canGenerateCombinedSummary"
              @click="handleGenerateCombinedSummary"
            >
              {{ isCombinedGenerating ? '요약 중' : '요약하기' }}
            </button>
          </div>
        </section>

        <div v-if="isCombinedGenerating" class="summary-empty compact">
          <LoadingHourglass :size="62" />
          <p>선택한 파일을 요약하고 있습니다.</p>
        </div>

        <div v-if="combinedMaterialSummary" class="speaker-summary-card material-summary-card">
          <div class="speaker-summary-top">
            <div class="speaker-summary-identity">
              <div class="speaker-summary-avatar material-summary-avatar">
                <span class="material-symbols-outlined">library_books</span>
              </div>
              <div class="min-w-0">
                <h3>{{ selectedCombinedMaterials.length > 1 ? '자료 통합요약' : '자료 요약' }}</h3>
                <p>{{ formatSummaryTime(combinedMaterialSummary.createdAt) || '저장된 요약' }}</p>
              </div>
            </div>
            <div class="speaker-summary-actions">
              <button
                type="button"
                class="summary-icon-button summary-download-button"
                title="DOC로 저장"
                @click="downloadSummaryDoc({
                  title: selectedCombinedMaterials.length > 1 ? '자료 통합요약' : '자료 요약',
                  summary: combinedMaterialSummary.summary,
                  sourceLabel: summarySourceMeta
                })"
              >
                <span class="material-symbols-outlined">download</span>
              </button>
              <button
                v-if="combinedMaterialSummary.id"
                type="button"
                class="summary-icon-button"
                title="요약 삭제"
                @click="handleDeleteSummary(combinedMaterialSummary.id)"
              >
                <span class="material-symbols-outlined">delete</span>
              </button>
            </div>
          </div>
          <div class="material-summary-markdown summary-scroll-body custom-scrollbar" v-html="renderMarkdown(combinedMaterialSummary.summary)"></div>
        </div>

        <div v-if="combinedRecordingSummary" class="speaker-summary-card">
          <div class="speaker-summary-top">
            <div class="speaker-summary-identity">
              <div class="speaker-summary-avatar speaker-summary-avatar-blue">
                <span class="material-symbols-outlined">graphic_eq</span>
              </div>
              <div class="min-w-0">
                <h3>{{ selectedCombinedRecordings.length > 1 ? '녹음 통합요약' : '녹음 요약' }}</h3>
                <p>{{ formatSummaryTime(combinedRecordingSummary.createdAt) || '저장된 요약' }}</p>
              </div>
            </div>
            <div class="speaker-summary-actions">
              <button
                type="button"
                class="summary-icon-button summary-download-button"
                title="DOC로 저장"
                @click="downloadSummaryDoc({
                  title: selectedCombinedRecordings.length > 1 ? '녹음 통합요약' : '녹음 요약',
                  summary: combinedRecordingSummary.summary,
                  sourceLabel: summarySourceMeta
                })"
              >
                <span class="material-symbols-outlined">download</span>
              </button>
              <button
                v-if="combinedRecordingSummary.id"
                type="button"
                class="summary-icon-button"
                title="요약 삭제"
                @click="handleDeleteSummary(combinedRecordingSummary.id)"
              >
                <span class="material-symbols-outlined">delete</span>
              </button>
            </div>
          </div>
          <div class="summary-text material-summary-markdown summary-scroll-body custom-scrollbar" v-html="renderMarkdown(combinedRecordingSummary.summary)"></div>
        </div>
      </section>

      <section v-else-if="summaryMode === 'live'" class="summary-view">
        <div class="summary-view-heading">
          <h2>실시간 요약</h2>
          <p>수업 녹음 중 생성되는 전체 요약과 화자별 요약을 확인합니다.</p>
        </div>

        <div v-if="isSummaryLoading && !hasLiveSummaryContent" class="summary-empty">
          <LoadingHourglass :size="70" />
          <p>저장된 요약을 불러오고 있습니다.</p>
        </div>

        <div v-else-if="!hasLiveSummaryContent" class="summary-empty">
          <LoadingHourglass
            v-if="isWaitingForFirstLiveSummary"
            src="/animations/Loading%20animation.json"
            :size="84"
            fallback-icon="hourglass_top"
          />
          <span v-else class="material-symbols-outlined">mic</span>
          <p>{{ isWaitingForFirstLiveSummary ? '녹음본을 불러오고 있습니다.' : liveEmptyMessage }}</p>
        </div>

        <div v-else class="ai-summary-list">
          <article v-if="sessionSummary || isFinalRecordingSummaryGenerating" class="speaker-summary-card session-summary-card transcription-item-enter">
            <div class="speaker-summary-top">
              <div class="speaker-summary-identity">
                <div class="speaker-summary-avatar speaker-summary-avatar-blue">
                  <span class="material-symbols-outlined">mic</span>
                </div>
                <div class="min-w-0">
                  <h3>{{ sessionSummaryTitle }}</h3>
                  <p>{{ isFinalRecordingSummaryGenerating ? '최종 요약 생성 중' : (formatSummaryTime(sessionSummary?.createdAt) || (isRecording ? '실시간 요약' : '저장된 요약')) }}</p>
                </div>
              </div>

              <div class="speaker-summary-actions">
                <div class="speaker-summary-status">
                  <span class="speaker-summary-dot speaker-summary-dot-blue"></span>
                  <span>{{ isFinalRecordingSummaryGenerating ? '생성 중' : (isRecording && !isRecordingPaused ? '실시간' : '전체') }}</span>
                </div>
                <button
                  v-if="sessionSummary?.summary && !isFinalRecordingSummaryGenerating"
                  type="button"
                  class="summary-icon-button summary-download-button"
                  title="DOC로 저장"
                  @click="downloadSummaryDoc({
                    title: sessionSummaryTitle,
                    summary: sessionSummary.summary,
                    sourceLabel: '녹음본 요약'
                  })"
                >
                  <span class="material-symbols-outlined">download</span>
                </button>
                <button
                  v-if="sessionSummary?.id && !isFinalRecordingSummaryGenerating"
                  type="button"
                  class="summary-icon-button"
                  title="요약 삭제"
                  @click="handleDeleteSummary(sessionSummary.id)"
                >
                  <span class="material-symbols-outlined">delete</span>
                </button>
              </div>
            </div>
            <div v-if="isFinalRecordingSummaryGenerating" class="summary-loading-inline">
              <LoadingHourglass :size="42" />
              <span>최종 요약을 생성하고 있습니다.</span>
            </div>
            <div v-else class="summary-text material-summary-markdown summary-scroll-body custom-scrollbar" v-html="renderMarkdown(sessionSummary.summary)"></div>
          </article>

          <article
            v-for="speaker in displayedSpeakerSummaryItems"
            :key="speaker.key"
            class="speaker-summary-card transcription-item-enter"
            :style="{ animationDelay: `${speaker.firstIndex * 0.05}s` }"
          >
            <div class="speaker-summary-top">
              <div class="speaker-summary-identity">
                <div class="speaker-summary-avatar" :class="speaker.accent.avatar">
                  <img :src="speaker.avatarSrc" :alt="speaker.label" />
                </div>
                <div class="min-w-0">
                  <h3>{{ speaker.label }}</h3>
                  <p>
                    <span v-if="speaker.utteranceCount">발화 {{ speaker.utteranceCount }}개</span>
                    <span v-else>저장된 요약</span>
                    <span v-if="speaker.lastUpdatedAt"> · {{ speaker.lastUpdatedAt }}</span>
                  </p>
                </div>
              </div>

              <div class="speaker-summary-actions">
                <div class="speaker-summary-status">
                  <span :class="['speaker-summary-dot', speaker.accent.dot]"></span>
                  <span>{{ isRecording && !isRecordingPaused ? '실시간' : '요약' }}</span>
                </div>
                <button
                  v-if="speaker.id"
                  type="button"
                  class="summary-icon-button"
                  title="요약 삭제"
                  @click="handleDeleteSummary(speaker.id)"
                >
                  <span class="material-symbols-outlined">delete</span>
                </button>
              </div>
            </div>
            <div class="summary-text material-summary-markdown summary-scroll-body custom-scrollbar" v-html="renderMarkdown(speaker.summary)"></div>
            <div v-if="speaker.latestText" class="speaker-summary-latest">
              <span class="material-symbols-outlined">graphic_eq</span>
              <span>{{ speaker.latestText }}</span>
            </div>
          </article>
        </div>
      </section>

    </div>
  </section>
</template>

<style scoped>
.summary-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  height: 100%;
  padding-bottom: 48px;
}

.summary-mode-tabs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px;
  padding: 4px;
  border: 1px solid #dfe4ec;
  border-radius: 8px;
  background: #e9edf2;
}

.summary-mode-tabs button {
  height: 38px;
  border: 0;
  border-radius: 6px;
  color: #7b8492;
  background: transparent;
  font-size: 13px;
  font-weight: 900;
}

.summary-mode-tabs button.active {
  color: #111827;
  background: #ffffff;
  box-shadow: 0 1px 5px rgba(15, 23, 42, 0.1);
}

.summary-view,
.summary-file-list,
.ai-summary-list {
  display: grid;
  gap: 14px;
}

.summary-view-heading {
  display: grid;
  gap: 4px;
}

.summary-view-heading h2 {
  margin: 0;
  color: #1d1d1f;
  font-size: 22px;
  font-weight: 950;
}

.summary-view-heading p {
  margin: 0;
  color: #8e8e93;
  font-size: 13px;
  font-weight: 800;
}

.summary-file-card,
.speaker-summary-card,
.summary-empty,
.summary-error {
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.045);
}

.summary-file-card,
.speaker-summary-card {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.summary-simple-builder {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 22px;
  border: 1px solid #dfe5ee;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 18px 36px rgba(15, 23, 42, 0.07);
}

.summary-simple-builder-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.summary-simple-title {
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.summary-simple-title h2 {
  margin: 0;
  color: #111111;
  font-size: 22px;
  font-weight: 900;
  line-height: 1.2;
}

.summary-selected-source-box {
  display: grid;
  gap: 8px;
  padding: 14px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f8fafc;
}

.summary-source-header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
}

.summary-source-picker-button,
.summary-source-clear-button {
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 0;
  border-radius: 8px;
  color: #64748b;
  background: #eef2f7;
  font-size: 12px;
  font-weight: 900;
  line-height: 1;
}

.summary-source-picker-button {
  padding: 0 12px;
}

.summary-source-clear-button {
  padding: 0 10px;
}

.summary-source-picker-button:hover,
.summary-source-clear-button:hover {
  background: #e2e8f0;
}

.summary-source-picker-button .material-symbols-outlined {
  font-size: 16px;
}

.summary-simple-actions {
  display: flex;
  justify-content: flex-end;
  padding-top: 16px;
}

.summary-simple-primary {
  height: 40px;
  padding: 0 18px;
  border: 0;
  border-radius: 8px;
  color: #ffffff;
  background: #111111;
  font-size: 13px;
  font-weight: 900;
  box-shadow: none;
  transition: background-color 0.18s ease, transform 0.18s ease;
}

.summary-simple-primary:hover {
  background: #252525;
}

.summary-simple-primary:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.summary-simple-primary:active {
  transform: scale(0.98);
}

.summary-source-picker-layer {
  position: fixed;
  inset: 0;
  z-index: 80;
  background: transparent;
}

.summary-source-picker-popover {
  position: fixed;
  width: min(360px, calc(100vw - 48px));
  max-height: min(520px, calc(100vh - 72px));
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 24px 54px rgba(15, 23, 42, 0.18);
}

.summary-source-picker-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.summary-source-picker-head strong {
  color: #1f2937;
  font-size: 13px;
  font-weight: 900;
}

.summary-source-picker-head button {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 999px;
  color: #64748b;
  background: #f1f5f9;
}

.summary-source-picker-head .material-symbols-outlined {
  font-size: 18px;
}

.summary-source-picker-actions {
  display: flex;
  justify-content: flex-end;
}

.summary-source-picker-actions button {
  height: 30px;
  padding: 0 10px;
  border: 0;
  border-radius: 8px;
  color: #2563eb;
  background: #eff6ff;
  font-size: 11px;
  font-weight: 900;
}

.summary-source-picker-actions button:hover {
  background: #dbeafe;
}

.summary-source-picker-list {
  min-height: 0;
  display: grid;
  gap: 8px;
  overflow-y: auto;
  padding-right: 2px;
}

.summary-source-picker-item {
  min-width: 0;
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) 22px;
  align-items: center;
  gap: 10px;
  padding: 10px;
  border: 0;
  border-radius: 8px;
  background: #f8fafc;
  text-align: left;
  transition: background-color 0.18s ease, opacity 0.18s ease;
}

.summary-source-picker-item:hover {
  background: #eef2f7;
}

.summary-source-picker-item.disabled {
  cursor: not-allowed;
  opacity: 0.46;
}

.summary-source-picker-icon {
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #2563eb;
  background: #dbeafe;
  font-size: 18px;
}

.summary-source-picker-item.is-recording .summary-source-picker-icon {
  color: #f59e0b;
  background: rgba(255, 242, 207, 0.9);
}

.summary-source-picker-copy {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.summary-source-picker-copy strong,
.summary-source-picker-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-source-picker-copy strong {
  color: #111827;
  font-size: 12px;
  font-weight: 900;
}

.summary-source-picker-copy small {
  color: #7b8492;
  font-size: 11px;
  font-weight: 800;
}

.summary-source-picker-check {
  color: #2563eb;
  font-size: 19px;
}

.summary-source-picker-empty {
  padding: 18px 10px;
  border-radius: 8px;
  color: #8e8e93;
  background: #f8fafc;
  font-size: 12px;
  font-weight: 800;
  text-align: center;
}

.summary-file-card.is-disabled {
  background: rgba(248, 250, 252, 0.86);
}

.summary-file-top,
.speaker-summary-top,
.summary-selected-source-top,
.summary-combined-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.summary-selected-source-top {
  color: #475569;
  font-size: 12px;
  font-weight: 900;
}

.summary-selected-source-top span {
  margin: 0;
  color: inherit;
  font-size: 12px;
  font-weight: 900;
}

.summary-selected-source-top strong {
  color: #111827;
  font-size: 13px;
  font-weight: 900;
}

.summary-file-identity,
.speaker-summary-identity {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.summary-file-icon {
  width: 42px;
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #2563eb;
  background: #eef4ff;
  font-size: 23px;
}

.summary-file-icon.recording {
  color: #075985;
  background: #e0f2fe;
}

.summary-file-identity h3,
.speaker-summary-identity h3 {
  margin: 0;
  overflow: hidden;
  color: #1d1d1f;
  font-size: 14px;
  font-weight: 900;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-file-identity p,
.speaker-summary-identity p {
  margin: 4px 0 0;
  color: #8e8e93;
  font-size: 11px;
  font-weight: 800;
  line-height: 1.2;
}

.summary-file-actions,
.speaker-summary-actions {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.summary-primary-button {
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 0 12px;
  border: 0;
  border-radius: 8px;
  color: #ffffff;
  background: #1d1d1f;
  font-size: 12px;
  font-weight: 900;
}

.summary-primary-button:disabled {
  color: #94a3b8;
  background: #e5e7eb;
  cursor: not-allowed;
}

.summary-primary-button .material-symbols-outlined {
  font-size: 17px;
}

.summary-toggle-button {
  width: 32px;
  height: 32px;
  min-width: 32px;
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 8px;
  color: #64748b;
  background: #f1f5f9;
  transition: background 0.18s ease, color 0.18s ease;
}

.summary-toggle-button:hover {
  color: #111827;
  background: #e2e8f0;
}

.summary-toggle-button .material-symbols-outlined {
  font-size: 22px;
}

.summary-icon-button {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 8px;
  color: #c7c7cc;
  background: transparent;
}

.summary-icon-button:hover {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
}

.summary-download-button {
  color: #64748b;
}

.summary-download-button:hover {
  color: #2563eb;
  background: rgba(37, 99, 235, 0.1);
}

.summary-icon-button .material-symbols-outlined {
  font-size: 18px;
}

.summary-loading-inline,
.summary-empty {
  min-height: 130px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #64748b;
  font-size: 13px;
  font-weight: 850;
}

.summary-empty {
  flex-direction: column;
  padding: 24px;
}

.summary-empty .material-symbols-outlined {
  color: #c7c7cc;
  font-size: 42px;
}

.summary-error {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 10px 12px;
  color: #b91c1c;
  background: #fef2f2;
  font-size: 12px;
  font-weight: 800;
}

.summary-result-body {
  padding-top: 10px;
  border-top: 1px solid rgba(226, 232, 240, 0.9);
}

.summary-text,
.material-summary-markdown {
  margin: 0;
  color: #1f2937;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.7;
  word-break: keep-all;
  overflow-wrap: anywhere;
}

.summary-scroll-body {
  max-height: min(52vh, 560px);
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 8px;
  overscroll-behavior: contain;
}

.material-summary-markdown :deep(p) {
  margin: 0 0 10px;
}

.material-summary-markdown :deep(h2),
.material-summary-markdown :deep(h3) {
  margin: 16px 0 8px;
  color: #111827;
  font-weight: 950;
  line-height: 1.35;
}

.material-summary-markdown :deep(h2) {
  font-size: 16px;
}

.material-summary-markdown :deep(h3) {
  font-size: 14px;
}

.material-summary-markdown :deep(ol),
.material-summary-markdown :deep(ul) {
  display: grid;
  gap: 8px;
  margin: 10px 0 0;
  padding-left: 20px;
}

.material-summary-markdown :deep(strong) {
  color: #111827;
  font-weight: 950;
}

.speaker-summary-avatar {
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  padding: 4px;
  overflow: hidden;
  border: 1.5px solid #1d1d1f;
}

.speaker-summary-avatar img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center bottom;
  border-radius: 999px;
  display: block;
}

.speaker-summary-avatar .material-symbols-outlined {
  color: #1d4ed8;
  font-size: 22px;
}

.speaker-summary-avatar-blue {
  background: #dbeafe;
}

.speaker-summary-avatar-amber {
  background: #fff1d6;
}

.speaker-summary-avatar-rose {
  background: #ffe4ea;
}

.speaker-summary-avatar-green {
  background: #dcfce7;
}

.material-summary-avatar {
  background: #fee2e2;
}

.material-summary-avatar .material-symbols-outlined {
  color: #dc2626;
}

.speaker-summary-status {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #6b7280;
  font-size: 11px;
  font-weight: 900;
}

.speaker-summary-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
}

.speaker-summary-dot-blue {
  background: #2563eb;
}

.speaker-summary-dot-amber {
  background: #d97706;
}

.speaker-summary-dot-rose {
  background: #e11d48;
}

.speaker-summary-dot-green {
  background: #059669;
}

.speaker-summary-latest {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  padding-top: 10px;
  border-top: 1px solid rgba(229, 229, 234, 0.8);
  color: #6b7280;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.5;
}

.speaker-summary-latest .material-symbols-outlined {
  flex: 0 0 auto;
  margin-top: 1px;
  color: #9ca3af;
  font-size: 15px;
}

.material-summary-card {
  border-color: rgba(254, 202, 202, 0.9);
}

.summary-selected-source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.summary-selected-source-chip,
.summary-selected-source-more {
  max-width: 230px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  overflow: hidden;
  padding: 6px 9px;
  border-radius: 8px;
  color: #1f2937;
  background: #eef4ff;
  font-size: 12px;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-selected-source-chip.material {
  color: #2563eb;
  background: rgba(239, 246, 255, 0.92);
}

.summary-selected-source-chip.recording {
  color: #f59e0b;
  background: rgba(255, 242, 207, 0.9);
}

.summary-selected-source-chip .material-symbols-outlined {
  flex: 0 0 auto;
  color: currentColor;
  font-size: 16px;
}

.summary-selected-source-chip span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-selected-source-empty {
  color: #8e8e93;
  font-size: 13px;
  font-weight: 800;
}

@media (max-width: 720px) {
  .summary-mode-tabs {
    grid-template-columns: 1fr;
  }

  .summary-file-top,
  .speaker-summary-top,
  .summary-combined-actions {
    align-items: flex-start;
    flex-direction: column;
  }

  .summary-file-actions {
    width: 100%;
  }

  .summary-primary-button {
    width: 100%;
  }
}
</style>
