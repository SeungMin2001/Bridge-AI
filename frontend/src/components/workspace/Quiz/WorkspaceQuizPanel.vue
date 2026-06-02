<script setup>
import { computed, ref, watch } from 'vue'
import { isWorkspaceUuid } from '../../../api/workspaceApi.js'
import { isPdfMaterial } from '../../../utils/pdfMaterial.js'
import LoadingHourglass from '../../ui/LoadingHourglass.vue'

const QUIZ_API_BASE = '/quiz'

const props = defineProps({
  tabAnim: { type: String, default: 'tab-slide-right' },
  activeFileName: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  currentPreviewMaterial: { type: Object, default: null },
  currentAttachments: { type: Array, default: () => [] },
  currentRecordings: { type: Array, default: () => [] },
  folderFiles: { type: Array, default: () => [] },
  quizSource: { type: Object, default: null }
})

const quizStatus = ref('idle')
const quizListStatus = ref('idle')
const quizError = ref('')
const quizMode = ref('create')
const quizList = ref([])
const activeQuiz = ref(null)
const quizAnswers = ref({})
const quizResult = ref(null)
const activeQuestionPage = ref(0)
const deletingQuizId = ref('')
const isSourcePickerOpen = ref(false)
const sourcePickerAnchor = ref({ left: 0, top: 0 })
const localQuizSourceIds = ref([])
const hasPickedQuizSources = ref(false)
const DEFAULT_QUIZ_COUNT = 5

function buildSingleTypeCounts(type, count = DEFAULT_QUIZ_COUNT) {
  return {
    MULTIPLE_CHOICE: type === 'MULTIPLE_CHOICE' ? count : 0,
    OX: type === 'OX' ? count : 0,
    SHORT_ANSWER: type === 'SHORT_ANSWER' ? count : 0
  }
}

const frontendQuizType = ref('MULTIPLE_CHOICE')
const quizTypeCounts = ref(buildSingleTypeCounts(frontendQuizType.value))

const quizModes = [
  { key: 'create', label: '퀴즈', icon: 'tune' },
  { key: 'list', label: '퀴즈 리스트', icon: 'format_list_bulleted' }
]

const quizTypeOptions = [
  {
    key: 'MULTIPLE_CHOICE',
    label: '객관식',
    icon: 'checklist',
    description: '보기 중 하나를 선택',
    tone: 'blue'
  },
  {
    key: 'OX',
    label: 'O/X',
    icon: 'rule',
    description: '참과 거짓을 판단',
    tone: 'amber'
  },
  {
    key: 'SHORT_ANSWER',
    label: '단답형',
    icon: 'short_text',
    description: '핵심 답안을 직접 입력',
    tone: 'green'
  }
]

const formatQuizApiDetail = (detail, fallback = '') => {
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => item?.msg || item?.message || JSON.stringify(item))
      .join('\n')
  }
  if (typeof detail === 'object') {
    return detail.message || detail.error || JSON.stringify(detail)
  }
  return String(detail)
}

const requestQuizJson = async (endpoint, options = {}) => {
  const response = await fetch(`${QUIZ_API_BASE}${endpoint}`, {
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
      detail = formatQuizApiDetail(parsed.detail || parsed.error, raw)
    } catch {
      detail = raw
    }
    throw new Error(detail || `Quiz API request failed: ${response.status}`)
  }

  return response.json()
}

const postQuizJson = (endpoint, payload) => requestQuizJson(endpoint, {
  method: 'POST',
  body: JSON.stringify(payload)
})

const quizSessionId = computed(() => props.quizSource?.sessionId || props.activeFileId)
const canUseQuiz = computed(() => isWorkspaceUuid(props.activeFileId))
const previewPdfMaterial = computed(() => (
  isPdfMaterial(props.currentPreviewMaterial) ? props.currentPreviewMaterial : null
))

const getSourceStableId = (item = {}, prefix = 'source', index = 0) => (
  item?.id || item?.materialId || item?.recordingId || item?.storedName || item?.url || item?.name || item?.title || `${prefix}-${index}`
)

const getAttachmentTitle = (material = {}, index = 0) => (
  material?.name || material?.title || material?.storedName || `강의자료 ${index + 1}`
)

const getRecordingTitle = (recording = {}, index = 0) => (
  recording?.title || recording?.name || `녹음본 ${index + 1}`
)

const collectRecordingTranscriptIds = (recording = {}) => {
  const ids = new Set()
  const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
  transcriptions.forEach((transcription) => {
    const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
    segments.forEach((segment) => {
      const id = segment?.transcript_id || segment?.transcriptId
      if (isWorkspaceUuid(String(id || '').trim())) ids.add(String(id).trim())
    })
  })
  return Array.from(ids)
}

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

const getQuizSourceTitle = (sourceTitle = '', file = null) => {
  const title = String(sourceTitle || '').trim() || '소스'
  const fileTitle = String(file?.name || '').trim()
  return fileTitle ? `${fileTitle} · ${title}` : title
}

const materialFromSource = (source = {}) => (
  source?.material || (
    source?.type === 'material'
      ? {
          id: source.id,
          name: source.title,
          title: source.title,
          type: source.fileType || source.mimeType || source.contentType || '',
          url: source.url || '',
          storedName: source.storedName || ''
        }
      : null
  )
)

const availableQuizSourceItems = computed(() => {
  const sources = []
  const seen = new Set()
  const seenIdentity = new Set()

  const addSource = (source) => {
    if (!source?.uid || seen.has(source.uid)) return
    const identity = `${source.type || 'source'}:${source.id || source.materialId || source.recordingId || source.title || source.uid}`
    if (seenIdentity.has(identity)) return
    seen.add(source.uid)
    seenIdentity.add(identity)
    sources.push(source)
  }

  const folderFiles = Array.isArray(props.folderFiles) && props.folderFiles.length
    ? props.folderFiles
    : [{
        id: props.activeFileId || 'current',
        name: props.activeFileName || '',
        attachments: props.currentAttachments,
        recordings: props.currentRecordings
      }]

  folderFiles.forEach((file, fileIndex) => {
    const fileId = file?.id || `file-${fileIndex}`
    getFileMaterials(file).forEach((material, index) => {
      const id = getSourceStableId(material, 'material', index)
      const pdf = isPdfMaterial(material)
      const title = getAttachmentTitle(material, index)
      addSource({
        uid: `material:${fileId}:${id}`,
        id,
        fileId,
        fileTitle: file?.name || '',
        type: 'material',
        title: getQuizSourceTitle(title, file),
        icon: pdf ? 'picture_as_pdf' : 'description',
        material,
        transcriptIds: [],
        disabled: !pdf,
        disabledReason: 'PDF 자료만 퀴즈로 만들 수 있습니다.'
      })
    })

    getFileRecordings(file).forEach((recording, index) => {
      const id = getSourceStableId(recording, 'recording', index)
      const transcriptIds = collectRecordingTranscriptIds(recording)
      const title = getRecordingTitle(recording, index)
      addSource({
        uid: `recording:${fileId}:${id}`,
        id,
        fileId,
        fileTitle: file?.name || '',
        type: 'recording',
        title: getQuizSourceTitle(title, file),
        icon: 'graphic_eq',
        recordingId: recording?.id || recording?.recordingId || id,
        recording,
        transcriptIds,
        disabled: transcriptIds.length === 0,
        disabledReason: file?.resourcesLoaded === false ? '전사문을 불러오는 중입니다.' : '전사된 녹음본만 퀴즈로 만들 수 있습니다.'
      })
    })
  })

  if (previewPdfMaterial.value) {
    const id = getSourceStableId(previewPdfMaterial.value, 'preview-material', 0)
    addSource({
      uid: `material:${id}`,
      id,
      type: 'material',
      title: getAttachmentTitle(previewPdfMaterial.value, 0),
      icon: 'picture_as_pdf',
      material: previewPdfMaterial.value,
      transcriptIds: [],
      disabled: false,
      disabledReason: ''
    })
  }

  return sources
})

const externalSelectedSourceItems = computed(() => (
  Array.isArray(props.quizSource?.sources)
    ? props.quizSource.sources
    : (props.quizSource?.title ? [{
        id: props.quizSource?.materialId || props.quizSource?.recordingId || props.quizSource?.title,
        type: props.quizSource?.type || 'source',
        title: props.quizSource?.title,
        material: props.quizSource?.material || null,
        transcriptIds: Array.isArray(props.quizSource?.transcriptIds) ? props.quizSource.transcriptIds : []
      }] : (previewPdfMaterial.value ? [{
        id: previewPdfMaterial.value.id || previewPdfMaterial.value.url || previewPdfMaterial.value.name,
        type: 'material',
        title: previewPdfMaterial.value.name || 'PDF 강의자료',
        material: previewPdfMaterial.value,
        transcriptIds: []
      }] : []))
))

const selectedSourceItems = computed(() => {
  if (!hasPickedQuizSources.value) return externalSelectedSourceItems.value
  const selectedIds = new Set(localQuizSourceIds.value)
  return availableQuizSourceItems.value.filter((source) => selectedIds.has(source.uid) && !source.disabled)
})
const selectedPdfCount = computed(() => selectedSourceItems.value.filter((source) => isPdfMaterial(materialFromSource(source))).length)

const sourceTranscriptIds = computed(() => {
  const ids = new Set()
  selectedSourceItems.value.forEach((source) => {
    ;(source.transcriptIds || []).forEach((id) => {
      const transcriptId = String(id || '').trim()
      if (isWorkspaceUuid(transcriptId)) ids.add(transcriptId)
    })
  })
  return Array.from(ids)
})
const hasSelectedQuizSource = computed(() => selectedSourceItems.value.length > 0)
const hasTranscriptScope = computed(() => sourceTranscriptIds.value.length > 0)
const selectedPdfMaterials = computed(() => {
  const materials = []
  const seen = new Set()

  const addMaterial = (material) => {
    if (!material || !isPdfMaterial(material)) return
    const key = material.id || material.url || material.storedName || material.name
    if (!key || seen.has(key)) return
    seen.add(key)
    materials.push(material)
  }

  selectedSourceItems.value.forEach((source) => addMaterial(materialFromSource(source)))

  return materials
})
const hasPdfScope = computed(() => selectedPdfMaterials.value.length > 0)
const hasGeneratableScope = computed(() => hasTranscriptScope.value || hasPdfScope.value)
const selectedQuizSourceTitle = computed(() => (
  selectedSourceItems.value[0]?.title
  || props.quizSource?.title
  || '파일을 선택하세요'
))
const quizQuestionCount = computed(() => Object.values(quizTypeCounts.value).reduce((sum, count) => sum + Number(count || 0), 0))
const quizCapacityPercent = computed(() => `${Math.min(100, Math.max(0, (quizQuestionCount.value / 20) * 100))}%`)
const hasActiveQuiz = computed(() => Array.isArray(activeQuiz.value?.quiz_data) && activeQuiz.value.quiz_data.length > 0)
const isQuizBusy = computed(() => ['generating', 'submitting'].includes(quizStatus.value))
const selectedSourceCount = computed(() => selectedSourceItems.value.length)
const visibleSelectedSourceItems = computed(() => selectedSourceItems.value.slice(0, 3))
const hiddenSelectedSourceCount = computed(() => Math.max(0, selectedSourceCount.value - visibleSelectedSourceItems.value.length))
const currentQuizQuestions = computed(() => getQuizQuestions())
const currentQuestion = computed(() => currentQuizQuestions.value[activeQuestionPage.value] || null)
const currentQuestionNumber = computed(() => activeQuestionPage.value + 1)
const currentQuestionAnswer = computed(() => (
  currentQuestion.value
    ? quizAnswers.value[String(currentQuestion.value.question_index)] || ''
    : ''
))
const currentQuestionIsGraded = computed(() => (
  currentQuestion.value?.is_correct !== null &&
  currentQuestion.value?.is_correct !== undefined
))
const isLastQuizPage = computed(() => activeQuestionPage.value >= currentQuizQuestions.value.length - 1)
const answeredQuestionCount = computed(() => (
  currentQuizQuestions.value.filter((question) => String(quizAnswers.value[String(question.question_index)] || '').trim()).length
))
const isCurrentQuestionAnswered = computed(() => String(currentQuestionAnswer.value || '').trim().length > 0)
const quizNextButtonLabel = computed(() => (
  quizStatus.value === 'submitting'
    ? '저장 중'
    : currentQuestionIsGraded.value && isLastQuizPage.value
    ? '완료'
    : '다음으로'
))
const canUseQuizNavigation = computed(() => (
  isCurrentQuestionAnswered.value && !quizResult.value && !isQuizBusy.value
))
const shouldShowQuizListLoading = computed(() => (
  quizMode.value === 'list' && quizListStatus.value === 'loading' && quizList.value.length === 0
))
const hasEmptyQuizList = computed(() => (
  quizMode.value === 'list' && quizListStatus.value !== 'loading' && quizList.value.length === 0
))
const canGenerateQuiz = computed(() => (
  isWorkspaceUuid(quizSessionId.value) &&
  hasSelectedQuizSource.value &&
  hasGeneratableScope.value &&
  quizQuestionCount.value >= 1 &&
  quizQuestionCount.value <= 20 &&
  !isQuizBusy.value
))
const selectedSourceMeta = computed(() => {
  if (!hasSelectedQuizSource.value) return '소스 추가를 눌러 퀴즈에 사용할 자료를 선택하세요.'
  if (!hasGeneratableScope.value) return '현재 파일에 연결된 전사 또는 PDF 텍스트가 없습니다.'
  const sourceCount = selectedSourceItems.value.length
  const scopeParts = []
  if (hasPdfScope.value) scopeParts.push(`PDF ${selectedPdfCount.value}개`)
  if (hasTranscriptScope.value) scopeParts.push(`전사 ${sourceTranscriptIds.value.length}개`)
  return `선택자료 ${sourceCount}개 · ${scopeParts.join(' · ')} 연결 됨`
})
const selectedQuizListSourceTitle = computed(() => {
  const firstSource = visibleSelectedSourceItems.value[0]
  if (!firstSource?.title) return ''
  if (selectedSourceCount.value > 1) return `${firstSource.title} 외 +${selectedSourceCount.value - 1}개 소스`
  return firstSource.title
})
const getSourceIcon = (source = {}) => {
  if (source.type === 'recording') return 'graphic_eq'
  if (isPdfMaterial(materialFromSource(source))) return 'picture_as_pdf'
  return 'draft'
}

const getAvailableSourceMeta = (source = {}) => {
  if (source.disabled) return source.disabledReason || '퀴즈 생성에 사용할 수 없습니다.'
  if (source.type === 'recording') return `전사 ${source.transcriptIds?.length || 0}개`
  if (isPdfMaterial(materialFromSource(source))) return 'PDF 자료'
  return '강의자료'
}

const getSourceIdentity = (source = {}) => (
  `${source.type || 'source'}:${source.id || source.materialId || source.recordingId || source.title || ''}`
)

const syncLocalSourcesFromCurrentSelection = () => {
  const selectedIdentities = new Set(externalSelectedSourceItems.value.map(getSourceIdentity))
  localQuizSourceIds.value = availableQuizSourceItems.value
    .filter((source) => selectedIdentities.has(getSourceIdentity(source)) && !source.disabled)
    .map((source) => source.uid)
}

const isQuizSourceSelected = (source) => localQuizSourceIds.value.includes(source.uid)
const selectableQuizSourceIds = computed(() => (
  availableQuizSourceItems.value
    .filter((source) => !source.disabled)
    .map((source) => source.uid)
))
const areAllQuizSourcesSelected = computed(() => (
  selectableQuizSourceIds.value.length > 0 &&
  selectableQuizSourceIds.value.every((uid) => localQuizSourceIds.value.includes(uid))
))

const updateSourcePickerAnchor = (target) => {
  const rect = target?.getBoundingClientRect?.()
  if (!rect) return
  const popoverWidth = Math.min(360, Math.max(280, window.innerWidth - 48))
  const gap = 10
  sourcePickerAnchor.value = {
    left: Math.min(rect.right + gap, Math.max(16, window.innerWidth - popoverWidth - 16)),
    top: Math.min(Math.max(16, rect.top), Math.max(16, window.innerHeight - 536))
  }
}

const toggleSourcePicker = (event) => {
  if (!isSourcePickerOpen.value && !hasPickedQuizSources.value) {
    syncLocalSourcesFromCurrentSelection()
  }
  if (!isSourcePickerOpen.value) {
    updateSourcePickerAnchor(event?.currentTarget)
  }
  isSourcePickerOpen.value = !isSourcePickerOpen.value
}

const closeSourcePicker = () => {
  isSourcePickerOpen.value = false
}

const toggleQuizSourceSelection = (source) => {
  if (!source || source.disabled) return
  hasPickedQuizSources.value = true
  const selected = new Set(localQuizSourceIds.value)
  if (selected.has(source.uid)) selected.delete(source.uid)
  else selected.add(source.uid)
  localQuizSourceIds.value = Array.from(selected)
}

const handleSourcePickerBackdrop = () => {
  closeSourcePicker()
}

const clearQuizSourceSelection = () => {
  hasPickedQuizSources.value = true
  localQuizSourceIds.value = []
}

const selectAllQuizFolderSources = () => {
  hasPickedQuizSources.value = true
  localQuizSourceIds.value = areAllQuizSourcesSelected.value
    ? []
    : selectableQuizSourceIds.value
}

const getQuizQuestions = (quiz = activeQuiz.value) => (
  Array.isArray(quiz?.quiz_data) ? quiz.quiz_data : []
)

const formatQuizDate = (value = '') => {
  if (!value) return '방금 생성됨'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return '저장된 퀴즈'
  return parsed.toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getQuizTypeLabel = (type = '') => {
  if (type === 'MULTIPLE_CHOICE') return '객관식'
  if (type === 'OX') return 'O/X'
  if (type === 'SHORT_ANSWER') return '단답형'
  return '문항'
}

const countQuizTypesFromQuestions = (questions = []) => (
  questions.reduce((counts, question) => {
    if (question?.type && counts[question.type] !== undefined) {
      counts[question.type] += 1
    }
    return counts
  }, {
    MULTIPLE_CHOICE: 0,
    OX: 0,
    SHORT_ANSWER: 0
  })
)

const getQuizListTypeLabel = (quiz = {}) => {
  const counts = quiz.type_counts || countQuizTypesFromQuestions(getQuizQuestions(quiz))
  const entries = Object.entries(counts).filter(([, count]) => Number(count || 0) > 0)
  if (entries.length === 1) return getQuizTypeLabel(entries[0][0])
  if (entries.length > 1) return '혼합형'
  return '퀴즈'
}

const getQuizListTitle = (quiz = {}) => {
  return quiz.source_title || quiz.sourceTitle || '소스 정보 없음'
}

const getQuizListTypeText = (quiz = {}) => {
  const totalQuestions = quiz.total_questions || getQuizQuestions(quiz).length || 0
  return `${getQuizListTypeLabel(quiz)} ${totalQuestions}문항`
}

const getQuizSummaryLabel = (quiz) => {
  if (quiz?.correct_count === null || quiz?.correct_count === undefined) return '미응시'
  return `${quiz.correct_count}/${quiz.total_questions || getQuizQuestions(quiz).length} 정답`
}

const buildQuizListItem = (quiz) => ({
  quiz_id: quiz.quiz_id,
  session_id: quiz.session_id || props.activeFileId,
  total_questions: quiz.total_questions || getQuizQuestions(quiz).length,
  correct_count: quiz.correct_count ?? null,
  type_counts: quiz.type_counts || countQuizTypesFromQuestions(getQuizQuestions(quiz)),
  source_title: quiz.source_title || '',
  created_at: quiz.created_at || new Date().toISOString(),
  is_demo: !!quiz.is_demo,
  quiz_data: quiz.is_demo ? getQuizQuestions(quiz) : undefined
})

const resetQuizState = () => {
  quizStatus.value = 'idle'
  quizListStatus.value = 'idle'
  quizError.value = ''
  isSourcePickerOpen.value = false
  localQuizSourceIds.value = []
  hasPickedQuizSources.value = false
  quizList.value = []
  activeQuiz.value = null
  quizAnswers.value = {}
  quizResult.value = null
  activeQuestionPage.value = 0
  quizMode.value = 'create'
}

const resetActiveQuiz = () => {
  activeQuiz.value = null
  quizAnswers.value = {}
  quizResult.value = null
  activeQuestionPage.value = 0
}

const hydrateQuizAnswers = (questions = []) => {
  quizAnswers.value = questions.reduce((answers, question) => {
    if (question?.user_answer) {
      answers[String(question.question_index)] = question.user_answer
    }
    return answers
  }, {})
}

const setActiveQuiz = (quiz) => {
  activeQuiz.value = quiz
  activeQuestionPage.value = 0
  const questions = getQuizQuestions(quiz)
  hydrateQuizAnswers(questions)
  quizResult.value = quiz?.correct_count === null || quiz?.correct_count === undefined
    ? null
    : {
        correct_count: quiz.correct_count,
        total_questions: quiz.total_questions || questions.length,
        score: questions.length ? Math.round((quiz.correct_count / questions.length) * 1000) / 10 : 0
      }
}

const loadQuizDetail = async (quizId, nextMode = 'create') => {
  if (!quizId) return
  quizListStatus.value = 'loading'
  quizError.value = ''
  try {
    const quiz = await requestQuizJson(`/${quizId}`)
    setActiveQuiz(quiz)
    quizMode.value = nextMode
    quizListStatus.value = 'done'
  } catch (error) {
    quizListStatus.value = 'error'
    quizError.value = error?.message || '퀴즈를 불러오지 못했습니다.'
  }
}

const loadQuizzesForSession = async (nextMode = 'list') => {
  if (!canUseQuiz.value) {
    resetQuizState()
    return
  }

  quizListStatus.value = 'loading'
  quizError.value = ''
  try {
    const result = await requestQuizJson(`/session/${props.activeFileId}`)
    quizList.value = Array.isArray(result.quizzes)
      ? result.quizzes.map((quiz) => buildQuizListItem(quiz))
      : []
    quizMode.value = nextMode
    quizListStatus.value = 'done'
  } catch (error) {
    quizListStatus.value = 'error'
    quizError.value = error?.message || '퀴즈 목록을 불러오지 못했습니다.'
  }
}

const setQuizMode = async (mode) => {
  quizError.value = ''
  quizMode.value = mode
  if (mode === 'list') {
    await loadQuizzesForSession('list')
  }
}

const startNewQuiz = () => {
  quizError.value = ''
  if (quizStatus.value === 'generating') {
    quizMode.value = 'create'
    return
  }
  quizStatus.value = 'idle'
  resetActiveQuiz()
  quizMode.value = 'create'
}

const returnToQuizBuilder = () => {
  quizStatus.value = 'idle'
  resetActiveQuiz()
  quizMode.value = 'create'
}

const applyPreset = (total) => {
  const mc = Math.max(1, Math.floor(total * 0.5))
  const ox = total >= 3 ? Math.max(1, Math.floor(total * 0.3)) : 0
  const shortAnswer = Math.max(0, total - mc - ox)
  quizTypeCounts.value = {
    MULTIPLE_CHOICE: mc,
    OX: ox,
    SHORT_ANSWER: shortAnswer
  }
}

const adjustTypeCount = (type, delta) => {
  const current = Number(quizTypeCounts.value[type] || 0)
  const next = Math.max(0, current + delta)
  const nextTotal = quizQuestionCount.value - current + next
  if (nextTotal > 20) return
  quizTypeCounts.value = {
    ...quizTypeCounts.value,
    [type]: next
  }
}

watch(frontendQuizType, (type) => {
  quizTypeCounts.value = buildSingleTypeCounts(type)
})

const generateQuizForSource = async () => {
  if (!canGenerateQuiz.value) return

  quizStatus.value = 'generating'
  quizError.value = ''
  quizResult.value = null
  try {
    const sourcePayload = {
      session_id: quizSessionId.value,
      num_questions: quizQuestionCount.value,
      type_counts: quizTypeCounts.value,
      source_title: selectedQuizListSourceTitle.value
    }
    const materialPayload = {
      material_ids: selectedPdfMaterials.value.map((material) => material.id).filter(Boolean),
      stored_names: selectedPdfMaterials.value.map((material) => material.storedName).filter(Boolean)
    }
    const transcriptPayload = {
      transcript_ids: sourceTranscriptIds.value
    }

    const quiz = hasPdfScope.value && hasTranscriptScope.value
      ? await postQuizJson('/generate/sources', {
          ...sourcePayload,
          ...materialPayload,
          ...transcriptPayload
        })
      : hasPdfScope.value
      ? await postQuizJson('/generate/materials', {
          ...sourcePayload,
          ...materialPayload
        })
      : await postQuizJson('/generate/transcripts', {
          ...sourcePayload,
          ...transcriptPayload
        })
    const generatedQuiz = {
      ...quiz,
      session_id: quizSessionId.value,
      correct_count: null,
      source_title: selectedQuizListSourceTitle.value
    }
    setActiveQuiz(generatedQuiz)
    quizList.value = [
      buildQuizListItem(generatedQuiz),
      ...quizList.value.filter((item) => item.quiz_id !== quiz.quiz_id)
    ]
    quizMode.value = 'create'
    quizStatus.value = 'done'
  } catch (error) {
    quizStatus.value = 'error'
    quizError.value = error?.message || '퀴즈 생성에 실패했습니다.'
  }
}

const buildDemoQuestions = () => {
  const sourceTitle = selectedQuizSourceTitle.value === '파일을 선택하세요'
    ? '선택한 자료'
    : selectedQuizSourceTitle.value

  return [
    {
      question_index: 1,
      type: 'MULTIPLE_CHOICE',
      question: `A대학교 교직원이 학생들의 자퇴율을 줄이기 위해 취업률 데이터를 분석하려고 합니다. 이때 두 변수 간의 연관성을 분석하기 위해 어떤 모델을 사용하는 것이 가장 적합합니까?`,
      options: [
        '취업률과 자퇴율을 동시에 분석하는 회귀 분석 모델',
        '취업률과 자퇴율을 개별적으로 학습하는 군집화 모델',
        'AI가 생성한 가짜 자퇴 데이터를 필터링하는 딥러닝 모델',
        '자퇴생의 감정 상태를 다중 클래스로 분류하는 나이브 베이즈 모델'
      ],
      correct_answer: '취업률과 자퇴율을 동시에 분석하는 회귀 분석 모델',
      explanation: '두 변수 간의 관계를 분석할 때는 회귀 분석 모델이 적합합니다. 군집화는 비슷한 특성을 묶는 방식이고, 딥러닝 필터링이나 감정 분류는 이 문제의 목적과 다릅니다.',
      is_correct: null,
      user_answer: null
    },
    {
      question_index: 2,
      type: 'OX',
      question: `${sourceTitle}를 바탕으로 만든 퀴즈는 선택한 자료 범위와 무관하게 생성해도 된다.`,
      options: ['O', 'X'],
      correct_answer: 'X',
      explanation: '자료 기반 퀴즈는 선택한 자료 범위를 기준으로 만들어져야 학습 맥락과 근거가 유지됩니다.',
      is_correct: null,
      user_answer: null
    },
    {
      question_index: 3,
      type: 'SHORT_ANSWER',
      question: `${sourceTitle}에서 퀴즈를 만들 때 선택한 자료 범위를 기준으로 삼아야 하는 핵심 이유를 입력하세요.`,
      options: [],
      correct_answer: '자료의 맥락과 근거를 유지하기 위해서',
      accepted_answers: ['맥락', '근거', '자료 범위', '선택한 자료'],
      explanation: '단답형은 답안에 핵심 키워드가 포함되는지 확인하는 방식으로 데모 채점됩니다.',
      is_correct: null,
      user_answer: null
    }
  ]
}

const generateDemoQuiz = () => {
  const demoQuestions = buildDemoQuestions()
  const demoQuiz = {
    quiz_id: `demo-${Date.now()}`,
    session_id: quizSessionId.value || 'demo-session',
    total_questions: demoQuestions.length,
    correct_count: null,
    created_at: new Date().toISOString(),
    is_demo: true,
    quiz_data: demoQuestions
  }

  setActiveQuiz(demoQuiz)
  quizList.value = [
    buildQuizListItem(demoQuiz),
    ...quizList.value.filter((item) => !item.is_demo)
  ]
  quizError.value = ''
  quizStatus.value = 'done'
  quizMode.value = 'create'
}

const normalizeDemoAnswer = (value) => String(value || '').trim().toLowerCase()

const isDemoAnswerCorrect = (question, answer) => {
  const normalizedAnswer = normalizeDemoAnswer(answer)
  const normalizedCorrectAnswer = normalizeDemoAnswer(question.correct_answer)
  if (!normalizedAnswer) return false
  if (question.type === 'SHORT_ANSWER') {
    const acceptedAnswers = Array.isArray(question.accepted_answers) && question.accepted_answers.length
      ? question.accepted_answers
      : [question.correct_answer]
    return acceptedAnswers.some((acceptedAnswer) => {
      const normalizedAcceptedAnswer = normalizeDemoAnswer(acceptedAnswer)
      return normalizedAnswer.includes(normalizedAcceptedAnswer) || normalizedAcceptedAnswer.includes(normalizedAnswer)
    })
  }
  return normalizedAnswer === normalizedCorrectAnswer
}

const updateQuestionByIndex = (questionIndex, patch) => {
  activeQuiz.value = {
    ...activeQuiz.value,
    quiz_data: getQuizQuestions().map((question) => (
      question.question_index === questionIndex ? { ...question, ...patch } : question
    ))
  }
}

const gradeCurrentQuestion = () => {
  const question = currentQuestion.value
  if (!question || !isCurrentQuestionAnswered.value) return
  const answer = currentQuestionAnswer.value
  updateQuestionByIndex(question.question_index, {
    user_answer: answer,
    is_correct: isDemoAnswerCorrect(question, answer)
  })
}

const finalizePagedQuiz = () => {
  const questions = getQuizQuestions()
  const correctCount = questions.filter((question) => question.is_correct).length
  const totalQuestions = questions.length
  const result = {
    correct_count: correctCount,
    total_questions: totalQuestions,
    score: totalQuestions ? Math.round((correctCount / totalQuestions) * 1000) / 10 : 0
  }

  activeQuiz.value = {
    ...activeQuiz.value,
    correct_count: correctCount,
    total_questions: totalQuestions
  }
  quizResult.value = result
  quizList.value = quizList.value.map((item) => (
    item.quiz_id === activeQuiz.value?.quiz_id
      ? {
          ...item,
          quiz_data: getQuizQuestions(),
          correct_count: correctCount,
          total_questions: totalQuestions
        }
      : item
  ))
}

const goToNextQuizPage = async () => {
  if (!currentQuestion.value || quizResult.value) return
  if (!currentQuestionIsGraded.value) {
    gradeCurrentQuestion()
    return
  }
  if (!isLastQuizPage.value) {
    activeQuestionPage.value += 1
    return
  }
  if (activeQuiz.value?.is_demo) {
    finalizePagedQuiz()
    returnToQuizBuilder()
    return
  }
  await submitQuizAnswers()
  if (quizStatus.value === 'done') {
    returnToQuizBuilder()
  }
}

const resetPagedQuiz = () => {
  activeQuiz.value = {
    ...activeQuiz.value,
    correct_count: null,
    quiz_data: getQuizQuestions().map((question) => ({
      ...question,
      is_correct: null,
      user_answer: null
    }))
  }
  quizAnswers.value = {}
  quizResult.value = null
  activeQuestionPage.value = 0
}

const getQuizOptionClass = (question, option) => {
  const answer = quizAnswers.value[String(question.question_index)] || ''
  const isSelected = answer === option
  const isGraded = question.is_correct !== null && question.is_correct !== undefined
  return {
    selected: isSelected,
    correct: isGraded && option === question.correct_answer,
    wrong: isGraded && isSelected && option !== question.correct_answer
  }
}

const submitDemoQuizAnswers = () => {
  const questions = getQuizQuestions()
  let correctCount = 0
  const gradedQuestions = questions.map((question) => {
    const answer = quizAnswers.value[String(question.question_index)] || ''
    const isCorrect = isDemoAnswerCorrect(question, answer)
    if (isCorrect) correctCount += 1
    return {
      ...question,
      user_answer: answer,
      is_correct: isCorrect
    }
  })
  const totalQuestions = gradedQuestions.length
  const result = {
    correct_count: correctCount,
    total_questions: totalQuestions,
    score: totalQuestions ? Math.round((correctCount / totalQuestions) * 1000) / 10 : 0
  }

  setActiveQuiz({
    ...activeQuiz.value,
    quiz_data: gradedQuestions,
    correct_count: correctCount,
    total_questions: totalQuestions
  })
  quizResult.value = result
  quizList.value = quizList.value.map((item) => (
    item.quiz_id === activeQuiz.value?.quiz_id
      ? {
          ...item,
          quiz_data: gradedQuestions,
          correct_count: correctCount,
          total_questions: totalQuestions
        }
      : item
  ))
  quizStatus.value = 'done'
}

const openQuizFromList = (quiz) => {
  if (quiz?.is_demo) {
    setActiveQuiz(quiz)
    quizMode.value = 'create'
    quizStatus.value = 'done'
    quizError.value = ''
    return
  }
  loadQuizDetail(quiz.quiz_id)
}

const deleteQuizFromList = async (quiz) => {
  if (!quiz?.quiz_id || deletingQuizId.value) return
  if (!confirm(`"${getQuizListTitle(quiz)}"을(를) 삭제할까요?`)) return

  deletingQuizId.value = quiz.quiz_id
  quizError.value = ''
  try {
    if (!quiz.is_demo) {
      await requestQuizJson(`/${quiz.quiz_id}`, { method: 'DELETE' })
    }
    quizList.value = quizList.value.filter((item) => item.quiz_id !== quiz.quiz_id)
    if (activeQuiz.value?.quiz_id === quiz.quiz_id) {
      resetActiveQuiz()
    }
  } catch (error) {
    quizError.value = error?.message || '퀴즈 삭제에 실패했습니다.'
  } finally {
    deletingQuizId.value = ''
  }
}

const setQuizAnswer = (question, answer) => {
  quizAnswers.value = {
    ...quizAnswers.value,
    [String(question.question_index)]: answer
  }
}

const isQuizOptionSelected = (question, option) => (
  quizAnswers.value[String(question.question_index)] === option
)

const submitQuizAnswers = async () => {
  if (!activeQuiz.value?.quiz_id || isQuizBusy.value || quizResult.value) return

  if (activeQuiz.value.is_demo) {
    submitDemoQuizAnswers()
    return
  }

  quizStatus.value = 'submitting'
  quizError.value = ''
  try {
    const result = await postQuizJson(`/${activeQuiz.value.quiz_id}/submit`, {
      answers: quizAnswers.value
    })
    const currentPage = activeQuestionPage.value
    setActiveQuiz({
      ...activeQuiz.value,
      ...result,
      correct_count: result.correct_count
    })
    activeQuestionPage.value = Math.min(currentPage, Math.max(0, getQuizQuestions().length - 1))
    quizResult.value = {
      correct_count: result.correct_count,
      total_questions: result.total_questions,
      score: result.score
    }
    const updatedItem = buildQuizListItem({
      ...activeQuiz.value,
      ...result,
      session_id: activeQuiz.value?.session_id || quizSessionId.value,
      correct_count: result.correct_count
    })
    const hasExistingItem = quizList.value.some((item) => item.quiz_id === result.quiz_id)
    quizList.value = hasExistingItem
      ? quizList.value.map((item) => (
          item.quiz_id === result.quiz_id ? { ...item, ...updatedItem } : item
        ))
      : [updatedItem, ...quizList.value]
    quizStatus.value = 'done'
  } catch (error) {
    quizStatus.value = 'error'
    quizError.value = error?.message || '퀴즈 채점에 실패했습니다.'
  }
}

watch(
  () => props.activeFileId,
  () => {
    resetQuizState()
  },
  { immediate: true }
)

watch(
  () => props.quizSource,
  () => {
    if (quizStatus.value === 'generating') return
    quizError.value = ''
    isSourcePickerOpen.value = false
    hasPickedQuizSources.value = false
    localQuizSourceIds.value = []
    resetActiveQuiz()
    quizMode.value = 'create'
  },
  { deep: true }
)

watch(
  availableQuizSourceItems,
  (sources) => {
    const availableIds = new Set(sources.filter((source) => !source.disabled).map((source) => source.uid))
    localQuizSourceIds.value = localQuizSourceIds.value.filter((id) => availableIds.has(id))
    if (!sources.length) isSourcePickerOpen.value = false
  },
  { deep: true }
)
</script>

<template>
  <section :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-y-auto overflow-x-hidden custom-scrollbar p-10 pt-4', tabAnim]">
    <div class="quiz-panel max-w-5xl mx-auto w-full min-h-full">
      <div class="quiz-mode-tabs" aria-label="퀴즈 화면 선택">
        <button
          v-for="mode in quizModes"
          :key="mode.key"
          type="button"
          :class="{ active: quizMode === mode.key }"
          @click="setQuizMode(mode.key)"
        >
          <span>{{ mode.label }}</span>
        </button>
      </div>

      <div v-if="quizError" class="quiz-error">
        <span class="material-symbols-outlined">error</span>
        <span>{{ quizError }}</span>
      </div>

      <div v-if="!canUseQuiz" class="quiz-empty">
        <span class="material-symbols-outlined">quiz</span>
        <p>워크스페이스 파일을 선택하면 퀴즈를 만들 수 있습니다.</p>
      </div>

      <div v-else-if="quizMode === 'create' && !hasActiveQuiz" class="quiz-create-view">
        <section
          :class="['quiz-simple-builder', { 'is-generating': quizStatus === 'generating' }]"
          aria-label="퀴즈 만들기 설정"
        >
          <div v-if="quizStatus === 'generating'" class="quiz-generating-state">
            <LoadingHourglass
              class="quiz-generating-animation"
              src="/animations/quizflip_loop.json"
              :size="118"
              fallback-icon="quiz"
            />
            <strong>퀴즈 생성중</strong>
          </div>

          <template v-else>
            <div class="quiz-simple-builder-header">
              <div class="quiz-simple-title">
                <h2>퀴즈 만들기</h2>
              </div>
              <div class="quiz-source-header-actions">
                <button
                  v-if="selectedSourceCount"
                  type="button"
                  class="quiz-source-clear-button"
                  @click="clearQuizSourceSelection"
                >
                  초기화
                </button>
                <button type="button" class="quiz-source-picker-button" @click="toggleSourcePicker">
                  <span class="material-symbols-outlined" aria-hidden="true">add</span>
                  <span>소스 추가</span>
                </button>
              </div>
            </div>

            <div class="quiz-selected-source-box">
              <div class="quiz-selected-source-top">
                <span>{{ selectedSourceMeta }}</span>
                <strong>{{ selectedSourceCount }}개</strong>
              </div>
              <div v-if="selectedSourceCount" class="quiz-selected-source-list">
                <span
                  v-for="source in visibleSelectedSourceItems"
                  :key="source.id || source.materialId || source.recordingId || source.title"
                  class="quiz-selected-source-chip"
                  :title="source.title"
                >
                  <span class="material-symbols-outlined" aria-hidden="true">{{ getSourceIcon(source) }}</span>
                  <span>{{ source.title }}</span>
                </span>
                <span v-if="hiddenSelectedSourceCount" class="quiz-selected-source-more">
                  +{{ hiddenSelectedSourceCount }}
                </span>
              </div>
            </div>

            <Teleport to="body">
              <div
                v-if="isSourcePickerOpen"
                class="quiz-source-picker-layer"
                @click.self="handleSourcePickerBackdrop"
              >
                <div
                  class="quiz-source-picker-popover"
                  role="dialog"
                  aria-modal="true"
                  aria-label="소스 선택"
                  :style="{ left: `${sourcePickerAnchor.left}px`, top: `${sourcePickerAnchor.top}px` }"
                >
                  <div class="quiz-source-picker-head">
                    <strong>소스 선택</strong>
                    <button type="button" aria-label="소스 선택 닫기" @click="closeSourcePicker">
                      <span class="material-symbols-outlined">close</span>
                    </button>
                  </div>

                  <div v-if="availableQuizSourceItems.length" class="quiz-source-picker-actions">
                    <button type="button" @click="selectAllQuizFolderSources">
                      {{ areAllQuizSourcesSelected ? '전체 해제' : '같은 폴더 전체 선택' }}
                    </button>
                  </div>

                  <div v-if="availableQuizSourceItems.length" class="quiz-source-picker-list">
                    <button
                      v-for="source in availableQuizSourceItems"
                      :key="source.uid"
                      type="button"
                      :class="[
                        'quiz-source-picker-item',
                        {
                          selected: isQuizSourceSelected(source),
                          disabled: source.disabled,
                          'is-recording': source.type === 'recording'
                        }
                      ]"
                      :disabled="source.disabled"
                      :title="source.disabled ? source.disabledReason : source.title"
                      @click="toggleQuizSourceSelection(source)"
                    >
                      <span class="quiz-source-picker-icon material-symbols-outlined">{{ source.icon || getSourceIcon(source) }}</span>
                      <span class="quiz-source-picker-copy">
                        <strong>{{ source.title }}</strong>
                        <small>{{ getAvailableSourceMeta(source) }}</small>
                      </span>
                      <span class="quiz-source-picker-check material-symbols-outlined">
                        {{ isQuizSourceSelected(source) ? 'check_circle' : 'radio_button_unchecked' }}
                      </span>
                    </button>
                  </div>

                  <div v-else class="quiz-source-picker-empty">
                    현재 세션에 저장된 소스가 없습니다.
                  </div>
                </div>
              </div>
            </Teleport>

            <div class="quiz-simple-form">
              <label class="quiz-simple-field">
                <span>퀴즈 유형</span>
                <span class="quiz-simple-select-wrap">
                  <select v-model="frontendQuizType" aria-label="퀴즈 유형">
                    <option v-for="option in quizTypeOptions" :key="option.key" :value="option.key">
                      {{ option.label }}
                    </option>
                  </select>
                  <span class="material-symbols-outlined" aria-hidden="true">expand_more</span>
                </span>
              </label>
            </div>

            <div class="quiz-simple-actions">
              <button
                type="button"
                class="quiz-simple-primary"
                :disabled="!canGenerateQuiz"
                @click="generateQuizForSource"
              >
                퀴즈 만들기
              </button>
            </div>
          </template>
        </section>
      </div>

      <div v-else-if="quizMode === 'list'" class="quiz-list-view">
        <div class="quiz-list-header">
          <button type="button" class="quiz-secondary-button" @click="startNewQuiz">
            <span class="material-symbols-outlined">add</span>
            <span>새로 만들기</span>
          </button>
        </div>

        <div v-if="shouldShowQuizListLoading" class="quiz-empty compact">
          <span class="material-symbols-outlined">assignment</span>
        </div>

        <div v-else-if="hasEmptyQuizList" class="quiz-empty compact">
          <span class="material-symbols-outlined">assignment</span>
          <p>아직 생성된 퀴즈가 없습니다.</p>
        </div>

        <div v-else class="quiz-session-list custom-scrollbar">
          <article
            v-for="quiz in quizList"
            :key="quiz.quiz_id"
            :class="['quiz-session-card', { active: activeQuiz?.quiz_id === quiz.quiz_id }]"
          >
            <span class="material-symbols-outlined">assignment</span>
            <div class="quiz-session-main">
              <strong>{{ getQuizListTitle(quiz) }}</strong>
              <p>
                <span class="quiz-session-type">{{ getQuizListTypeText(quiz) }}</span>
                <span>{{ formatQuizDate(quiz.created_at) }} · {{ getQuizSummaryLabel(quiz) }}</span>
              </p>
            </div>
            <div class="quiz-session-actions">
              <button type="button" class="quiz-secondary-button" @click="openQuizFromList(quiz)">
                <span class="material-symbols-outlined">play_arrow</span>
                <span>{{ quiz.correct_count === null || quiz.correct_count === undefined ? '풀기' : '다시 보기' }}</span>
              </button>
              <button
                type="button"
                class="quiz-delete-button"
                :disabled="deletingQuizId === quiz.quiz_id"
                :aria-label="`${getQuizListTitle(quiz)} 삭제`"
                @click.stop="deleteQuizFromList(quiz)"
              >
                <span class="material-symbols-outlined">delete</span>
              </button>
            </div>
          </article>
        </div>
      </div>

      <div v-else-if="quizMode === 'create' && hasActiveQuiz" class="quiz-solve-view">
        <div v-if="!hasActiveQuiz" class="quiz-empty compact">
          <span class="material-symbols-outlined">edit_note</span>
          <p>퀴즈 목록에서 풀 퀴즈를 선택하거나 새 퀴즈를 생성하세요.</p>
        </div>

        <template v-else>
          <div v-if="quizResult" class="quiz-result-bar">
            <div>
              <strong>{{ quizResult.score }}점</strong>
              <span>{{ quizResult.correct_count }} / {{ quizResult.total_questions }} 정답</span>
            </div>
            <button type="button" class="quiz-secondary-button" @click="resetPagedQuiz">
              다시 풀기
            </button>
          </div>

          <article
            v-if="currentQuestion"
            :class="[
              'quiz-play-card',
              `type-${currentQuestion.type}`,
              {
                graded: currentQuestionIsGraded,
                correct: currentQuestionIsGraded && currentQuestion.is_correct,
                wrong: currentQuestionIsGraded && !currentQuestion.is_correct
              }
            ]"
          >
            <div class="quiz-play-question">
              <span class="quiz-play-qbadge">Q{{ currentQuestionNumber }}</span>
              <h2>{{ currentQuestion.question }}</h2>
            </div>

            <div v-if="currentQuestion.type === 'SHORT_ANSWER'" class="quiz-play-short-answer">
              <input
                type="text"
                :value="quizAnswers[String(currentQuestion.question_index)] || ''"
                :disabled="currentQuestionIsGraded || !!quizResult"
                placeholder="단답을 입력하세요"
                @input="setQuizAnswer(currentQuestion, $event.target.value)"
                @keydown.enter.prevent="goToNextQuizPage"
              />
            </div>

            <div v-else class="quiz-play-options">
              <button
                v-for="option in currentQuestion.options"
                :key="option"
                type="button"
                :class="['quiz-play-option', getQuizOptionClass(currentQuestion, option)]"
                :disabled="currentQuestionIsGraded || !!quizResult"
                @click="setQuizAnswer(currentQuestion, option)"
              >
                {{ option }}
              </button>
            </div>

            <div v-if="currentQuestionIsGraded" class="quiz-play-feedback">
              <div :class="['quiz-play-grade', currentQuestion.is_correct ? 'correct' : 'wrong']">
                <span class="material-symbols-outlined">
                  {{ currentQuestion.is_correct ? 'check_circle' : 'cancel' }}
                </span>
                <strong>{{ currentQuestion.is_correct ? '정답입니다' : '오답입니다' }}</strong>
              </div>
              <p v-if="!currentQuestion.is_correct">정답: {{ currentQuestion.correct_answer }}</p>
              <p v-if="currentQuestion.explanation">{{ currentQuestion.explanation }}</p>
            </div>
          </article>

          <div class="quiz-play-footer">
            <span>{{ answeredQuestionCount }} / {{ currentQuizQuestions.length }} 응답</span>
            <button
              type="button"
              class="quiz-primary-button"
              :disabled="!canUseQuizNavigation"
              @click="goToNextQuizPage"
            >
              <span class="material-symbols-outlined">arrow_forward</span>
              <span>{{ quizNextButtonLabel }}</span>
            </button>
          </div>
        </template>
      </div>
    </div>
  </section>
</template>

<style scoped>
.quiz-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 100%;
  overflow: visible;
  padding-bottom: 48px;
}

.quiz-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding: 6px 0 0;
}

.quiz-header h1 {
  margin: 0;
  color: #1d1d1f;
  font-size: 30px;
  font-weight: 900;
  line-height: 1.15;
}

.quiz-builder-top p,
.quiz-list-header p,
.quiz-solve-top p {
  margin: 6px 0 0;
  color: #8e8e93;
  font-size: 13px;
  font-weight: 800;
  line-height: 1.4;
}

.quiz-mode-tabs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px;
  padding: 4px;
  border: 1px solid #dfe4ec;
  border-radius: 8px;
  background: #e9edf2;
}

.quiz-mode-tabs button {
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0 12px;
  border: 0;
  border-radius: 6px;
  color: #7b8492;
  background: transparent;
  font-size: 13px;
  font-weight: 900;
  white-space: nowrap;
  transition: color 0.18s ease, background-color 0.18s ease, box-shadow 0.18s ease;
}

.quiz-mode-tabs button.active {
  color: #111827;
  background: #ffffff;
  box-shadow: 0 1px 5px rgba(15, 23, 42, 0.1);
}

.quiz-mode-tabs .material-symbols-outlined,
.quiz-primary-button .material-symbols-outlined,
.quiz-secondary-button .material-symbols-outlined {
  font-size: 18px;
}

.quiz-create-view,
.quiz-list-view,
.quiz-solve-view {
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 2px;
}

.quiz-solve-view {
  position: relative;
  justify-content: flex-start;
  padding: 33px 0 24px;
}

.quiz-simple-builder {
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

.quiz-simple-builder.is-generating {
  min-height: 320px;
  align-items: center;
  justify-content: center;
}

.quiz-generating-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  text-align: center;
}

.quiz-generating-animation {
  margin-bottom: 2px;
}

.quiz-generating-state strong {
  color: #111827;
  font-size: 17px;
  font-weight: 900;
  line-height: 1.25;
}

.quiz-simple-builder-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.quiz-simple-title {
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.quiz-simple-title h2 {
  margin: 0;
  color: #111111;
  font-size: 22px;
  font-weight: 900;
  line-height: 1.2;
}

.quiz-selected-source-box {
  display: grid;
  gap: 8px;
  padding: 14px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f8fafc;
}

.quiz-selected-source-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #475569;
  font-size: 12px;
  font-weight: 900;
}

.quiz-selected-source-top strong {
  color: #111827;
  font-size: 13px;
  font-weight: 900;
}

.quiz-source-header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
}

.quiz-source-picker-button,
.quiz-source-clear-button {
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 0;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 900;
  line-height: 1;
}

.quiz-source-picker-button {
  padding: 0 12px;
  color: #64748b;
  background: #eef2f7;
}

.quiz-source-picker-button .material-symbols-outlined {
  font-size: 16px;
}

.quiz-source-clear-button {
  padding: 0 10px;
  color: #64748b;
  background: #eef2f7;
}

.quiz-source-picker-button:hover,
.quiz-source-clear-button:hover {
  background: #e2e8f0;
}

.quiz-selected-source-empty {
  margin: 0;
  color: #8e8e93;
  font-size: 12px;
  font-weight: 800;
}

.quiz-selected-source-list {
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.quiz-selected-source-chip,
.quiz-selected-source-more {
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: 145px;
  height: 28px;
  padding: 0 9px;
  border: 1px solid #dbeafe;
  border-radius: 999px;
  background: #eff6ff;
  color: #1f3f68;
  font-size: 11px;
  font-weight: 900;
}

.quiz-selected-source-chip span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-selected-source-chip .material-symbols-outlined {
  flex: 0 0 auto;
  color: #2563eb;
  font-size: 14px;
}

.quiz-selected-source-more {
  border-color: #e2e8f0;
  background: #ffffff;
  color: #64748b;
}

.quiz-source-picker-layer {
  position: fixed;
  inset: 0;
  z-index: 80;
  background: transparent;
}

.quiz-source-picker-popover {
  position: fixed;
  width: min(360px, calc(100vw - 48px));
  max-height: min(520px, calc(100vh - 72px));
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid #dfe5ee;
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14);
}

.quiz-source-picker-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  color: #475569;
  font-size: 12px;
  font-weight: 900;
}

.quiz-source-picker-head button {
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

.quiz-source-picker-head .material-symbols-outlined {
  font-size: 17px;
}

.quiz-source-picker-actions {
  display: flex;
  justify-content: flex-end;
}

.quiz-source-picker-actions button {
  height: 30px;
  padding: 0 10px;
  border: 0;
  border-radius: 8px;
  color: #2563eb;
  background: #eff6ff;
  font-size: 11px;
  font-weight: 900;
}

.quiz-source-picker-actions button:hover {
  background: #dbeafe;
}

.quiz-source-picker-list {
  max-height: 390px;
  display: grid;
  gap: 6px;
  overflow-y: auto;
}

.quiz-source-picker-item {
  min-width: 0;
  height: 48px;
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) 22px;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  border: 0;
  border-radius: 8px;
  color: #1f2937;
  background: #f8fafc;
  text-align: left;
}

.quiz-source-picker-item.selected {
  box-shadow: inset 0 0 0 2px #2563eb;
  background: #eef4ff;
}

.quiz-source-picker-item.disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.quiz-source-picker-icon {
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

.quiz-source-picker-item.is-recording .quiz-source-picker-icon {
  color: #f59e0b;
  background: rgba(255, 242, 207, 0.9);
}

.quiz-source-picker-copy {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.quiz-source-picker-copy strong,
.quiz-source-picker-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-source-picker-copy strong {
  color: #111827;
  font-size: 12px;
  font-weight: 900;
}

.quiz-source-picker-copy small {
  color: #7b8492;
  font-size: 11px;
  font-weight: 800;
}

.quiz-source-picker-check {
  color: #2563eb;
  font-size: 19px;
}

.quiz-source-picker-empty {
  padding: 18px 10px;
  border-radius: 8px;
  color: #8e8e93;
  background: #f8fafc;
  font-size: 12px;
  font-weight: 800;
  text-align: center;
}

.quiz-simple-form {
  display: grid;
  gap: 14px;
}

.quiz-simple-field {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 180px;
  align-items: center;
  gap: 20px;
  color: #111111;
  font-size: 13px;
  font-weight: 600;
}

.quiz-simple-select-wrap {
  position: relative;
  min-width: 0;
  display: block;
}

.quiz-simple-select-wrap select {
  width: 100%;
  height: 42px;
  appearance: none;
  -webkit-appearance: none;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 0 38px 0 16px;
  color: #111111;
  background: #ffffff;
  font-size: 14px;
  font-weight: 700;
  outline: none;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.quiz-simple-select-wrap select:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

.quiz-simple-select-wrap .material-symbols-outlined {
  position: absolute;
  top: 50%;
  right: 14px;
  transform: translateY(-50%);
  color: #6b7280;
  font-size: 20px;
  pointer-events: none;
}

.quiz-simple-actions {
  display: flex;
  justify-content: flex-end;
  padding-top: 16px;
}

.quiz-simple-primary {
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

.quiz-simple-primary:hover {
  background: #252525;
}

.quiz-simple-primary:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.quiz-simple-primary:active {
  transform: scale(0.98);
}

.quiz-source-card,
.quiz-builder,
.quiz-session-card,
.quiz-question-card {
  border: 1px solid #dfe5ee;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 18px 36px rgba(15, 23, 42, 0.07);
}

.quiz-source-card {
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 18px;
  background:
    linear-gradient(135deg, rgba(248, 250, 252, 0.95), rgba(255, 255, 255, 0.98) 56%),
    #ffffff;
}

.quiz-source-card.empty {
  border-style: dashed;
  background: #f8fafc;
}

.quiz-source-card.blocked {
  border-color: #fed7aa;
  background: #fff7ed;
}

.quiz-source-icon {
  flex: 0 0 auto;
  width: 48px;
  height: 48px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #111827;
  background: #e8eef7;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.quiz-source-icon .material-symbols-outlined {
  font-size: 25px;
}

.quiz-source-main {
  flex: 1;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.quiz-source-label {
  color: #475569;
  font-size: 11.5px;
  font-weight: 900;
}

.quiz-source-card strong {
  display: block;
  overflow: hidden;
  margin-top: 4px;
  color: #1d1d1f;
  font-size: 17px;
  font-weight: 900;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-source-card p {
  margin: 6px 0 0;
  color: #8e8e93;
  font-size: 12px;
  font-weight: 800;
}

.quiz-source-stats {
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  overflow: hidden;
  border: 1px solid #dce3ec;
  border-radius: 8px;
  background: rgba(248, 250, 252, 0.86);
}

.quiz-source-stats span {
  min-width: 58px;
  display: grid;
  gap: 2px;
  padding: 9px 11px;
  color: #64748b;
  font-size: 10.5px;
  font-weight: 900;
  text-align: center;
}

.quiz-source-stats span + span {
  border-left: 1px solid #e2e8f0;
}

.quiz-source-stats strong {
  margin: 0;
  color: #111827;
  font-size: 15px;
  line-height: 1;
}

.quiz-source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.quiz-source-chip,
.quiz-source-more {
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 210px;
  padding: 6px 9px;
  border: 1px solid #dbeafe;
  border-radius: 999px;
  background: #eff6ff;
  color: #1f3f68;
  font-size: 11px;
  font-weight: 900;
  line-height: 1.1;
}

.quiz-source-chip {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-source-chip .material-symbols-outlined {
  flex: 0 0 auto;
  color: #2563eb;
  font-size: 14px;
}

.quiz-source-more {
  background: #f1f5f9;
  border-color: #e2e8f0;
  color: #64748b;
}

.quiz-builder {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 20px;
}

.quiz-builder-top,
.quiz-list-header,
.quiz-solve-top,
.quiz-generate-row,
.quiz-submit-row,
.quiz-result-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.quiz-list-header {
  justify-content: flex-end;
}

.quiz-builder-top h2,
.quiz-list-header h2,
.quiz-solve-top h2 {
  margin: 0;
  color: #1d1d1f;
  font-size: 19px;
  font-weight: 900;
  line-height: 1.2;
}

.quiz-count-presets {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px;
  border: 1px solid #dce3ec;
  border-radius: 8px;
  background: #f3f6fa;
}

.quiz-count-presets button {
  width: 42px;
  height: 34px;
  border: 0;
  border-radius: 6px;
  color: #64748b;
  background: transparent;
  font-size: 13px;
  font-weight: 900;
  transition: color 0.18s ease, background-color 0.18s ease, box-shadow 0.18s ease;
}

.quiz-count-presets button.active {
  color: #1d1d1f;
  background: #ffffff;
  box-shadow: 0 8px 16px rgba(15, 23, 42, 0.1);
}

.quiz-builder-meter {
  display: grid;
  gap: 7px;
  padding: 2px 0 0;
}

.quiz-builder-meter-track {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: #edf1f6;
}

.quiz-builder-meter-track span {
  display: block;
  width: var(--quiz-progress);
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #2563eb, #0f766e 56%, #f59e0b);
  transition: width 0.22s ease;
}

.quiz-builder-meter-labels {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 10px;
  color: #8e8e93;
  font-size: 10.5px;
  font-weight: 900;
}

.quiz-builder-meter-labels span:last-child {
  text-align: right;
}

.quiz-builder-meter-labels strong {
  color: #111827;
  font-size: 12px;
}

.quiz-type-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.quiz-type-card {
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
  padding: 14px;
  border: 1px solid #dfe5ee;
  border-radius: 8px;
  background: #ffffff;
  transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}

.quiz-type-card::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 4px;
  background: var(--type-accent, #2563eb);
}

.quiz-type-card:hover {
  transform: translateY(-1px);
  border-color: #cbd5e1;
  box-shadow: 0 14px 26px rgba(15, 23, 42, 0.09);
}

.quiz-type-card.tone-blue {
  --type-accent: #2563eb;
  --type-bg: #eff6ff;
  --type-text: #1d4ed8;
}

.quiz-type-card.tone-amber {
  --type-accent: #f59e0b;
  --type-bg: #fffbeb;
  --type-text: #b45309;
}

.quiz-type-card.tone-green {
  --type-accent: #0f766e;
  --type-bg: #ecfdf5;
  --type-text: #047857;
}

.quiz-type-main {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.quiz-type-main > .material-symbols-outlined {
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: var(--type-text, #2563eb);
  background: var(--type-bg, #eff6ff);
  font-size: 20px;
}

.quiz-type-main strong {
  display: block;
  color: #1d1d1f;
  font-size: 14px;
  font-weight: 900;
}

.quiz-type-main p {
  margin: 3px 0 0;
  color: #8e8e93;
  font-size: 11px;
  font-weight: 800;
  line-height: 1.35;
}

.quiz-type-count-badge {
  margin-left: auto;
  min-width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #dfe5ee;
  border-radius: 999px;
  color: var(--type-text, #2563eb);
  background: var(--type-bg, #eff6ff);
  font-size: 12px;
  font-weight: 950;
}

.quiz-stepper {
  display: grid;
  grid-template-columns: 34px minmax(34px, 1fr) 34px;
  align-items: center;
  gap: 7px;
}

.quiz-stepper button {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #dce3ec;
  border-radius: 8px;
  color: #1d1d1f;
  background: #f8fafc;
  transition: background-color 0.18s ease, border-color 0.18s ease;
}

.quiz-stepper button:not(:disabled):hover {
  border-color: #cbd5e1;
  background: #ffffff;
}

.quiz-stepper button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
}

.quiz-stepper .material-symbols-outlined {
  font-size: 17px;
}

.quiz-stepper > span {
  color: #1d1d1f;
  font-size: 20px;
  font-weight: 900;
  text-align: center;
}

.quiz-generate-row {
  padding-top: 2px;
}

.quiz-generate-row > span,
.quiz-submit-row > span {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 0 11px;
  border-radius: 999px;
  color: #475569;
  background: #f3f6fa;
  font-size: 12px;
  font-weight: 900;
}

.quiz-primary-button,
.quiz-secondary-button {
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  border: 0;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 900;
  line-height: 1;
  white-space: nowrap;
}

.quiz-primary-button {
  padding: 0 16px;
  color: #ffffff;
  background: #1d1d1f;
  box-shadow: none;
}

.quiz-secondary-button {
  padding: 0 13px;
  color: #64748b;
  background: #eef2f7;
}

.quiz-secondary-button:hover:not(:disabled) {
  background: #e2e8f0;
}

.quiz-primary-button:disabled,
.quiz-secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.quiz-session-list {
  min-height: 0;
  flex: 1;
  display: grid;
  align-content: start;
  gap: 12px;
  overflow-y: auto;
  padding: 2px 2px 12px;
}

.quiz-session-card {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  align-items: center;
  gap: 14px;
  padding: 14px;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.quiz-session-card:hover {
  transform: translateY(-1px);
  border-color: #cbd5e1;
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
}

.quiz-session-card.active {
  border-color: #111827;
  box-shadow: inset 0 0 0 1px #111827, 0 16px 30px rgba(15, 23, 42, 0.1);
}

.quiz-session-card > .material-symbols-outlined {
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #475569;
  background: #f1f5f9;
  font-size: 22px;
}

.quiz-session-main {
  min-width: 0;
}

.quiz-session-main strong {
  display: block;
  overflow: hidden;
  color: #1d1d1f;
  font-size: 14px;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-session-main p {
  margin: 4px 0 0;
  overflow: hidden;
  display: flex;
  align-items: center;
  gap: 7px;
  color: #8e8e93;
  font-size: 12px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-session-type {
  flex: 0 0 auto;
  color: #475569;
  font-weight: 900;
}

.quiz-session-actions {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.quiz-delete-button {
  width: 42px;
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #fee2e2;
  border-radius: 8px;
  color: #ef4444;
  background: #fff7f7;
  transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease;
}

.quiz-delete-button:hover:not(:disabled) {
  border-color: #fecaca;
  color: #dc2626;
  background: #fff1f2;
}

.quiz-delete-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.quiz-delete-button .material-symbols-outlined {
  font-size: 20px;
}

.quiz-error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid #fecdd3;
  border-radius: 8px;
  color: #be123c;
  background: #fff1f2;
  font-size: 12px;
  font-weight: 900;
}

.quiz-error .material-symbols-outlined {
  font-size: 18px;
}

.quiz-empty {
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  border: 1px dashed #d7deea;
  border-radius: 8px;
  background: #fafcff;
  color: #8e8e93;
  font-size: 14px;
  font-weight: 800;
  text-align: center;
}

.quiz-empty.compact {
  min-height: 220px;
}

.quiz-empty .material-symbols-outlined {
  color: #c7c7cc;
  font-size: 42px;
}

.quiz-result-bar {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 2;
  padding: 14px 16px;
  border: 1px solid #a7f3d0;
  border-radius: 8px;
  background: linear-gradient(135deg, #ecfdf5, #ffffff);
  box-shadow: 0 14px 28px rgba(21, 128, 61, 0.08);
}

.quiz-result-bar div {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.quiz-result-bar strong {
  color: #166534;
  font-size: 18px;
  font-weight: 900;
}

.quiz-result-bar span {
  color: #15803d;
  font-size: 12px;
  font-weight: 900;
}

.quiz-play-footer {
  display: flex;
  align-items: center;
  gap: 10px;
}

.quiz-play-footer {
  justify-content: space-between;
}

.quiz-play-card {
  position: relative;
  z-index: 1;
  min-height: 280px;
  height: auto;
  max-height: none;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 30px;
  overflow: visible;
  padding: 4px 0 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}

.quiz-play-card.graded.correct {
  border-color: transparent;
}

.quiz-play-card.graded.wrong {
  border-color: transparent;
}

.quiz-play-question {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: flex-start;
  gap: 12px;
}

.quiz-play-qbadge {
  min-width: 40px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #1d1d1f;
  background: #f3f4f6;
  font-size: 13px;
  font-weight: 950;
}

.quiz-play-question h2 {
  margin: 0;
  color: #1d1d1f;
  font-size: 15px;
  font-weight: 950;
  line-height: 1.45;
  word-break: keep-all;
  overflow-wrap: anywhere;
}

.quiz-play-options {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}

.quiz-play-card.type-OX .quiz-play-options {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.quiz-play-option {
  min-height: 44px;
  padding: 10px 13px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  color: #1f2937;
  background: #ffffff;
  font-size: 12.5px;
  font-weight: 900;
  line-height: 1.35;
  text-align: left;
  transition: border-color 0.18s ease, background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.quiz-play-option:not(:disabled):hover {
  transform: translateY(-1px);
  border-color: #cbd5e1;
  background: #fbfdff;
}

.quiz-play-option.selected {
  border-color: #2563eb;
  color: #1d4ed8;
  background: #eff6ff;
}

.quiz-play-option.correct {
  border-color: #22c55e;
  color: #14532d;
  background: #dcfce7;
}

.quiz-play-option.wrong {
  border-color: #fca5a5;
  color: #7f1d1d;
  background: #fee2e2;
}

.quiz-play-option:disabled {
  cursor: default;
}

.quiz-play-short-answer input {
  width: 100%;
  height: 48px;
  padding: 0 14px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  color: #1f2937;
  background: #ffffff;
  font-size: 13px;
  font-weight: 850;
  line-height: 48px;
  outline: none;
}

.quiz-play-short-answer input:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

.quiz-play-feedback {
  display: grid;
  gap: 7px;
  padding-top: 12px;
  border-top: 1px solid #e5e7eb;
  color: #1d1d1f;
}

.quiz-play-feedback p {
  margin: 0;
  color: #1f2937;
  font-size: 12.5px;
  font-weight: 750;
  line-height: 1.5;
}

.quiz-play-grade {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  width: fit-content;
  font-size: 15px;
  font-weight: 950;
}

.quiz-play-grade .material-symbols-outlined {
  font-size: 18px;
  font-variation-settings: 'FILL' 1;
}

.quiz-play-grade.correct {
  color: #15803d;
}

.quiz-play-grade.wrong {
  color: #ef4444;
}

.quiz-play-footer {
  position: sticky;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 3;
  margin-top: auto;
  padding: 14px 0 2px;
  background: linear-gradient(180deg, rgba(250, 252, 255, 0), rgba(250, 252, 255, 0.98) 34%, rgba(250, 252, 255, 1));
}

.quiz-play-footer > span {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 0 10px;
  border-radius: 999px;
  color: #475569;
  background: #f3f6fa;
  font-size: 12px;
  font-weight: 950;
}

.quiz-question-list {
  min-height: 0;
  flex: 1;
  display: grid;
  grid-auto-rows: max-content;
  align-content: start;
  gap: 16px;
  overflow-y: auto;
  padding: 2px 3px 96px;
}

.quiz-question-card {
  position: relative;
  overflow: visible;
  display: flex;
  flex-direction: column;
  align-self: start;
  gap: 16px;
  min-height: 0;
  padding: 20px 22px;
}

.quiz-question-card::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 -1px;
  width: 4px;
  border-radius: 8px 0 0 8px;
  background: #cbd5e1;
}

.quiz-question-card.graded::before {
  background: #0f766e;
}

.quiz-question-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 32px;
}

.quiz-type-chip {
  display: inline-flex;
  align-items: center;
  height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  color: #1f2937;
  background: #edf2f7;
  font-size: 11px;
  font-weight: 900;
}

.quiz-question-index {
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #1d1d1f;
  border: 1px solid #dfe5ee;
  background: #ffffff;
  font-size: 12px;
  font-weight: 900;
}

.quiz-question-card h2 {
  margin: 0;
  color: #1f2937;
  font-size: 16px;
  font-weight: 900;
  line-height: 1.5;
  word-break: keep-all;
  overflow-wrap: anywhere;
}

.quiz-options {
  display: grid;
  gap: 9px;
  min-height: 0;
}

.quiz-option {
  min-height: 46px;
  padding: 11px 13px;
  border: 1px solid #dfe5ee;
  border-radius: 8px;
  color: #334155;
  background: #fbfcfe;
  font-size: 13px;
  font-weight: 800;
  line-height: 1.45;
  text-align: left;
  transition: border-color 0.18s ease, background-color 0.18s ease, color 0.18s ease;
}

.quiz-option:not(:disabled):hover {
  border-color: #cbd5e1;
  background: #ffffff;
}

.quiz-option.selected {
  border-color: #2563eb;
  color: #1d4ed8;
  background: #eff6ff;
}

.quiz-option:disabled {
  cursor: default;
}

.quiz-short-answer input {
  width: 100%;
  min-height: 46px;
  padding: 11px 13px;
  border: 1px solid #dfe5ee;
  border-radius: 8px;
  color: #1f2937;
  background: #ffffff;
  font-size: 14px;
  font-weight: 800;
  outline: none;
}

.quiz-question-card.type-MULTIPLE_CHOICE .quiz-options {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.quiz-question-card.type-OX .quiz-options {
  grid-template-columns: repeat(2, minmax(0, 180px));
}

.quiz-question-card.type-SHORT_ANSWER {
  gap: 14px;
}

.quiz-short-answer input:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

.quiz-explanation {
  display: grid;
  gap: 6px;
  padding: 12px 13px;
  border: 1px solid #dfe5ee;
  border-radius: 8px;
  background: #f8fafc;
  color: #475569;
  font-size: 12px;
  font-weight: 800;
  line-height: 1.55;
}

.quiz-explanation p {
  margin: 0;
}

.quiz-grade {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  width: fit-content;
  font-size: 12px;
  font-weight: 900;
}

.quiz-grade .material-symbols-outlined {
  font-size: 17px;
}

.quiz-grade.correct {
  color: #15803d;
}

.quiz-grade.wrong {
  color: #be123c;
}

@media (max-width: 920px) {
  .quiz-source-main {
    align-items: stretch;
    flex-direction: column;
  }

  .quiz-source-stats {
    width: 100%;
  }

  .quiz-builder-top,
  .quiz-list-header,
  .quiz-solve-top,
  .quiz-generate-row,
  .quiz-submit-row,
  .quiz-result-bar,
  .quiz-play-footer {
    align-items: stretch;
    flex-direction: column;
  }

  .quiz-play-card {
    padding: 16px;
  }

  .quiz-play-question {
    grid-template-columns: 1fr;
  }

  .quiz-play-card.type-OX .quiz-play-options {
    grid-template-columns: 1fr;
  }

  .quiz-type-grid {
    grid-template-columns: 1fr;
  }

  .quiz-mode-tabs {
    width: 100%;
  }

  .quiz-mode-tabs button {
    flex: 1;
  }

  .quiz-session-card {
    grid-template-columns: 30px minmax(0, 1fr);
  }

  .quiz-session-actions {
    grid-column: 1 / -1;
    width: 100%;
  }

  .quiz-session-actions .quiz-secondary-button {
    flex: 1;
  }
}
</style>
