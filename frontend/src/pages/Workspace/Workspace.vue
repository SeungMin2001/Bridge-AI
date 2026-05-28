<!-- 음성 녹음, 실시간 전사, AI 분석 및 교차 참조가 이루어지는 작업실 페이지 컴포넌트입니다. -->
<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import LeftSidebar from '../../components/workspace/LeftSidebar.vue'
import MainContent from '../../components/workspace/MainContent.vue'
import RightSidebar from '../../components/workspace/RightSidebar.vue'
import CitationPopover from '../../components/workspace/citations/CitationPopover.vue'
import { useChat } from '../../composables/useChat'
import { deleteWorkspaceRecordingData, getWorkspaceSession, isWorkspaceUuid, saveSessionResources } from '../../api/workspaceApi.js'

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  isRecording: { type: Boolean, default: false },
  isRecordingPaused: { type: Boolean, default: false },
  recordingMode: { type: String, default: 'lecture' },
  recordingTimeText: { type: String, default: '00:00:00' },
  recordingAudioLevel: { type: Number, default: 0 },
  diarizationEnabled: { type: Boolean, default: false },
  diarizationStatus: { type: String, default: 'idle' },
  activeFileName: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  activeFileType: { type: String, default: 'lecture' },
  currentAttachments: { type: Array, default: () => [] },
  currentRecordings: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  isRightSidebarVisible: { type: Boolean, default: true },
  scheduleExtractionNotice: { type: Object, default: null },
  summaryState: { type: Object, default: () => ({}) },
  summaryNotes: { type: Array, default: () => [] },
  aiInput: { type: String, default: '' }
})

const emit = defineEmits([
  'navigateHome',
  'navigate',
  'fileSelect',
  'update:fileTree',
  'update:favorites',
  'update:aiInput',
  'dismissScheduleNotice',
  'confirmAndSyncSchedule',
  'ignoreSchedule',
  'startRecording',
  'pauseRecording',
  'resumeRecording',
  'stopRecording',
  'generateMaterialSummary',
  'generateRecordingSummary',
  'deleteSummary',
  'rightSidebarToggle',
  'addToNote',
  'askAi',
  'uploadLectureMaterials',
  'uploadRecordingFile',
  'closePreviewMaterial',
  'openStoredMaterial',
  'openRecording'
])

const isLeftSidebarCollapsed = ref(false)
const { showCitePopover, currentCite, citePopoverPos, closeCitePopover, clearHistory } = useChat()
const citationSourceRequest = ref(null)
const recordingSourceRequest = ref(null)
const materialEvidenceRequest = ref(null)
const mainContentTabRequest = ref(null)
const isMiniSourceOpen = ref(false)
const workspaceActiveMainTab = ref('materials')
const workspaceUnifiedCardRef = ref(null)
const sourceUploadFileInput = ref(null)
const miniSourceMenu = ref({ visible: false, x: 0, y: 0, source: null })
const scriptPaneWidth = ref(50)
const isScriptPaneResizing = ref(false)
const selectedMiniSourceIds = ref(new Set())
const isSourceUploadDialogOpen = ref(false)
const isSourceUploadDragging = ref(false)
const pendingSourceUploadRecording = ref(null)
const pendingStoppedRecordingPlayer = ref(null)
const hydratingFolderFileIds = new Set()

const DEFAULT_SCRIPT_PANE_PERCENT = 50
const MAX_SCRIPT_PANE_PERCENT = 72
const MIN_SCRIPT_PANE_WIDTH = 280
const MIN_MAIN_PANE_WIDTH = 340
const RESIZE_KEY_STEP = 2
const SOURCE_UPLOAD_ACCEPT = [
  '.pdf',
  '.ppt',
  '.pptx',
  'application/pdf',
  'application/vnd.ms-powerpoint',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  'audio/*',
  '.aac',
  '.flac',
  '.m4a',
  '.mp3',
  '.ogg',
  '.opus',
  '.wav',
  '.webm'
].join(',')

function toggleMiniSourcePanel() {
  isMiniSourceOpen.value = !isMiniSourceOpen.value
}

function handleMiniCardClick(event) {
  if (isMiniSourceOpen.value) return

  const target = event.target
  if (target instanceof Element && target.closest('button, input, .mini-source-context-menu')) {
    return
  }

  isMiniSourceOpen.value = true
}

function openSourceUploadDialog() {
  if (!props.activeFileId) return
  isSourceUploadDialogOpen.value = true
}

function closeSourceUploadDialog() {
  isSourceUploadDialogOpen.value = false
  isSourceUploadDragging.value = false
}

function browseSourceUploadFiles() {
  sourceUploadFileInput.value?.click()
}

function isMaterialUploadFile(file) {
  return (
    /(\.pdf|\.ppt|\.pptx)$/i.test(file?.name || '') ||
    [
      'application/pdf',
      'application/vnd.ms-powerpoint',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    ].includes(file?.type || '')
  )
}

function isRecordingUploadFile(file) {
  return !!file && (
    file.type?.startsWith('audio/') ||
    /\.(aac|flac|m4a|mp3|ogg|opus|wav|webm)$/i.test(file.name || '')
  )
}

function getSourceUploadFileStem(filename = '') {
  return String(filename || '')
    .replace(/\.[^.]+$/, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeSourceUploadText(value = '') {
  return String(value || '').replace(/\s+/g, ' ').trim().toLowerCase()
}

function matchesPendingUploadedRecording(source = {}, pending = {}) {
  if (source.type !== 'recording') return false

  const recording = source.recording || {}
  const pendingTitle = normalizeSourceUploadText(pending.title)
  const pendingFileName = normalizeSourceUploadText(pending.fileName)
  const candidates = [
    source.title,
    recording.title,
    recording.name,
    recording.originalName,
    recording.original_name,
    recording.fileName,
    recording.file_name
  ].map(normalizeSourceUploadText)

  return candidates.some((candidate) => (
    candidate &&
    (candidate === pendingTitle || candidate === pendingFileName)
  ))
}

function uploadSourceFiles(files = []) {
  const sourceFiles = Array.from(files || []).filter(Boolean)
  if (!sourceFiles.length) return

  const sourceFile = sourceFiles.find((file) => isMaterialUploadFile(file) || isRecordingUploadFile(file))
  if (!sourceFile) {
    alert('PDF, PPT 또는 음성 파일만 추가할 수 있습니다.')
    return
  }

  if (isMaterialUploadFile(sourceFile)) {
    emit('uploadLectureMaterials', [sourceFile])
  } else {
    pendingSourceUploadRecording.value = {
      fileName: sourceFile.name || '',
      title: getSourceUploadFileStem(sourceFile.name),
      requestedAt: Date.now()
    }
    emit('uploadRecordingFile', [sourceFile])
  }

  closeSourceUploadDialog()
}

function handleSourceUploadFileChange(event) {
  uploadSourceFiles(event.target.files)
  event.target.value = ''
}

function handleSourceUploadDrop(event) {
  isSourceUploadDragging.value = false
  uploadSourceFiles(event.dataTransfer?.files)
}

function getScriptPaneMinPercent() {
  const cardWidth = workspaceUnifiedCardRef.value?.getBoundingClientRect().width || 0
  if (!cardWidth) return 28
  return Math.min(DEFAULT_SCRIPT_PANE_PERCENT, Math.max(24, (MIN_SCRIPT_PANE_WIDTH / cardWidth) * 100))
}

function getScriptPaneMaxPercent() {
  const cardWidth = workspaceUnifiedCardRef.value?.getBoundingClientRect().width || 0
  if (!cardWidth) return MAX_SCRIPT_PANE_PERCENT
  const maxByMainPane = 100 - ((MIN_MAIN_PANE_WIDTH / cardWidth) * 100)
  return Math.max(getScriptPaneMinPercent(), Math.min(MAX_SCRIPT_PANE_PERCENT, maxByMainPane))
}

function setScriptPaneWidth(nextPercent) {
  const minPercent = getScriptPaneMinPercent()
  const maxPercent = getScriptPaneMaxPercent()
  const clamped = Math.min(maxPercent, Math.max(minPercent, nextPercent))
  scriptPaneWidth.value = Number(clamped.toFixed(2))
}

function setScriptPaneWidthFromPointer(clientX) {
  const rect = workspaceUnifiedCardRef.value?.getBoundingClientRect()
  if (!rect?.width) return
  setScriptPaneWidth(((clientX - rect.left) / rect.width) * 100)
}

function stopScriptPaneResize() {
  if (!isScriptPaneResizing.value) return
  isScriptPaneResizing.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.body.classList.remove('is-resizing')
  window.removeEventListener('pointermove', handleScriptPanePointerMove)
  window.removeEventListener('pointerup', stopScriptPaneResize)
  window.removeEventListener('pointercancel', stopScriptPaneResize)
}

function handleScriptPanePointerMove(event) {
  if (!isScriptPaneResizing.value) return
  setScriptPaneWidthFromPointer(event.clientX)
}

function handleScriptPanePointerDown(event) {
  if (event.button !== undefined && event.button !== 0) return
  event.preventDefault()
  event.currentTarget?.setPointerCapture?.(event.pointerId)
  isScriptPaneResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
  setScriptPaneWidthFromPointer(event.clientX)
  window.addEventListener('pointermove', handleScriptPanePointerMove)
  window.addEventListener('pointerup', stopScriptPaneResize)
  window.addEventListener('pointercancel', stopScriptPaneResize)
}

function handleScriptPaneResizeKeydown(event) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()

  if (event.key === 'Home') {
    setScriptPaneWidth(getScriptPaneMinPercent())
    return
  }

  if (event.key === 'End') {
    setScriptPaneWidth(DEFAULT_SCRIPT_PANE_PERCENT)
    return
  }

  const direction = event.key === 'ArrowLeft' ? -1 : 1
  setScriptPaneWidth(scriptPaneWidth.value + (direction * RESIZE_KEY_STEP))
}

function resetScriptPaneWidth() {
  setScriptPaneWidth(DEFAULT_SCRIPT_PANE_PERCENT)
}

function handleWorkspaceResize() {
  setScriptPaneWidth(scriptPaneWidth.value)
}

onMounted(() => {
  window.addEventListener('resize', handleWorkspaceResize)
})

onUnmounted(() => {
  stopScriptPaneResize()
  window.removeEventListener('resize', handleWorkspaceResize)
})

watch(() => props.activeFileId, () => {
  citationSourceRequest.value = null
  recordingSourceRequest.value = null
  materialEvidenceRequest.value = null
  pendingStoppedRecordingPlayer.value = null
  closeMiniSourceMenu()
  selectedMiniSourceIds.value = new Set()
  clearHistory()
  closeCitePopover()
  emit('update:aiInput', '')
})

const scheduleNoticeItems = computed(() => props.scheduleExtractionNotice?.items || [])
const visibleScheduleNoticeItems = computed(() => scheduleNoticeItems.value.slice(0, 3))
const hiddenScheduleNoticeCount = computed(() => Math.max(scheduleNoticeItems.value.length - 3, 0))
const scheduleNoticeTitle = computed(() => (
  scheduleNoticeItems.value.length > 1
    ? `새 일정 ${scheduleNoticeItems.value.length}개가 추가되었습니다`
    : '새 일정이 추가되었습니다'
))

function formatScheduleNoticeDate(value = '') {
  if (!value) return ''

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value

  const year = parsed.getFullYear()
  const month = parsed.getMonth() + 1
  const day = parsed.getDate()
  let hour = parsed.getHours()
  const minute = String(parsed.getMinutes()).padStart(2, '0')
  const meridiem = hour < 12 ? '오전' : '오후'
  hour %= 12
  if (hour === 0) hour = 12

  return `${year}년 ${month}월 ${day}일 ${meridiem} ${hour}:${minute}`
}

function closeScheduleNotice() {
  emit('dismissScheduleNotice')
}

function goSchedulePageFromNotice() {
  emit('dismissScheduleNotice')
  emit('navigate', 'schedule')
}

const confirmingScheduleIds = ref(new Set())

async function confirmNoticeItem(item) {
  confirmingScheduleIds.value.add(item.id)
  try {
    await emit('confirmAndSyncSchedule', item.id)
  } catch { /* handled upstream */ }
  confirmingScheduleIds.value.delete(item.id)
  // If all items are confirmed/ignored, close the notice
  const remaining = scheduleNoticeItems.value.filter(
    (i) => !confirmedNoticeIds.value.has(i.id) && !ignoredNoticeIds.value.has(i.id)
  )
  if (remaining.length === 0) closeScheduleNotice()
}

const confirmedNoticeIds = ref(new Set())
const ignoredNoticeIds = ref(new Set())

function markNoticeItemConfirmed(item) {
  confirmedNoticeIds.value.add(item.id)
  confirmNoticeItem(item)
}

function markNoticeItemIgnored(item) {
  ignoredNoticeIds.value.add(item.id)
  emit('ignoreSchedule', item.id) // 누락되었던 백엔드 상태 동기화 호출
  // Remove from visible list by tracking ignored ids
  const remaining = scheduleNoticeItems.value.filter(
    (i) => !confirmedNoticeIds.value.has(i.id) && !ignoredNoticeIds.value.has(i.id)
  )
  if (remaining.length === 0) closeScheduleNotice()
}

function confirmAllNoticeItems() {
  for (const item of scheduleNoticeItems.value) {
    if (!confirmedNoticeIds.value.has(item.id) && !ignoredNoticeIds.value.has(item.id)) {
      markNoticeItemConfirmed(item)
    }
  }
}

const activeNoticeItems = computed(() =>
  scheduleNoticeItems.value.filter(
    (i) => !confirmedNoticeIds.value.has(i.id) && !ignoredNoticeIds.value.has(i.id)
  )
)

const visibleActiveNoticeItems = computed(() => activeNoticeItems.value.slice(0, 3))
const hiddenActiveNoticeCount = computed(() => Math.max(activeNoticeItems.value.length - 3, 0))

function findNodeById(nodes = [], id = '') {
  for (const node of nodes) {
    if (node?.id === id) return node
    if (Array.isArray(node?.children)) {
      const found = findNodeById(node.children, id)
      if (found) return found
    }
  }
  return null
}

function findParentFolderByChildId(nodes = [], id = '', parent = null) {
  for (const node of nodes) {
    if (node?.id === id) return parent
    if (Array.isArray(node?.children)) {
      const found = findParentFolderByChildId(node.children, id, node)
      if (found) return found
    }
  }
  return null
}

function replaceNodeById(nodes = [], id = '', replacement = null) {
  return nodes.map((node) => {
    if (node?.id === id) return replacement || node
    if (Array.isArray(node?.children)) {
      return {
        ...node,
        children: replaceNodeById(node.children, id, replacement)
      }
    }
    return node
  })
}

function findMaterialInNode(node, cite = {}) {
  const materialId = String(cite?.material_id || '')
  const storedName = String(cite?.stored_name || '')
  const weeks = Array.isArray(node?.weeks) ? node.weeks : []

  for (const week of weeks) {
    const materials = Array.isArray(week?.materials) ? week.materials : []
    for (const material of materials) {
      if (materialId && String(material?.id || '') === materialId) return material
      if (storedName && String(material?.storedName || '') === storedName) return material
    }
  }
  return null
}

function openCitationSource(cite) {
  const sessionId = cite?.session_id
  if (!sessionId) return

  const node = findNodeById(props.fileTree, sessionId)
  if (!node) return

  emit('fileSelect', sessionId, node)
  isLeftSidebarCollapsed.value = false
  if (cite?.source_type === 'material') {
    const material = findMaterialInNode(node, cite)
    const materialId = material?.id || cite?.material_id
    if (materialId) {
      emit('openStoredMaterial', materialId)
    }
    closeCitePopover()
    return
  }

  citationSourceRequest.value = {
    id: `${sessionId}-${cite?.transcript_id || cite?.citation || Date.now()}`,
    cite,
    node
  }
  clearHistory()
  emit('update:aiInput', '')
  closeCitePopover()
}

async function openEvidenceSource(cite) {
  if (!cite) return

  if (cite.source_type !== 'material') {
    openCitationSource(cite)
    return
  }

  const sessionId = cite.session_id
  if (!sessionId) return

  const node = findNodeById(props.fileTree, sessionId)
  if (!node) return

  emit('fileSelect', sessionId, node)
  isLeftSidebarCollapsed.value = false

  const material = findMaterialInNode(node, cite)
  const materialId = material?.id || cite.material_id
  if (materialId) {
    emit('openStoredMaterial', materialId)
  }

  await nextTick()

  materialEvidenceRequest.value = {
    id: `${sessionId}-${materialId || cite.stored_name || cite.citation}-${cite.page || 0}-${Date.now()}`,
    cite,
    materialId,
    page: Number(cite.page || 1),
    text: cite.text || ''
  }
  closeCitePopover()
}

async function handleOpenRecording(payload) {
  emit('openRecording', payload)
  await nextTick()

  mainContentTabRequest.value = {
    id: `${payload?.sessionId || ''}-${payload?.recordingId || ''}-${Date.now()}`,
    tab: 'summary',
    summaryTab: 'summary'
  }
}

function collectTranscriptIds(recordings = []) {
  const ids = new Set()
  recordings.forEach((recording) => {
    const transcriptions = Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
    transcriptions.forEach((transcription) => {
      const segments = Array.isArray(transcription?.segments) ? transcription.segments : []
      segments.forEach((segment) => {
        const id = segment?.transcript_id || segment?.transcriptId
        if (id) ids.add(String(id))
      })
    })
  })
  return Array.from(ids)
}

const activeSourceNode = computed(() => (
  props.activeFileId ? findNodeById(props.fileTree, props.activeFileId) : null
))

const activeFolderFiles = computed(() => {
  if (!props.activeFileId) return []
  const parent = findParentFolderByChildId(props.fileTree, props.activeFileId)
  const siblings = Array.isArray(parent?.children)
    ? parent.children
    : (activeSourceNode.value ? [activeSourceNode.value] : [])
  return siblings.filter((node) => node?.type === 'file')
})

watch(
  activeFolderFiles,
  async (files) => {
    const targets = files.filter((file) => (
      file?.type === 'file' &&
      isWorkspaceUuid(file.id) &&
      file.resourcesLoaded === false &&
      !hydratingFolderFileIds.has(file.id)
    ))
    if (!targets.length) return

    const hydratedNodes = await Promise.all(targets.map(async (file) => {
      hydratingFolderFileIds.add(file.id)
      try {
        return {
          id: file.id,
          node: await getWorkspaceSession(file.id)
        }
      } catch (error) {
        console.warn('[workspace] folder file hydration failed:', file.id, error)
        return null
      } finally {
        hydratingFolderFileIds.delete(file.id)
      }
    }))

    const validNodes = hydratedNodes.filter(Boolean)
    if (!validNodes.length) return
    const nextTree = validNodes.reduce(
      (tree, item) => replaceNodeById(tree, item.id, item.node),
      props.fileTree
    )
    emit('update:fileTree', nextTree)
  },
  { immediate: true, deep: true }
)

const miniSourceWeeks = computed(() => {
  const weeks = Array.isArray(activeSourceNode.value?.weeks) ? activeSourceNode.value.weeks : []

  if (weeks.length) {
    return weeks.map((week, index) => ({
      id: week?.id || `week-${index + 1}`,
      label: week?.label || `${index + 1}주차`,
      materials: Array.isArray(week?.materials) ? week.materials : [],
      recordings: Array.isArray(week?.recordings) ? week.recordings : []
    }))
  }

  const materials = Array.isArray(props.currentAttachments) ? props.currentAttachments : []
  const recordings = Array.isArray(props.currentRecordings) ? props.currentRecordings : []
  if (!materials.length && !recordings.length) return []

  return [{
    id: 'current-week',
    label: '1주차',
    materials,
    recordings
  }]
})

const miniSourceTotalCount = computed(() => (
  miniSourceWeeks.value.reduce((total, week) => (
    total + week.materials.length + week.recordings.length
  ), 0)
))

function getMiniSourceId(item, prefix, index) {
  return item?.id || item?.recordingId || item?.storedName || item?.url || item?.name || `${prefix}-${index}`
}

function getMiniMaterialIcon(material = {}) {
  const name = material?.name || material?.storedName || ''
  if (/\.(ppt|pptx)$/i.test(name)) return 'slideshow'
  if (/\.pdf$/i.test(name)) return 'picture_as_pdf'
  return 'description'
}

function getMiniMaterialTitle(material = {}, index = 0) {
  return material?.name || material?.title || `강의자료 ${index + 1}`
}

function getMiniRecordingTitle(recording = {}, index = 0) {
  return recording?.title || recording?.name || `녹음본 ${index + 1}`
}

function getMiniRecordingIdentity(source = {}) {
  const recording = source.recording || source
  return String(
    source.recordingId ||
    source.id ||
    recording?.id ||
    recording?.recordingId ||
    recording?.storedName ||
    recording?.audioUrl ||
    recording?.title ||
    ''
  )
}

const miniSourceItems = computed(() => (
  miniSourceWeeks.value.flatMap((week) => [
    ...week.materials.map((material, index) => {
      const id = getMiniSourceId(material, 'material', index)
      return {
        uid: `material:${id}`,
        type: 'material',
        weekId: week.id,
        materialId: id,
        title: getMiniMaterialTitle(material, index),
        icon: getMiniMaterialIcon(material),
        material,
        transcriptIds: []
      }
    }),
    ...week.recordings.map((recording, index) => {
      const id = getMiniSourceId(recording, 'recording', index)
      return {
        uid: `recording:${id}`,
        id,
        type: 'recording',
        weekId: week.id,
        title: getMiniRecordingTitle(recording, index),
        icon: 'graphic_eq',
        recordingId: recording?.id || recording?.recordingId || '',
        recording,
        transcriptIds: collectTranscriptIds([recording])
      }
    })
  ])
))

const selectedMiniSourceItems = computed(() => (
  miniSourceItems.value.filter((source) => selectedMiniSourceIds.value.has(source.uid))
))

const areAllMiniSourcesSelected = computed(() => (
  miniSourceItems.value.length > 0 && selectedMiniSourceItems.value.length === miniSourceItems.value.length
))

watch(
  miniSourceItems,
  (sources) => {
    const validIds = new Set(sources.map((source) => source.uid))
    selectedMiniSourceIds.value = new Set(
      Array.from(selectedMiniSourceIds.value).filter((id) => validIds.has(id))
    )

    const recordingSources = sources.filter((source) => source.type === 'recording')
    const pending = pendingSourceUploadRecording.value
    if (pending) {
      if (Date.now() - pending.requestedAt > 30000) {
        pendingSourceUploadRecording.value = null
      } else {
        const targetSource = recordingSources.find((source) => matchesPendingUploadedRecording(source, pending))
          || recordingSources[recordingSources.length - 1]

        if (targetSource?.recording) {
          pendingSourceUploadRecording.value = null
          selectedMiniSourceIds.value = new Set([...selectedMiniSourceIds.value, targetSource.uid])
          openMiniRecording(targetSource.recording)
        }
      }
    }

    const pendingStoppedRecording = pendingStoppedRecordingPlayer.value
    if (!pendingStoppedRecording) return

    if (Date.now() - pendingStoppedRecording.requestedAt > 30000) {
      pendingStoppedRecordingPlayer.value = null
      return
    }

    const stoppedRecordingSource = recordingSources.find((source) => (
      !pendingStoppedRecording.existingIds.has(getMiniRecordingIdentity(source))
    ))
    if (!stoppedRecordingSource?.recording) return

    pendingStoppedRecordingPlayer.value = null
    openMiniRecording(stoppedRecordingSource.recording)
  },
  { immediate: true }
)

function isMiniSourceSelected(uid) {
  return selectedMiniSourceIds.value.has(uid)
}

function toggleMiniSource(uid) {
  const next = new Set(selectedMiniSourceIds.value)
  if (next.has(uid)) next.delete(uid)
  else next.add(uid)
  selectedMiniSourceIds.value = next
}

function toggleAllMiniSources() {
  selectedMiniSourceIds.value = areAllMiniSourcesSelected.value
    ? new Set()
    : new Set(miniSourceItems.value.map((source) => source.uid))
}

function openMiniMaterial(material = {}) {
  if (!material?.id) return
  emit('openStoredMaterial', material.id)
  mainContentTabRequest.value = {
    id: `mini-material-${material.id}-${Date.now()}`,
    tab: 'materials'
  }
}

function openMiniRecording(recording = {}) {
  const recordingId = recording?.id || recording?.recordingId || ''
  recordingSourceRequest.value = {
    id: `mini-recording-${recordingId || Date.now()}-${Date.now()}`,
    fileId: props.activeFileId,
    node: activeSourceNode.value,
    recordingId,
    recording
  }
}

function handleStopRecordingRequest() {
  pendingStoppedRecordingPlayer.value = {
    fileId: props.activeFileId,
    requestedAt: Date.now(),
    existingIds: new Set(
      miniSourceItems.value
        .filter((source) => source.type === 'recording')
        .map((source) => getMiniRecordingIdentity(source))
        .filter(Boolean)
    )
  }
  emit('stopRecording')
}

function closeMiniSourceMenu() {
  miniSourceMenu.value = { visible: false, x: 0, y: 0, source: null }
}

function openMiniSourceMenu(source, event) {
  event?.stopPropagation?.()
  const rect = event?.currentTarget?.getBoundingClientRect?.()
  const menuWidth = 158
  const menuHeight = 102
  const baseX = rect ? rect.right + 8 : event?.clientX || 0
  const baseY = rect ? rect.top : event?.clientY || 0
  miniSourceMenu.value = {
    visible: true,
    x: Math.min(baseX, window.innerWidth - menuWidth - 12),
    y: Math.min(baseY, window.innerHeight - menuHeight - 12),
    source
  }
}

function matchesMiniMaterial(item = {}, source = {}, index = 0) {
  return getMiniSourceId(item, 'material', index) === source.materialId
    || (source.material?.id && item?.id === source.material.id)
}

function matchesMiniRecording(item = {}, source = {}, index = 0) {
  const sourceId = source.recordingId || source.id
  return getMiniSourceId(item, 'recording', index) === source.id
    || (sourceId && (item?.id === sourceId || item?.recordingId === sourceId))
}

function syncMiniNodeFlatResources(node = {}) {
  if (!Array.isArray(node.weeks)) return
  node.attachments = node.weeks.flatMap((week) => (
    Array.isArray(week?.materials) ? week.materials : []
  ))
  node.recordings = node.weeks.flatMap((week) => (
    Array.isArray(week?.recordings) ? week.recordings : []
  ))
}

async function persistMiniSourceTree(nextTree, node) {
  emit('update:fileTree', nextTree)
  emit('fileSelect', props.activeFileId, node)

  if (!isWorkspaceUuid(props.activeFileId) || !Array.isArray(node?.weeks)) return
  await saveSessionResources(props.activeFileId, node.weeks)
}

async function handleMiniSourceAction(action) {
  const source = miniSourceMenu.value.source
  closeMiniSourceMenu()
  if (!source || !props.activeFileId) return

  if (action === 'rename') {
    const nextName = prompt(
      source.type === 'recording' ? '새 녹음본 이름을 입력하세요:' : '새 파일 이름을 입력하세요:',
      source.title
    )
    if (!nextName?.trim()) return

    const nextTree = JSON.parse(JSON.stringify(props.fileTree))
    const node = findNodeById(nextTree, props.activeFileId)
    const week = Array.isArray(node?.weeks)
      ? node.weeks.find((item) => (item?.id || '') === source.weekId)
      : null
    if (!node) return

    if (source.type === 'material') {
      if (Array.isArray(week?.materials)) {
        week.materials = week.materials.map((item, index) => (
          matchesMiniMaterial(item, source, index)
            ? { ...item, name: nextName.trim(), title: nextName.trim() }
            : item
        ))
      }
      node.attachments = Array.isArray(node.attachments)
        ? node.attachments.map((item, index) => (
          matchesMiniMaterial(item, source, index)
            ? { ...item, name: nextName.trim(), title: nextName.trim() }
            : item
        ))
        : node.attachments
    } else {
      if (Array.isArray(week?.recordings)) {
        week.recordings = week.recordings.map((item, index) => (
          matchesMiniRecording(item, source, index)
            ? { ...item, title: nextName.trim(), name: nextName.trim() }
            : item
        ))
      }
      node.recordings = Array.isArray(node.recordings)
        ? node.recordings.map((item, index) => (
          matchesMiniRecording(item, source, index)
            ? { ...item, title: nextName.trim(), name: nextName.trim() }
            : item
        ))
        : node.recordings
    }

    syncMiniNodeFlatResources(node)
    try {
      await persistMiniSourceTree(nextTree, node)
    } catch (error) {
      console.error('[workspace] mini source rename failed:', error)
      alert('이름 변경 저장에 실패했습니다.')
    }
    return
  }

  if (action !== 'delete') return
  if (!confirm(`"${source.title}"을(를) 삭제할까요?`)) return

  if (source.type === 'recording' && isWorkspaceUuid(props.activeFileId) && source.recordingId) {
    try {
      const result = await deleteWorkspaceRecordingData(props.activeFileId, source.recordingId)
      if (result?.node) {
        const nextTree = replaceNodeById(props.fileTree, props.activeFileId, result.node)
        selectedMiniSourceIds.value = new Set(
          Array.from(selectedMiniSourceIds.value).filter((uid) => uid !== source.uid)
        )
        emit('update:fileTree', nextTree)
        emit('fileSelect', props.activeFileId, result.node)
        return
      }
    } catch (error) {
      console.error('[workspace] mini recording delete failed:', error)
      alert('녹음본 삭제에 실패했습니다.')
      return
    }
  }

  const nextTree = JSON.parse(JSON.stringify(props.fileTree))
  const node = findNodeById(nextTree, props.activeFileId)
  const week = Array.isArray(node?.weeks)
    ? node.weeks.find((item) => (item?.id || '') === source.weekId)
    : null
  if (!node) return

  if (source.type === 'material') {
    if (Array.isArray(week?.materials)) {
      week.materials = week.materials.filter((item, index) => !matchesMiniMaterial(item, source, index))
    }
    node.attachments = Array.isArray(node.attachments)
      ? node.attachments.filter((item, index) => !matchesMiniMaterial(item, source, index))
      : node.attachments
  } else {
    if (Array.isArray(week?.recordings)) {
      week.recordings = week.recordings.filter((item, index) => !matchesMiniRecording(item, source, index))
    }
    node.recordings = Array.isArray(node.recordings)
      ? node.recordings.filter((item, index) => !matchesMiniRecording(item, source, index))
      : node.recordings
  }

  syncMiniNodeFlatResources(node)
  selectedMiniSourceIds.value = new Set(
    Array.from(selectedMiniSourceIds.value).filter((uid) => uid !== source.uid)
  )
  try {
    await persistMiniSourceTree(nextTree, node)
  } catch (error) {
    console.error('[workspace] mini source delete failed:', error)
    alert('삭제 저장에 실패했습니다.')
  }
}

const activeWorkspaceSource = computed(() => {
  if (!props.activeFileId) return null

  // 좌측 소스 사이드바에서 체크된 항목만 퀴즈 생성 범위로 넘긴다.
  const sources = selectedMiniSourceItems.value.map(({ uid, icon, ...source }) => ({
    id: source.material?.id || source.recordingId || uid,
    ...source
  }))
  const recordings = sources
    .filter((source) => source.type === 'recording' && source.recording)
    .map((source) => source.recording)
  const transcriptIds = Array.from(new Set(sources.flatMap((source) => source.transcriptIds || [])))
  const sourceCount = sources.length

  return {
    type: sourceCount ? 'workspace' : 'empty',
    title: props.activeFileName ? `${props.activeFileName} 선택 자료` : '현재 파일 선택 자료',
    sessionId: props.activeFileId,
    sourceCount,
    sources,
    recordings,
    transcriptIds
  }
})
</script>

<template>
  <div 
    class="workspace-page-shell p-[12px] flex relative h-full w-full text-[#1e293b] overflow-hidden transition-all duration-400"
    :class="[
      { 'gap-[8px]': !isLeftSidebarCollapsed || isRightSidebarVisible }
    ]"
  >
    <transition name="schedule-notice-fade">
      <section
        v-if="scheduleNoticeItems.length"
        class="workspace-schedule-notice"
        role="status"
        aria-live="polite"
      >
        <div class="workspace-schedule-notice-top">
          <div class="workspace-schedule-notice-icon">
            <span class="material-symbols-outlined">event_available</span>
          </div>
          <div class="workspace-schedule-notice-heading">
            <span>AI 일정 감지</span>
            <strong>{{ scheduleNoticeTitle }}</strong>
          </div>
          <button
            type="button"
            class="workspace-schedule-notice-close"
            aria-label="일정 알림 닫기"
            @click="closeScheduleNotice"
          >
            <span class="material-symbols-outlined">close</span>
          </button>
        </div>

        <div class="workspace-schedule-notice-list">
          <article
            v-for="item in visibleActiveNoticeItems"
            :key="item.id"
            class="workspace-schedule-notice-item"
          >
            <div class="workspace-schedule-notice-item-info">
              <strong>{{ item.title }}</strong>
              <span>{{ formatScheduleNoticeDate(item.dueDate) }}</span>
            </div>
            <div class="workspace-schedule-notice-item-actions">
              <button
                type="button"
                class="workspace-schedule-notice-item-btn confirm"
                :disabled="confirmingScheduleIds.has(item.id)"
                @click="markNoticeItemConfirmed(item)"
              >
                <span class="material-symbols-outlined">check</span>
                확정
              </button>
              <button
                type="button"
                class="workspace-schedule-notice-item-btn ignore"
                @click="markNoticeItemIgnored(item)"
              >
                <span class="material-symbols-outlined">close</span>
                무시
              </button>
            </div>
          </article>
          <div v-if="hiddenActiveNoticeCount" class="workspace-schedule-notice-more">
            외 {{ hiddenActiveNoticeCount }}개 일정
          </div>
        </div>

        <div class="workspace-schedule-notice-actions">
          <button type="button" class="workspace-schedule-notice-secondary" @click="closeScheduleNotice">
            닫기
          </button>
          <button type="button" class="workspace-schedule-notice-primary" @click="goSchedulePageFromNotice">
            일정관리로 이동
          </button>
        </div>
      </section>
    </transition>

    <aside
      :class="['workspace-mini-card', { 'is-open': isMiniSourceOpen }]"
      aria-label="워크스페이스 빠른 메뉴"
      @click="handleMiniCardClick"
    >
      <div class="workspace-mini-top">
        <button
          type="button"
          class="workspace-mini-toggle"
          aria-label="홈으로 이동"
          title="홈으로 이동"
          @click="emit('navigateHome')"
        >
          <img class="workspace-mini-toggle-logo" src="/images/logo.png" alt="" draggable="false" />
        </button>

        <div class="workspace-mini-actions">
          <div class="workspace-mini-add-wrap">
            <button
              type="button"
              class="workspace-mini-action workspace-mini-add-btn"
              aria-label="소스 추가"
              title="소스 추가"
              data-label="소스 추가"
              :disabled="!activeFileId"
              @click.stop="openSourceUploadDialog"
            >
              <span class="material-symbols-outlined">add</span>
              <span class="workspace-mini-add-label">소스 추가</span>
            </button>
          </div>
          <input
            ref="sourceUploadFileInput"
            class="hidden"
            type="file"
            :accept="SOURCE_UPLOAD_ACCEPT"
            @change="handleSourceUploadFileChange"
          />
        </div>
      </div>

      <div class="workspace-mini-source-panel">
        <div class="workspace-mini-source-heading">
          <small>{{ miniSourceTotalCount }}개</small>
        </div>

        <div v-if="activeFileId && miniSourceItems.length" class="workspace-mini-source-tree custom-scrollbar">
          <label class="mini-source-select-all">
            <span>모두 선택</span>
            <input
              type="checkbox"
              :checked="areAllMiniSourcesSelected"
              @change="toggleAllMiniSources"
            />
          </label>

          <div
            v-for="source in miniSourceItems"
            :key="source.uid"
            class="mini-source-check-row"
          >
            <button
              type="button"
              class="mini-source-icon-button"
              :class="source.type"
              :aria-label="`${source.title} 메뉴 열기`"
              :title="`${source.title} 메뉴`"
              @click="openMiniSourceMenu(source, $event)"
            >
              <span class="material-symbols-outlined mini-source-item-icon default-icon">{{ source.icon }}</span>
              <span class="material-symbols-outlined mini-source-item-icon hover-icon">more_vert</span>
            </button>
            <button
              type="button"
              class="mini-source-open-btn"
              :title="source.title"
              @click.prevent="source.type === 'material' ? openMiniMaterial(source.material) : openMiniRecording(source.recording)"
            >
              <span>{{ source.title }}</span>
            </button>
            <input
              type="checkbox"
              :checked="isMiniSourceSelected(source.uid)"
              @click.stop
              @change="toggleMiniSource(source.uid)"
            />
          </div>
        </div>

        <div v-else class="workspace-mini-source-empty">
          {{ activeFileId ? '저장된 소스가 없습니다.' : '파일을 선택하세요.' }}
        </div>
      </div>

      <div
        v-if="miniSourceMenu.visible"
        class="mini-source-menu-backdrop"
        @click="closeMiniSourceMenu"
      ></div>
      <div
        v-if="miniSourceMenu.visible"
        class="mini-source-context-menu"
        :style="{ left: `${miniSourceMenu.x}px`, top: `${miniSourceMenu.y}px` }"
      >
        <button type="button" @click="handleMiniSourceAction('rename')">
          <span class="material-symbols-outlined">edit</span>
          <span>이름 변경</span>
        </button>
        <div class="mini-source-menu-divider"></div>
        <button type="button" class="danger" @click="handleMiniSourceAction('delete')">
          <span class="material-symbols-outlined">delete</span>
          <span>삭제</span>
        </button>
      </div>
    </aside>

    <section
      ref="workspaceUnifiedCardRef"
      class="workspace-unified-card card relative z-10"
      :class="{
        'has-script-tab-line': workspaceActiveMainTab === 'materials' || workspaceActiveMainTab === 'quiz',
        'is-resizing-script-pane': isScriptPaneResizing
      }"
      :style="{ '--workspace-script-pane-width': `${scriptPaneWidth}%` }"
    >
      <div id="workspace-unified-audio-player-host" class="workspace-unified-audio-player-host"></div>

      <LeftSidebar
        embedded
        class="workspace-unified-script-pane"
        :isCollapsed="false"
        :isRecording="isRecording"
        :isRecordingPaused="isRecordingPaused"
        :recordingMode="recordingMode"
        :recordingTimeText="recordingTimeText"
        :recordingAudioLevel="recordingAudioLevel"
        :diarization-enabled="diarizationEnabled"
        :diarization-status="diarizationStatus"
        :transcriptions="transcriptions"
        :fileTree="fileTree"
        :favorites="favorites"
        :activeFileName="activeFileName"
        :activeFileId="activeFileId"
        :activeFileType="activeFileType"
        :citationSourceRequest="citationSourceRequest"
        :recordingSourceRequest="recordingSourceRequest"
        :source-panel-open="isMiniSourceOpen"
        :script-tab-line-visible="true"
        @toggle-source-panel="toggleMiniSourcePanel"
        @navigateHome="emit('navigateHome')"
        @fileSelect="(id, node) => emit('fileSelect', id, node)"
        @update:fileTree="emit('update:fileTree', $event)"
        @update:favorites="emit('update:favorites', $event)"
        @addToNote="(text, source) => emit('addToNote', text, source)"
        @askAi="(word) => emit('askAi', word)"
        @openStoredMaterial="emit('openStoredMaterial', $event)"
        @openRecording="handleOpenRecording"
        @startRecording="emit('startRecording', $event)"
        @pauseRecording="emit('pauseRecording')"
        @resumeRecording="emit('resumeRecording')"
        @stopRecording="handleStopRecordingRequest"
      />

      <div
        class="workspace-unified-resizer"
        role="separator"
        aria-label="스크립트 영역 너비 조절"
        aria-orientation="vertical"
        aria-valuemin="24"
        aria-valuemax="72"
        :aria-valuenow="Math.round(scriptPaneWidth)"
        tabindex="0"
        title="드래그해서 스크립트 영역 너비 조절"
        @pointerdown="handleScriptPanePointerDown"
        @keydown="handleScriptPaneResizeKeydown"
        @dblclick="resetScriptPaneWidth"
      ></div>

      <MainContent
        embedded
        class="workspace-unified-main-pane"
        :isRecording="isRecording"
        :isRecordingPaused="isRecordingPaused"
        :recordingMode="recordingMode"
        :recordingTimeText="recordingTimeText"
        :recordingAudioLevel="recordingAudioLevel"
        :diarization-enabled="diarizationEnabled"
        :diarization-status="diarizationStatus"
        :activeFileName="activeFileName"
        :activeFileId="activeFileId"
        :activeFileType="activeFileType"
        :currentAttachments="currentAttachments"
        :currentRecordings="currentRecordings"
        :folderFiles="activeFolderFiles"
        :transcriptions="transcriptions"
        :currentPreviewMaterial="currentPreviewMaterial"
        :materialEvidenceRequest="materialEvidenceRequest"
        :summaryState="summaryState"
        :summaryNotes="summaryNotes"
        :quizSource="activeWorkspaceSource"
        :summarySource="activeWorkspaceSource"
        :tabRequest="mainContentTabRequest"
        @startRecording="emit('startRecording', $event)"
        @pauseRecording="emit('pauseRecording')"
        @resumeRecording="emit('resumeRecording')"
        @stopRecording="handleStopRecordingRequest"
        @generateMaterialSummary="emit('generateMaterialSummary', $event)"
        @generateRecordingSummary="emit('generateRecordingSummary', $event)"
        @deleteSummary="emit('deleteSummary', $event)"
        @mainSidebarToggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
        @rightSidebarToggle="emit('rightSidebarToggle')"
        @askAi="(word) => emit('askAi', word)"
        @addToNote="(text, source) => emit('addToNote', text, source)"
        @uploadLectureMaterials="emit('uploadLectureMaterials', $event)"
        @closePreviewMaterial="emit('closePreviewMaterial')"
        @openStoredMaterial="emit('openStoredMaterial', $event)"
        @activeTabChange="workspaceActiveMainTab = $event"
      />
    </section>
    
    <RightSidebar 
      class="relative z-10"
      :visible="isRightSidebarVisible" 
      :aiInput="aiInput"
      :activeFileId="activeFileId"
      :chatSource="activeWorkspaceSource"
      @update:aiInput="emit('update:aiInput', $event)"
      @openEvidenceSource="openEvidenceSource"
    />

  </div>

  <Teleport to="body">
    <transition name="source-upload-modal">
      <div
        v-if="isSourceUploadDialogOpen"
        class="source-upload-backdrop"
        @click.self="closeSourceUploadDialog"
      >
        <section
          class="source-upload-dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby="source-upload-title"
        >
          <button
            type="button"
            class="source-upload-close"
            aria-label="소스 추가 창 닫기"
            @click="closeSourceUploadDialog"
          >
            <span class="material-symbols-outlined">close</span>
          </button>

          <div class="source-upload-heading">
            <span class="source-upload-kicker">소스 추가</span>
            <h2 id="source-upload-title">자료와 음성 파일을 추가하세요</h2>
            <p>음성파일또는 강의자료를 워크페이스 소스로 저장합니다.</p>
          </div>

          <button
            type="button"
            class="source-upload-dropzone"
            :class="{ 'is-dragging': isSourceUploadDragging }"
            @click="browseSourceUploadFiles"
            @dragenter.prevent="isSourceUploadDragging = true"
            @dragover.prevent="isSourceUploadDragging = true"
            @dragleave.prevent="isSourceUploadDragging = false"
            @drop.prevent="handleSourceUploadDrop"
          >
            <span class="material-symbols-outlined source-upload-icon">upload_file</span>
            <strong>파일을 드래그하거나 클릭해서 추가</strong>
          </button>
        </section>
      </div>
    </transition>
  </Teleport>

  <CitationPopover
    :visible="showCitePopover"
    :cite="currentCite"
    :position="citePopoverPos"
    @close="closeCitePopover"
    @openSource="openEvidenceSource"
  />
</template>

<style scoped>
.workspace-schedule-notice {
  position: fixed;
  top: 24px;
  right: 24px;
  z-index: 80;
  width: min(380px, calc(100vw - 32px));
  padding: 16px;
  color: #1f2937;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: 20px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}

.workspace-mini-card {
  --workspace-mini-collapsed-width: 76px;
  --workspace-mini-expanded-width: 220px;
  --workspace-mini-bg: var(--copy-bg, #050506);
  --workspace-mini-fg: #f8fafc;
  --workspace-mini-muted: #9ca3af;
  --workspace-mini-panel: #2b2d33;
  --workspace-mini-panel-hover: #343740;
  --workspace-mini-line: #e5e7eb;
  flex: 0 0 var(--workspace-mini-collapsed-width);
  width: var(--workspace-mini-collapsed-width);
  min-width: var(--workspace-mini-collapsed-width);
  position: relative;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 12px;
  padding: 14px;
  overflow: visible;
  border: 1px solid var(--workspace-mini-bg);
  border-radius: 24px;
  background: var(--workspace-mini-bg);
  box-shadow: none;
  transition: flex-basis 0.2s ease, width 0.2s ease, min-width 0.2s ease, box-shadow 0.2s ease;
}

.workspace-mini-card:not(.is-open) {
  cursor: pointer;
}

.workspace-mini-card.is-open {
  flex-basis: var(--workspace-mini-expanded-width);
  width: var(--workspace-mini-expanded-width);
  min-width: var(--workspace-mini-expanded-width);
  box-shadow: none;
}

.workspace-mini-toggle {
  width: 48px;
  height: 48px;
  flex: 0 0 48px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 999px;
  color: var(--workspace-mini-fg);
  background: transparent;
  box-shadow: none;
  outline: none;
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.workspace-mini-toggle:hover {
  color: var(--workspace-mini-fg);
  transform: translateY(-1px);
}

.workspace-mini-toggle-logo {
  width: 43px;
  height: 43px;
  display: block;
  object-fit: cover;
  border-radius: 999px;
  user-select: none;
  pointer-events: none;
}

.workspace-mini-top {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  flex: 0 0 auto;
  width: 100%;
}

.workspace-mini-card.is-open .workspace-mini-top {
  align-items: flex-start;
}

.workspace-mini-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  outline: none;
  transition: background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.workspace-mini-action:hover {
  transform: translateY(-1px);
}

.workspace-mini-actions {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
}

.workspace-mini-card.is-open .workspace-mini-actions {
  align-items: stretch;
  width: 100%;
}

.workspace-mini-add-wrap {
  position: relative;
}

.workspace-mini-action {
  position: relative;
  width: 44px;
  height: 44px;
  flex: 0 0 44px;
  border-radius: 14px;
  color: var(--workspace-mini-fg);
  background: var(--workspace-mini-bg);
}

.workspace-mini-add-label {
  display: none;
  font-size: 15px;
  font-weight: 800;
  color: inherit;
  white-space: nowrap;
}

.workspace-mini-card.is-open .workspace-mini-add-wrap {
  width: 100%;
}

.workspace-mini-card.is-open .workspace-mini-add-btn {
  width: 100%;
  height: 40px;
  flex-basis: 40px;
  border: 0;
  border-radius: 999px;
  gap: 9px;
  color: var(--workspace-mini-fg);
  background: var(--workspace-mini-panel);
  box-shadow: none;
}

.workspace-mini-card.is-open .workspace-mini-add-btn .material-symbols-outlined {
  font-size: 22px;
}

.workspace-mini-card.is-open .workspace-mini-add-label {
  display: inline;
}

.workspace-mini-action:hover {
  color: var(--workspace-mini-fg);
  background: var(--workspace-mini-panel-hover);
}

.workspace-mini-action:disabled {
  cursor: not-allowed;
  opacity: 0.45;
  transform: none;
}

.workspace-mini-action .material-symbols-outlined {
  font-size: 22px;
}

.workspace-mini-action::after {
  content: attr(data-label);
  position: absolute;
  top: 50%;
  left: calc(100% + 10px);
  z-index: 20;
  transform: translate(-6px, -50%);
  width: max-content;
  max-width: 116px;
  padding: 8px 11px;
  border-radius: 999px;
  color: #1f2937;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.16);
  border: 1px solid rgba(226, 232, 240, 0.95);
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.16s ease, transform 0.16s ease;
}

.workspace-mini-action:hover::after,
.workspace-mini-action:focus-visible::after {
  opacity: 1;
  transform: translate(0, -50%);
}

.workspace-mini-add-btn::after {
  display: none;
}

.workspace-mini-source-panel {
  flex: 1 1 auto;
  min-height: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
  opacity: 0;
  pointer-events: none;
  transform: translateX(-4px);
  transition: opacity 0.16s ease, transform 0.16s ease;
  overflow: hidden;
}

.workspace-mini-card.is-open .workspace-mini-source-panel {
  opacity: 1;
  pointer-events: auto;
  transform: translateX(0);
}

.workspace-mini-source-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 4px;
  color: var(--workspace-mini-fg);
}

.workspace-mini-source-heading span {
  font-size: 13px;
  font-weight: 800;
}

.workspace-mini-source-heading small {
  min-width: 30px;
  padding: 3px 7px;
  border-radius: 999px;
  text-align: center;
  font-size: 11px;
  font-weight: 800;
  color: #64748b;
  background: #f1f5f9;
}

.workspace-mini-source-tree {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 7px;
  overflow-y: auto;
  padding-right: 2px;
}

.mini-source-select-all,
.mini-source-check-row {
  width: 100%;
  min-width: 0;
  display: grid;
  align-items: center;
  column-gap: 8px;
  color: var(--workspace-mini-fg);
}

.mini-source-select-all {
  grid-template-columns: minmax(0, 1fr) 18px;
  padding: 2px 0 7px;
  font-size: 12px;
  font-weight: 800;
}

.mini-source-check-row {
  grid-template-columns: 26px minmax(0, 1fr) 18px;
  min-height: 36px;
  padding: 6px 0;
}

.mini-source-open-btn {
  grid-column: 2;
  min-width: 0;
  display: flex;
  align-items: center;
  border: 0;
  border-radius: 10px;
  padding: 6px 6px;
  color: #cbd5e1;
  background: transparent;
  text-align: left;
}

.mini-source-open-btn:hover {
  color: var(--workspace-mini-fg);
  background: var(--workspace-mini-panel);
}

.mini-source-open-btn span:last-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.25;
}

.mini-source-icon-button {
  grid-column: 1;
  position: relative;
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 8px;
  background: rgba(239, 246, 255, 0.92);
  color: #2563eb;
  cursor: pointer;
  transition: background-color 0.16s ease, transform 0.16s ease;
}

.mini-source-icon-button.recording {
  background: rgba(255, 242, 207, 0.9);
  color: #f59e0b;
}

.mini-source-icon-button:hover {
  transform: translateY(-1px);
}

.mini-source-item-icon {
  position: absolute;
  inset: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: currentColor;
  transition: opacity 0.14s ease;
}

.mini-source-icon-button .hover-icon {
  opacity: 0;
}

.mini-source-check-row:hover .mini-source-icon-button .default-icon,
.mini-source-icon-button:focus-visible .default-icon,
.mini-source-menu-open .default-icon {
  opacity: 0;
}

.mini-source-check-row:hover .mini-source-icon-button .hover-icon,
.mini-source-icon-button:focus-visible .hover-icon,
.mini-source-menu-open .hover-icon {
  opacity: 1;
}

.mini-source-select-all input,
.mini-source-check-row input {
  position: relative;
  flex: 0 0 auto;
  width: 17px;
  height: 17px;
  appearance: none;
  border: 1px solid #cbd5e1;
  border-radius: 3px;
  background: #f1f5f9;
  cursor: pointer;
}

.mini-source-select-all input:checked,
.mini-source-check-row input:checked {
  border-color: #cbd5e1;
  background: #d5dbe4;
}

.mini-source-select-all input:checked::after,
.mini-source-check-row input:checked::after {
  content: '';
  position: absolute;
  left: 5px;
  top: 1px;
  width: 5px;
  height: 10px;
  border: solid #64748b;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.mini-source-menu-backdrop {
  position: fixed;
  inset: 0;
  z-index: 80;
}

.mini-source-context-menu {
  position: fixed;
  z-index: 90;
  width: 158px;
  overflow: hidden;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
}

.mini-source-context-menu button {
  width: 100%;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  border: 0;
  padding: 12px 14px;
  color: #1f2937;
  background: transparent;
  font-size: 13px;
  font-weight: 800;
  text-align: left;
}

.mini-source-context-menu button:hover {
  background: #f8fafc;
}

.mini-source-context-menu button.danger {
  color: #ef4444;
}

.mini-source-context-menu .material-symbols-outlined {
  font-size: 17px;
  color: currentColor;
}

.mini-source-menu-divider {
  height: 1px;
  background: #e5e7eb;
}

.workspace-mini-source-empty {
  color: #94a3b8;
  font-size: 12px;
  font-weight: 700;
}

.workspace-mini-source-empty {
  flex: 1 1 auto;
  display: grid;
  place-items: center;
  min-height: 120px;
  border: 1px dashed #d7dee9;
  border-radius: 14px;
  text-align: center;
}

.workspace-unified-card {
  --workspace-script-pane-width: 50%;
  --workspace-audio-player-height: 96px;
  flex: 1 1 0%;
  min-width: 0;
  height: 100%;
  display: flex;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(226, 224, 232, 0.9);
  border-radius: 24px;
  box-shadow:
    0 26px 52px rgba(148, 163, 184, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.workspace-unified-card.is-resizing-script-pane,
.workspace-unified-card.is-resizing-script-pane * {
  cursor: col-resize !important;
  user-select: none;
}

.workspace-unified-card::before,
.workspace-unified-card::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  z-index: 12;
  height: 1px;
  background: rgba(0, 0, 0, 0.06);
  pointer-events: none;
}

.workspace-unified-card::before {
  top: 82px;
  display: none;
}

.workspace-unified-card.has-script-tab-line::before {
  display: none;
}

.workspace-unified-card::after {
  top: 48px;
}

.workspace-unified-audio-player-host {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 75;
  height: var(--workspace-audio-player-height);
  pointer-events: none;
}

.workspace-unified-audio-player-host :deep(*) {
  pointer-events: auto;
}

.workspace-unified-script-pane {
  position: relative;
  flex: 0 0 var(--workspace-script-pane-width);
  width: var(--workspace-script-pane-width) !important;
  transition: flex-basis 0.16s ease, width 0.16s ease;
}

.workspace-unified-script-pane::after {
  display: none;
}

.workspace-unified-main-pane {
  flex: 1 1 0%;
  min-width: 340px !important;
}

.workspace-unified-resizer {
  position: relative;
  z-index: 96;
  flex: 0 0 24px;
  width: 24px;
  align-self: stretch;
  margin-left: -12px;
  margin-right: -12px;
  cursor: col-resize;
  touch-action: none;
  outline: none;
  background: transparent;
}

.workspace-unified-resizer::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 2px;
  transform: translateX(-50%);
  background: rgba(226, 224, 232, 0.95);
  transition: width 0.16s ease, background-color 0.16s ease;
}

.workspace-unified-resizer::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 6px;
  height: 54px;
  border-radius: 999px;
  transform: translate(-50%, -50%);
  background: #c7ccd6;
  opacity: 0;
  transition: opacity 0.16s ease, background-color 0.16s ease;
}

.workspace-unified-resizer:hover::before,
.workspace-unified-resizer:focus-visible::before,
.workspace-unified-card.is-resizing-script-pane .workspace-unified-resizer::before {
  width: 3px;
  background: #9aa3b2;
}

.workspace-unified-resizer:hover::after,
.workspace-unified-resizer:focus-visible::after,
.workspace-unified-card.is-resizing-script-pane .workspace-unified-resizer::after {
  opacity: 1;
}

.workspace-unified-card.is-resizing-script-pane .workspace-unified-script-pane {
  transition: none;
}

.workspace-unified-card:has(.is-unified-audio-player) .workspace-unified-resizer {
  align-self: flex-start;
  height: calc(100% - var(--workspace-audio-player-height));
}

@media (max-width: 1280px) {
  .workspace-unified-card {
    --workspace-audio-player-height: 98px;
  }
}

@media (max-width: 1440px) {
  .workspace-page-shell {
    padding: 10px !important;
  }

  .workspace-unified-script-pane {
    flex-basis: var(--workspace-script-pane-width) !important;
    width: var(--workspace-script-pane-width) !important;
  }

  .workspace-unified-main-pane {
    min-width: 340px !important;
  }
}

@media (max-width: 1180px) {
  .workspace-page-shell {
    gap: 8px !important;
  }

  .workspace-mini-card {
    --workspace-mini-expanded-width: 220px;
    padding: 14px 10px;
    border-radius: 20px;
  }

  .workspace-unified-card {
    border-radius: 20px;
  }

  .workspace-unified-script-pane {
    flex: 0 0 var(--workspace-script-pane-width) !important;
    width: var(--workspace-script-pane-width) !important;
  }
}

@media (max-width: 1024px) {
  .workspace-page-shell {
    padding: 8px !important;
    gap: 0 !important;
  }

  .workspace-mini-card {
    display: none;
  }

  .workspace-unified-card {
    width: 100%;
    min-height: calc(100vh - 16px);
    height: calc(100vh - 16px);
  }

  .workspace-unified-resizer {
    width: 28px;
    flex-basis: 28px;
    margin-left: -14px;
    margin-right: -14px;
  }
}

@media (max-width: 760px) {
  .workspace-page-shell {
    overflow-y: auto !important;
    align-items: stretch;
  }

  .workspace-unified-card {
    flex-direction: column;
    min-height: 100%;
    height: auto;
    overflow: visible;
  }

  .workspace-unified-script-pane {
    flex: 0 0 auto !important;
    width: 100% !important;
    min-height: 320px;
    max-height: 42vh;
    overflow: hidden;
  }

  .workspace-unified-main-pane {
    flex: 1 1 auto;
    min-height: 560px;
    width: 100%;
  }

  .workspace-unified-resizer {
    display: none;
  }

  .workspace-unified-card::after {
    top: 0;
    display: none;
  }
}

.workspace-schedule-notice-top {
  display: flex;
  align-items: center;
  gap: 12px;
}

.workspace-schedule-notice-icon {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #2563eb;
  background: #eef4ff;
  border: 1px solid #dbe7ff;
  flex: 0 0 auto;
}

.workspace-schedule-notice-icon .material-symbols-outlined {
  font-size: 20px;
}

.workspace-schedule-notice-heading {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.workspace-schedule-notice-heading span {
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

.workspace-schedule-notice-heading strong {
  color: #111827;
  font-size: 16px;
  font-weight: 900;
  line-height: 1.25;
}

.workspace-schedule-notice-close {
  width: 32px;
  height: 32px;
  border: 0;
  border-radius: 999px;
  color: #94a3b8;
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.workspace-schedule-notice-close:hover {
  color: #475569;
  background: #f1f5f9;
}

.workspace-schedule-notice-close .material-symbols-outlined {
  font-size: 19px;
}

.workspace-schedule-notice-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}

.workspace-schedule-notice-item {
  padding: 12px;
  border-radius: 14px;
  background: #f8fafc;
  border: 1px solid #e5edf6;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.workspace-schedule-notice-item-info {
  flex: 1;
  min-width: 0;
}

.workspace-schedule-notice-item-info strong,
.workspace-schedule-notice-item-info span {
  display: block;
  overflow-wrap: anywhere;
}

.workspace-schedule-notice-item-info strong {
  color: #111827;
  font-size: 14px;
  font-weight: 900;
  line-height: 1.35;
}

.workspace-schedule-notice-item-info span {
  margin-top: 5px;
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

.workspace-schedule-notice-item-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.workspace-schedule-notice-item-btn {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  border: 0;
  border-radius: 8px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 800;
  cursor: pointer;
  transition: background 0.15s ease, transform 0.15s ease;
}

.workspace-schedule-notice-item-btn .material-symbols-outlined {
  font-size: 14px;
}

.workspace-schedule-notice-item-btn.confirm {
  color: #ffffff;
  background: #2563eb;
}

.workspace-schedule-notice-item-btn.confirm:hover {
  background: #1d4ed8;
  transform: translateY(-1px);
}

.workspace-schedule-notice-item-btn.confirm:disabled {
  opacity: 0.6;
  cursor: default;
  transform: none;
}

.workspace-schedule-notice-item-btn.ignore {
  color: #64748b;
  background: #e2e8f0;
}

.workspace-schedule-notice-item-btn.ignore:hover {
  background: #cbd5e1;
  transform: translateY(-1px);
}

.workspace-schedule-notice-more {
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
  padding: 0 4px;
}

.workspace-schedule-notice-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}

.workspace-schedule-notice-primary,
.workspace-schedule-notice-secondary {
  min-height: 38px;
  border: 0;
  border-radius: 999px;
  padding: 0 15px;
  font-size: 13px;
  font-weight: 900;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.workspace-schedule-notice-primary {
  color: #ffffff;
  background: #1f2937;
}

.workspace-schedule-notice-secondary {
  color: #475569;
  background: #f1f5f9;
}

.workspace-schedule-notice-primary:hover,
.workspace-schedule-notice-secondary:hover {
  transform: translateY(-1px);
}

.schedule-notice-fade-enter-active,
.schedule-notice-fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.schedule-notice-fade-enter-from,
.schedule-notice-fade-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

.source-upload-backdrop {
  position: fixed;
  inset: 0;
  z-index: 120;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(15, 23, 42, 0.34);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.source-upload-dialog {
  position: relative;
  width: min(620px, calc(100vw - 32px));
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 34px;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 24px;
  color: #111827;
  background: #ffffff;
  box-shadow: 0 30px 80px rgba(15, 23, 42, 0.22);
}

.source-upload-close {
  position: absolute;
  top: 18px;
  right: 18px;
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 999px;
  color: #111827;
  background: #f1eef6;
  cursor: pointer;
  transition: background 0.16s ease, color 0.16s ease, transform 0.16s ease;
}

.source-upload-close:hover {
  color: #111827;
  background: #e8e3ee;
  transform: translateY(-1px);
}

.source-upload-close .material-symbols-outlined {
  font-size: 20px;
}

.source-upload-heading {
  max-width: 520px;
  padding-right: 28px;
}

.source-upload-kicker {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  margin-bottom: 10px;
  padding: 0 10px;
  border-radius: 999px;
  color: #2563eb;
  background: rgba(219, 234, 254, 0.9);
  font-size: 12px;
  font-weight: 900;
}

.source-upload-heading h2 {
  margin: 0;
  color: #111827;
  font-size: 25px;
  line-height: 1.25;
  font-weight: 900;
}

.source-upload-heading p {
  margin: 10px 0 0;
  color: #64748b;
  font-size: 14px;
  line-height: 1.55;
  font-weight: 700;
}

.source-upload-dropzone {
  width: 100%;
  min-height: 220px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 28px;
  border: 2px dashed #cbd5e1;
  border-radius: 20px;
  color: #334155;
  background: rgba(248, 250, 252, 0.82);
  cursor: pointer;
  text-align: center;
  transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}

.source-upload-dropzone:hover,
.source-upload-dropzone.is-dragging {
  border-color: #2563eb;
  background: rgba(239, 246, 255, 0.92);
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.14);
  transform: translateY(-1px);
}

.source-upload-icon {
  width: 52px;
  height: 52px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 16px;
  color: #2563eb;
  background: #dbeafe;
  font-size: 28px;
}

.source-upload-dropzone strong {
  color: #111827;
  font-size: 18px;
  line-height: 1.35;
  font-weight: 900;
}

.source-upload-dropzone span:not(.material-symbols-outlined) {
  color: #64748b;
  font-size: 13px;
  line-height: 1.45;
  font-weight: 800;
}

.source-upload-modal-enter-active,
.source-upload-modal-leave-active {
  transition: opacity 0.18s ease;
}

.source-upload-modal-enter-active .source-upload-dialog,
.source-upload-modal-leave-active .source-upload-dialog {
  transition: transform 0.18s ease, opacity 0.18s ease;
}

.source-upload-modal-enter-from,
.source-upload-modal-leave-to {
  opacity: 0;
}

.source-upload-modal-enter-from .source-upload-dialog,
.source-upload-modal-leave-to .source-upload-dialog {
  opacity: 0;
  transform: translateY(10px) scale(0.98);
}

/* 팝오버 스크롤바 디자인 */
.custom-scrollbar::-webkit-scrollbar {
  width: 5px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.08);
  border-radius: 10px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.15);
}
</style>
