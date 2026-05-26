<!-- 워크스페이스의 중앙 영역으로, 강의 자료 뷰어, 녹음 조작, 노점 요약 내용을 표시합니다. -->
<script setup>
import { ref, computed, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import WorkspaceWordCard from './MainContent/WorkspaceWordCard.vue'
import WorkspaceHeader from './MainContent/WorkspaceHeader.vue'
import LecturePreviewPanel from './MainContent/LecturePreviewPanel.vue'
import WorkspaceQuizPanel from './Quiz/WorkspaceQuizPanel.vue'
import WorkspaceSummaryPanel from './Summary/WorkspaceSummaryPanel.vue'
import LoadingHourglass from '../ui/LoadingHourglass.vue'

const {
  selectedWordData,
  isWordCardVisible,
  hideSelectedWordCard,
  showSelectedWordCard,
  clearSelectedWord
} = useChat()

const props = defineProps({
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingMode: { type: String, default: 'lecture' },
  recordingTimeText: String,
  recordingAudioLevel: { type: Number, default: 0 },
  diarizationEnabled: { type: Boolean, default: false },
  diarizationStatus: { type: String, default: 'idle' },
  activeFileName: String,
  activeFileId: String,
  activeFileType: { type: String, default: 'lecture' },
  currentAttachments: { type: Array, default: () => [] },
  currentRecordings: { type: Array, default: () => [] },
  transcriptions: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  materialEvidenceRequest: { type: Object, default: null },
  summaryState: { type: Object, default: () => ({}) },
  summaryNotes: { type: Array, default: () => [] },
  quizSource: { type: Object, default: null },
  tabRequest: { type: Object, default: null },
  embedded: { type: Boolean, default: false }
})

const emit = defineEmits([
  'startRecording',
  'pauseRecording',
  'resumeRecording',
  'stopRecording',
  'generateMaterialSummary',
  'deleteSummary',
  'mainSidebarToggle',
  'rightSidebarToggle',
  'askAi',
  'addToNote',
  'uploadLectureMaterials',
  'openStoredMaterial',
  'closePreviewMaterial',
  'activeTabChange'
])

const activeTab = ref('materials')
const tabAnim = ref('tab-slide-right')
const isMaterialDragOver = ref(false)
const showDiarizationChoice = ref(false)
const materialInputRef = ref(null)
const folderOpenAnimationRef = ref(null)
const pdfSearchQuery = ref('')
const pdfSearchCommand = ref(null)
const pdfSearchState = ref({ total: 0, activeIndex: 0 })
const pdfSearchCommandSeq = ref(0)
let prevTab = 'materials'

const TAB_ORDER = ['materials', 'summary', 'quiz']

const tabs = computed(() => [
  { key: 'materials', label: '자료' },
  { key: 'summary', label: '요약' },
  { key: 'quiz', label: '퀴즈' }
])

const materialTitle = computed(() => props.activeFileName || '파일을 선택하세요')
const materialCards = computed(() => Array.isArray(props.currentAttachments) ? props.currentAttachments : [])
const isCurrentPreviewPdf = computed(() => {
  const material = props.currentPreviewMaterial
  if (!material) return false
  return material.type === 'application/pdf' || /\.pdf$/i.test(material.name || material.storedName || '')
})

const allowedMaterialTypes = [
  'application/pdf',
  'application/vnd.ms-powerpoint',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation'
]

const getDefaultTabByFileType = () => 'materials'

const handleTabChange = (newTab) => {
  const prevIdx = TAB_ORDER.indexOf(prevTab)
  const nextIdx = TAB_ORDER.indexOf(newTab)
  tabAnim.value = nextIdx > prevIdx ? 'tab-slide-right' : 'tab-slide-left'
  prevTab = newTab
  activeTab.value = newTab
  emit('activeTabChange', newTab)
}

watch(activeTab, (newVal) => {
  if (newVal !== prevTab) {
    handleTabChange(newVal)
  }
})

watch(
  () => [props.activeFileId, props.activeFileType],
  ([nextFileId, nextFileType], [prevFileId, prevFileType] = []) => {
    if (nextFileId === prevFileId && nextFileType === prevFileType) return

    const defaultTab = getDefaultTabByFileType(nextFileType)
    prevTab = defaultTab
    activeTab.value = defaultTab
    emit('activeTabChange', defaultTab)
  },
  { immediate: true }
)

watch(
  () => props.currentPreviewMaterial,
  (nextMaterial, prevMaterial) => {
    if (!nextMaterial || nextMaterial.id !== prevMaterial?.id) {
      resetPdfSearch()
    }
    if (!nextMaterial || nextMaterial.id === prevMaterial?.id) return
    handleTabChange('materials')
  }
)

watch(isCurrentPreviewPdf, (isPdf) => {
  if (!isPdf) resetPdfSearch()
})

watch(
  () => props.materialEvidenceRequest,
  (request) => {
    if (!request) return
    handleTabChange('materials')
  }
)

watch(
  () => props.tabRequest,
  (request) => {
    if (!request || !TAB_ORDER.includes(request.tab)) return
    handleTabChange(request.tab)
  }
)

const isLectureMaterialFile = (file) => {
  if (!file) return false
  return allowedMaterialTypes.includes(file.type) || /\.(pdf|ppt|pptx)$/i.test(file.name)
}

const openMaterialInMaterials = (file) => {
  emit('uploadLectureMaterials', [file])
  handleTabChange('materials')
}

const handleMaterialSelection = (file) => {
  if (isLectureMaterialFile(file)) {
    openMaterialInMaterials(file)
  }
}

const handleDroppedMaterial = (event) => {
  event.preventDefault()
  isMaterialDragOver.value = false
  const file = Array.from(event.dataTransfer?.files || []).find(isLectureMaterialFile)
  if (file) {
    openMaterialInMaterials(file)
  }
}

const triggerMaterialUpload = () => {
  materialInputRef.value?.click()
}

const playFolderOpenAnimation = () => {
  folderOpenAnimationRef.value?.playFromStart?.()
}

const sendPdfSearchCommand = (action) => {
  pdfSearchCommandSeq.value += 1
  pdfSearchCommand.value = { action, nonce: pdfSearchCommandSeq.value }
}

const resetPdfSearch = () => {
  pdfSearchQuery.value = ''
  pdfSearchState.value = { total: 0, activeIndex: 0 }
  sendPdfSearchCommand('clear')
}

const handlePdfSearchChange = (query) => {
  pdfSearchQuery.value = query
}

const handlePdfSearchResults = (payload = {}) => {
  pdfSearchState.value = {
    total: Number(payload.total || 0),
    activeIndex: Number(payload.activeIndex || 0)
  }
}

const handleMaterialInputChange = (event) => {
  const file = Array.from(event.target.files || []).find(isLectureMaterialFile)
  if (file) handleMaterialSelection(file)
  event.target.value = ''
}

const handleOpenStoredMaterial = (material) => {
  const materialId = material?.id
  if (!materialId) return
  emit('openStoredMaterial', materialId)
  handleTabChange('materials')
}

const getMaterialIcon = (material) => (
  /\.(ppt|pptx)$/i.test(material?.name || material?.storedName || '') ? 'slideshow' : 'picture_as_pdf'
)

const formatMaterialSize = (size = 0) => {
  const bytes = Number(size) || 0
  if (!bytes) return '파일'
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

const formatMaterialDate = (value = '') => {
  if (!value) return ''
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return ''
  const month = parsed.getMonth() + 1
  const day = parsed.getDate()
  return `${month}.${day}`
}

const handleAskAi = () => {
  if (selectedWordData.value) {
    emit('askAi', selectedWordData.value.word)
    clearSelectedWord()
  }
}

const handleAddToNote = () => {
  if (selectedWordData.value) {
    emit('addToNote', selectedWordData.value.desc, selectedWordData.value.source)
    clearSelectedWord()
  }
}

const handleWordInsightButtonClick = () => {
  if (selectedWordData.value) {
    if (isWordCardVisible.value) {
      hideSelectedWordCard()
    } else {
      showSelectedWordCard()
    }
  }
}

const getRecordingModeForActiveFile = () => (
  props.activeFileType === 'meeting' ? 'meeting' : 'lecture'
)

const handleStartRecording = () => {
  showDiarizationChoice.value = true
}

const startRecordingWithDiarization = (enabled) => {
  showDiarizationChoice.value = false
  emit('startRecording', {
    mode: getRecordingModeForActiveFile(),
    diarizationEnabled: enabled
  })
}

const postRecordingProcessing = computed(() => {
  if (props.diarizationEnabled && props.diarizationStatus === 'finalizing') {
    return {
      icon: 'graphic_eq',
      title: '전체 녹음 화자분리 중',
      description: '녹음 전체를 다시 분석해 화자 구간을 정리하고 있습니다.',
      tone: 'speaker'
    }
  }

  if (!props.isRecording && props.summaryState?.status === 'generating') {
    const isSpeakerSummary = props.summaryState?.diarizationEnabled ?? props.diarizationEnabled
    return {
      icon: 'auto_awesome',
      title: '최종 요약 생성 중',
      description: isSpeakerSummary
        ? '전체 녹음 요약과 화자별 요약을 함께 생성하고 있습니다.'
        : '전체 전사문을 기준으로 녹음 요약을 생성하고 있습니다.',
      tone: 'summary'
    }
  }

  return null
})
</script>

<template>
  <main
    class="flex-1 flex flex-col gap-[12px] h-full min-w-0"
    :class="{ 'main-content-embedded': embedded }"
    :style="embedded
      ? { flex: '1 1 0%', minWidth: '340px' }
      : { flex: '1 1 0%', minWidth: '300px' }"
  >
    <transition name="word-card">
      <WorkspaceWordCard
        v-if="selectedWordData && isWordCardVisible"
        :word-data="selectedWordData"
        @close="hideSelectedWordCard"
        @ask-ai="handleAskAi"
        @add-to-note="handleAddToNote"
      />
    </transition>

    <div
      id="tab-contents-container"
      class="card workspace-shell-card flex-1 flex flex-col relative min-h-0 min-w-0 overflow-hidden"
      :class="{ 'is-embedded': embedded }"
    >
      <WorkspaceHeader
        :embedded="embedded"
        :is-recording="isRecording"
        :is-recording-paused="isRecordingPaused"
        :recording-time-text="recordingTimeText"
        :recording-audio-level="recordingAudioLevel"
        :tabs="tabs"
        :active-tab="activeTab"
        :show-close-preview="activeTab === 'materials' && !!currentPreviewMaterial"
        :show-pdf-search="activeTab === 'materials' && isCurrentPreviewPdf"
        :pdf-search-total="pdfSearchState.total"
        :pdf-search-active-index="pdfSearchState.activeIndex"
        :has-word-insight="!!selectedWordData"
        :word-insight-visible="!!selectedWordData && isWordCardVisible"
        @start-recording="handleStartRecording"
        @pause-recording="emit('pauseRecording')"
        @resume-recording="emit('resumeRecording')"
        @stop-recording="emit('stopRecording')"
        @tab-change="handleTabChange"
        @main-sidebar-toggle="emit('mainSidebarToggle')"
        @right-sidebar-toggle="emit('rightSidebarToggle')"
        @material-selected="handleMaterialSelection"
        @word-insight-click="handleWordInsightButtonClick"
        @close-preview-material="emit('closePreviewMaterial')"
        @pdf-search-change="handlePdfSearchChange"
        @pdf-search-next="sendPdfSearchCommand('next')"
        @pdf-search-prev="sendPdfSearchCommand('prev')"
        @pdf-search-clear="resetPdfSearch"
      />

      <transition name="post-processing">
        <section
          v-if="postRecordingProcessing"
          class="post-processing-band"
          :class="`is-${postRecordingProcessing.tone}`"
          role="status"
          aria-live="polite"
        >
          <div class="post-processing-icon">
            <span class="material-symbols-outlined">{{ postRecordingProcessing.icon }}</span>
          </div>
          <div class="post-processing-copy">
            <strong>{{ postRecordingProcessing.title }}</strong>
            <span>{{ postRecordingProcessing.description }}</span>
          </div>
          <div class="post-processing-meter" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
        </section>
      </transition>

      <div class="flex-1 flex flex-col relative min-h-0 min-w-0">
        <section
          v-if="activeTab === 'materials'"
          :key="'tab-materials'"
          :class="[
            'tab-content flex-1 flex flex-col relative overflow-hidden materials-canvas',
            currentPreviewMaterial ? 'px-4 pt-4 pb-0' : 'p-10 pt-4',
            tabAnim
          ]"
          @dragover.prevent="isMaterialDragOver = true"
          @dragenter.prevent="isMaterialDragOver = true"
          @dragleave.prevent="isMaterialDragOver = false"
          @drop="handleDroppedMaterial"
        >
          <div :class="[currentPreviewMaterial ? 'w-full h-full flex flex-col' : 'materials-tab-shell']">
            <div v-if="currentPreviewMaterial" class="preview-panel-wrap">
              <LecturePreviewPanel
                :material="currentPreviewMaterial"
                :evidence-request="materialEvidenceRequest"
                :pdf-search-query="pdfSearchQuery"
                :pdf-search-command="pdfSearchCommand"
                @pdf-search-results="handlePdfSearchResults"
              />
            </div>

            <template v-else>
              <div class="materials-tab-head">
                <div>
                  <h1>강의자료</h1>
                  <p>{{ materialTitle }}</p>
                </div>
              </div>

              <div
                v-if="materialCards.length"
                class="materials-grid"
                :class="{ 'is-drag-over': isMaterialDragOver }"
              >
                <button
                  v-for="material in materialCards"
                  :key="material.id || material.name"
                  type="button"
                  class="material-card"
                  @click="handleOpenStoredMaterial(material)"
                >
                  <span class="material-card-icon material-symbols-outlined">{{ getMaterialIcon(material) }}</span>
                  <span class="material-card-copy">
                    <strong>{{ material.name || material.title || material.storedName || '강의자료' }}</strong>
                    <small>
                      {{ formatMaterialSize(material.size) }}
                      <template v-if="formatMaterialDate(material.uploadedAt || material.createdAt)">
                        · {{ formatMaterialDate(material.uploadedAt || material.createdAt) }}
                      </template>
                    </small>
                  </span>
                  <span class="material-card-open material-symbols-outlined">open_in_new</span>
                </button>
              </div>

              <button
                v-else
                type="button"
                class="materials-empty-state"
                :class="{ 'is-drag-over': isMaterialDragOver }"
                aria-label="강의자료 추가"
                @click="triggerMaterialUpload"
                @pointerenter="playFolderOpenAnimation"
                @mouseenter="playFolderOpenAnimation"
                @focus="playFolderOpenAnimation"
              >
                <LoadingHourglass
                  ref="folderOpenAnimationRef"
                  class="materials-empty-animation"
                  src="/animations/Folder%20Open.json"
                  width="88px"
                  height="88px"
                  :autoplay="false"
                  :loop="false"
                  fallback-icon="folder_open"
                />
                <strong>강의자료가 없습니다</strong>
              </button>
            </template>

            <input
              ref="materialInputRef"
              type="file"
              accept=".pdf,.ppt,.pptx,application/pdf,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"
              class="hidden"
              @change="handleMaterialInputChange"
            />
          </div>
        </section>

        <WorkspaceSummaryPanel
          v-else-if="activeTab === 'summary'"
          :key="'tab-summary'"
          :tab-anim="tabAnim"
          :is-recording="isRecording"
          :is-recording-paused="isRecordingPaused"
          :recording-mode="recordingMode"
          :diarization-enabled="diarizationEnabled"
          :transcriptions="transcriptions"
          :summary-state="summaryState"
          :current-recordings="currentRecordings"
          :active-file-id="activeFileId"
          @deleteSummary="emit('deleteSummary', $event)"
          @askAi="emit('askAi', $event)"
          @addToNote="(text, source) => emit('addToNote', text, source)"
        />

        <WorkspaceQuizPanel
          v-else-if="activeTab === 'quiz'"
          :key="'tab-quiz'"
          :tab-anim="tabAnim"
          :active-file-name="activeFileName"
          :active-file-id="activeFileId"
          :current-preview-material="currentPreviewMaterial"
          :quiz-source="quizSource"
        />
      </div>
    </div>

    <Teleport to="body">
      <transition name="recording-choice-fade">
        <div
          v-if="showDiarizationChoice"
          class="recording-choice-overlay"
          @click.self="showDiarizationChoice = false"
        >
          <section class="recording-choice-dialog" role="dialog" aria-modal="true" aria-label="녹음 방식 선택">
            <div class="recording-choice-icon">
              <span class="material-symbols-outlined">graphic_eq</span>
            </div>
            <h2>화자분리를 사용할까요?</h2>
            <p>
              화자분리를 사용하면 시작 직후 몇 초 동안 화자를 분석한 뒤 전사가 표시됩니다.
              사용하지 않으면 바로 전사하고 전체 녹음 요약만 생성합니다.
            </p>
            <div class="recording-choice-actions">
              <button type="button" class="recording-choice-secondary" @click="startRecordingWithDiarization(false)">
                바로 녹음
              </button>
              <button type="button" class="recording-choice-primary" @click="startRecordingWithDiarization(true)">
                화자분리 사용
              </button>
            </div>
          </section>
        </div>
      </transition>
    </Teleport>
  </main>
</template>

<style scoped>
.workspace-shell-card {
  background: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow:
    0 26px 52px rgba(148, 163, 184, 0.08),
    0 10px 24px rgba(0, 0, 0, 0.02);
}

.workspace-shell-card::before {
  display: none;
}

.workspace-shell-card::after {
  display: none;
}

.main-content-embedded {
  gap: 0;
  min-width: 340px !important;
}

.workspace-shell-card.is-embedded {
  border: 0;
  border-radius: 0;
  background: #ffffff;
  box-shadow: none;
}

.workspace-shell-card.is-embedded::before,
.workspace-shell-card.is-embedded::after {
  display: none;
}

@media (max-width: 760px) {
  .main-content-embedded {
    min-width: 0 !important;
  }
}

.post-processing-band {
  width: 100%;
  min-height: 58px;
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px;
  padding: 10px 22px;
  border-top: 1px solid rgba(226, 232, 240, 0.75);
  border-bottom: 1px solid rgba(226, 232, 240, 0.85);
  background: #f8fafc;
}

.post-processing-band.is-speaker {
  background: #f8fbff;
}

.post-processing-band.is-summary {
  background: #fbfaf7;
}

.post-processing-icon {
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #1d4ed8;
  background: #eaf2ff;
}

.post-processing-band.is-summary .post-processing-icon {
  color: #9a3412;
  background: #fff3e7;
}

.post-processing-icon .material-symbols-outlined {
  font-size: 22px;
}

.post-processing-copy {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.post-processing-copy strong {
  color: #111827;
  font-size: 14px;
  font-weight: 950;
  letter-spacing: 0;
}

.post-processing-copy span {
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.45;
  word-break: keep-all;
}

.post-processing-meter {
  width: 58px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 5px;
}

.post-processing-meter span {
  width: 5px;
  height: 8px;
  border-radius: 999px;
  background: #2563eb;
  opacity: 0.36;
  animation: post-processing-pulse 1s ease-in-out infinite;
}

.post-processing-band.is-summary .post-processing-meter span {
  background: #c2410c;
}

.post-processing-meter span:nth-child(2) {
  animation-delay: 0.12s;
}

.post-processing-meter span:nth-child(3) {
  animation-delay: 0.24s;
}

.post-processing-meter span:nth-child(4) {
  animation-delay: 0.36s;
}

@keyframes post-processing-pulse {
  0%, 100% {
    height: 8px;
    opacity: 0.35;
  }
  50% {
    height: 22px;
    opacity: 0.95;
  }
}

.post-processing-enter-active,
.post-processing-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.post-processing-enter-from,
.post-processing-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

.preview-panel-wrap {
  flex: 1;
  min-height: 0;
}

.materials-tab-shell {
  width: min(100%, 980px);
  height: 100%;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.materials-tab-head {
  flex: 0 0 auto;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 24px;
}

.materials-tab-head h1 {
  margin: 0;
  color: #1d1d1f;
  font-size: 28px;
  font-weight: 950;
  letter-spacing: 0;
}

.materials-tab-head p {
  margin: 6px 0 0;
  color: #8e8e93;
  font-size: 13px;
  font-weight: 800;
}

.materials-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 12px;
  padding: 2px;
  overflow-y: auto;
}

.materials-grid.is-drag-over,
.materials-empty-state.is-drag-over {
  outline: 1.5px dashed rgba(59, 130, 246, 0.44);
  outline-offset: 8px;
  background: rgba(239, 246, 255, 0.42);
}

.material-card {
  min-width: 0;
  min-height: 78px;
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) 24px;
  align-items: center;
  gap: 12px;
  padding: 14px;
  border-radius: 8px;
  border: 1px solid rgba(226, 232, 240, 0.9);
  background: #ffffff;
  text-align: left;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.045);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.material-card:hover {
  transform: translateY(-1px);
  border-color: rgba(148, 163, 184, 0.5);
  box-shadow: 0 20px 42px rgba(15, 23, 42, 0.075);
}

.material-card-icon {
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #2563eb;
  background: #eef4ff;
  font-size: 21px;
  font-variation-settings: 'FILL' 1;
}

.material-card-copy {
  min-width: 0;
  display: grid;
  gap: 4px;
}

.material-card-copy strong {
  overflow: hidden;
  color: #1f2937;
  font-size: 13px;
  font-weight: 950;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.material-card-copy small {
  color: #8e8e93;
  font-size: 11px;
  font-weight: 800;
}

.material-card-open {
  color: #9ca3af;
  font-size: 18px;
}

.materials-empty-state {
  min-height: 240px;
  display: grid;
  place-items: center;
  align-content: center;
  gap: 12px;
  border: 1px dashed rgba(203, 213, 225, 0.95);
  border-radius: 8px;
  color: #9ca3af;
  background: rgba(248, 250, 252, 0.74);
  cursor: pointer;
  text-align: center;
  transition: background-color 0.2s ease, border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}

.materials-empty-state:hover,
.materials-empty-state:focus-visible {
  border-color: rgba(59, 130, 246, 0.34);
  background: rgba(239, 246, 255, 0.5);
  box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.08);
}

.materials-empty-state:active {
  transform: scale(0.995);
}

.materials-empty-animation {
  opacity: 0.9;
  user-select: none;
  pointer-events: none;
}

.materials-empty-state strong {
  color: #8e8e93;
  font-size: 14px;
  font-weight: 900;
}

.recording-choice-overlay {
  position: fixed;
  inset: 0;
  z-index: 120;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(15, 23, 42, 0.28);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

.recording-choice-dialog {
  width: min(420px, 100%);
  display: grid;
  gap: 14px;
  padding: 24px;
  border: 1px solid rgba(226, 232, 240, 0.92);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 24px 56px rgba(15, 23, 42, 0.18);
}

.recording-choice-icon {
  width: 42px;
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #2563eb;
  background: #eef4ff;
  border: 1px solid #dbe7ff;
}

.recording-choice-icon .material-symbols-outlined {
  font-size: 23px;
}

.recording-choice-dialog h2 {
  margin: 0;
  color: #111827;
  font-size: 20px;
  font-weight: 950;
  letter-spacing: 0;
}

.recording-choice-dialog p {
  margin: 0;
  color: #64748b;
  font-size: 13px;
  font-weight: 700;
  line-height: 1.65;
  word-break: keep-all;
}

.recording-choice-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding-top: 4px;
}

.recording-choice-primary,
.recording-choice-secondary {
  min-height: 38px;
  padding: 0 15px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 900;
  transition: transform 0.2s ease, background-color 0.2s ease, border-color 0.2s ease;
}

.recording-choice-primary {
  color: #ffffff;
  background: #2563eb;
}

.recording-choice-secondary {
  color: #334155;
  background: #ffffff;
  border: 1px solid rgba(203, 213, 225, 0.9);
}

.recording-choice-primary:hover {
  background: #1d4ed8;
}

.recording-choice-secondary:hover {
  border-color: #94a3b8;
  background: #f8fafc;
}

.recording-choice-primary:active,
.recording-choice-secondary:active {
  transform: scale(0.98);
}

.recording-choice-fade-enter-active,
.recording-choice-fade-leave-active {
  transition: opacity 0.18s ease;
}

.recording-choice-fade-enter-from,
.recording-choice-fade-leave-to {
  opacity: 0;
}

.word-card-enter-active {
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.word-card-leave-active {
  transition: all 0.2s ease;
}

.word-card-enter-from {
  opacity: 0;
  transform: translateY(-8px) scaleY(0.9);
  max-height: 0;
}

.word-card-enter-to {
  opacity: 1;
  transform: translateY(0) scaleY(1);
  max-height: 200px;
}

.word-card-leave-from {
  opacity: 1;
  transform: translateY(0) scaleY(1);
  max-height: 200px;
}

.word-card-leave-to {
  opacity: 0;
  transform: translateY(-8px) scaleY(0.9);
  max-height: 0;
}
</style>
