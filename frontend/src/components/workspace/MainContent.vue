<!-- 워크스페이스의 중앙 영역으로, 강의 자료 뷰어, 녹음 조작, 노점 요약 내용을 표시합니다. -->
<script setup>
import { ref, computed, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import WorkspaceWordCard from './MainContent/WorkspaceWordCard.vue'
import WorkspaceHeader from './MainContent/WorkspaceHeader.vue'
import WorkspaceFloatingTabs from './MainContent/WorkspaceFloatingTabs.vue'
import LecturePreviewPanel from './MainContent/LecturePreviewPanel.vue'
import WorkspaceQuizPanel from './Quiz/WorkspaceQuizPanel.vue'
import WorkspaceSummaryNotesPanel from './Summary/WorkspaceSummaryNotesPanel.vue'
import WorkspaceSummaryPanel from './Summary/WorkspaceSummaryPanel.vue'

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
  currentRecordings: { type: Array, default: () => [] },
  transcriptions: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  materialEvidenceRequest: { type: Object, default: null },
  summaryState: { type: Object, default: () => ({}) },
  summaryNotes: { type: Array, default: () => [] },
  quizSource: { type: Object, default: null }
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
  'closePreviewMaterial'
])

const activeTab = ref('note')
const activeSummaryTab = ref('summary')
const noteContent = ref('')
const isNoteFocused = ref(false)
const tabAnim = ref('tab-slide-right')
const isNoteDragOver = ref(false)
const showDiarizationChoice = ref(false)
let prevTab = 'note'

const TAB_ORDER = ['note', 'summary-note', 'summary', 'quiz']

const tabs = computed(() => [
  { key: 'note', label: '메모' },
  { key: 'summary-note', label: '정리' },
  { key: 'summary', label: '요약' },
  { key: 'quiz', label: '퀴즈' }
])

const noteTitle = computed(() => props.activeFileName || '파일을 선택하세요')

const allowedMaterialTypes = [
  'application/pdf',
  'application/vnd.ms-powerpoint',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation'
]

const getDefaultTabByFileType = () => 'note'

const handleTabChange = (newTab) => {
  const prevIdx = TAB_ORDER.indexOf(prevTab)
  const nextIdx = TAB_ORDER.indexOf(newTab)
  tabAnim.value = nextIdx > prevIdx ? 'tab-slide-right' : 'tab-slide-left'
  prevTab = newTab
  activeTab.value = newTab
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
  },
  { immediate: true }
)

watch(
  () => props.currentPreviewMaterial,
  (nextMaterial, prevMaterial) => {
    if (!nextMaterial || nextMaterial.id === prevMaterial?.id) return
    handleTabChange('note')
  }
)

watch(
  () => props.materialEvidenceRequest,
  (request) => {
    if (!request) return
    handleTabChange('note')
  }
)

const isLectureMaterialFile = (file) => {
  if (!file) return false
  return allowedMaterialTypes.includes(file.type) || /\.(pdf|ppt|pptx)$/i.test(file.name)
}

const openMaterialInMemo = (file) => {
  emit('uploadLectureMaterials', [file])
  activeTab.value = 'note'
}

const handleMaterialSelection = (file) => {
  if (isLectureMaterialFile(file)) {
    openMaterialInMemo(file)
  }
}

const handleDroppedMaterial = (event) => {
  event.preventDefault()
  isNoteDragOver.value = false
  const file = Array.from(event.dataTransfer?.files || []).find(isLectureMaterialFile)
  if (file) {
    openMaterialInMemo(file)
  }
}

const onNoteBlur = (e) => {
  isNoteFocused.value = false
  noteContent.value = e.target.innerText
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
  <main class="flex-1 flex flex-col gap-[12px] h-full min-w-0" style="flex: 1 1 0%; min-width: 300px;">
    <transition name="word-card">
      <WorkspaceWordCard
        v-if="selectedWordData && isWordCardVisible"
        :word-data="selectedWordData"
        @close="hideSelectedWordCard"
        @ask-ai="handleAskAi"
        @add-to-note="handleAddToNote"
      />
    </transition>

    <div id="tab-contents-container" class="card workspace-shell-card flex-1 flex flex-col relative min-h-0 min-w-0 overflow-hidden">
      <WorkspaceHeader
        :is-recording="isRecording"
        :is-recording-paused="isRecordingPaused"
        :recording-time-text="recordingTimeText"
        :recording-audio-level="recordingAudioLevel"
        :show-close-preview="!!currentPreviewMaterial"
        :has-word-insight="!!selectedWordData"
        :word-insight-visible="!!selectedWordData && isWordCardVisible"
        @start-recording="handleStartRecording"
        @pause-recording="emit('pauseRecording')"
        @resume-recording="emit('resumeRecording')"
        @stop-recording="emit('stopRecording')"
        @main-sidebar-toggle="emit('mainSidebarToggle')"
        @right-sidebar-toggle="emit('rightSidebarToggle')"
        @material-selected="handleMaterialSelection"
        @word-insight-click="handleWordInsightButtonClick"
        @close-preview-material="emit('closePreviewMaterial')"
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
          v-if="activeTab === 'note'"
          :key="'tab-note'"
          :class="[
            'tab-content flex-1 flex flex-col relative overflow-hidden note-canvas',
            currentPreviewMaterial ? 'px-4 pt-4 pb-0' : 'p-10 pt-4',
            tabAnim
          ]"
          @dragover.prevent="isNoteDragOver = true"
          @dragenter.prevent="isNoteDragOver = true"
          @dragleave.prevent="isNoteDragOver = false"
          @drop="handleDroppedMaterial"
        >
          <div :class="[currentPreviewMaterial ? 'w-full h-full flex flex-col' : 'max-w-4xl mx-auto w-full h-full']">
            <h1 v-if="!currentPreviewMaterial" class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">{{ noteTitle }}</h1>

            <div v-if="currentPreviewMaterial" class="preview-panel-wrap">
              <LecturePreviewPanel
                :material="currentPreviewMaterial"
                :evidence-request="materialEvidenceRequest"
              />
            </div>

            <div
              class="text-[16px] leading-relaxed min-h-[200px] focus:outline-none"
              :class="{ 'note-drop-target': isNoteDragOver }"
              id="note-body"
              contenteditable="true"
              v-show="!currentPreviewMaterial"
              :style="{ color: isNoteFocused || noteContent ? '#1d1d1f' : '#aeaeb2' }"
              @focus="isNoteFocused = true"
              @blur="onNoteBlur"
            >
              {{ (!noteContent && !isNoteFocused) ? '여기에 타이핑을 시작하거나 파일을 업로드하세요.' : noteContent }}
            </div>
          </div>
        </section>

        <WorkspaceSummaryNotesPanel
          v-else-if="activeTab === 'summary-note'"
          :key="'tab-summary-note'"
          :tab-anim="tabAnim"
          :summary-notes="summaryNotes"
        />

        <WorkspaceSummaryPanel
          v-else-if="activeTab === 'summary'"
          :key="'tab-summary'"
          v-model:active-summary-tab="activeSummaryTab"
          :tab-anim="tabAnim"
          :is-recording="isRecording"
          :is-recording-paused="isRecordingPaused"
          :recording-mode="recordingMode"
          :diarization-enabled="diarizationEnabled"
          :transcriptions="transcriptions"
          :summary-state="summaryState"
          :current-recordings="currentRecordings"
          :active-file-id="activeFileId"
          :current-preview-material="currentPreviewMaterial"
          :quiz-source="quizSource"
          @generateMaterialSummary="emit('generateMaterialSummary', $event)"
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

        <WorkspaceFloatingTabs :tabs="tabs" :active-tab="activeTab" @change="handleTabChange" />
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

.note-drop-target {
  border-radius: 24px;
  background: rgba(239, 246, 255, 0.5);
  outline: 1.5px dashed rgba(59, 130, 246, 0.42);
  outline-offset: 16px;
}

.preview-panel-wrap {
  flex: 1;
  min-height: 0;
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
