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
  activeFileName: String,
  activeFileId: String,
  activeFileType: { type: String, default: 'lecture' },
  transcriptions: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  summaryState: { type: Object, default: () => ({}) },
  summaryNotes: { type: Array, default: () => [] },
  quizSource: { type: Object, default: null }
})

const emit = defineEmits([
  'startRecording',
  'pauseRecording',
  'resumeRecording',
  'stopRecording',
  'mainSidebarToggle',
  'rightSidebarToggle',
  'askAi',
  'addToNote',
  'uploadLectureMaterials',
  'closePreviewMaterial'
])

const activeTab = ref('note')
const activeSummaryTab = ref('ai-summary')
const noteContent = ref('')
const isNoteFocused = ref(false)
const tabAnim = ref('tab-slide-right')
const isNoteDragOver = ref(false)
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

const handleStartRecording = () => {
  emit('startRecording', props.activeFileType === 'meeting' ? 'meeting' : 'lecture')
}
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
          :transcriptions="transcriptions"
          :summary-state="summaryState"
          @askAi="emit('askAi', $event)"
          @addToNote="(text, source) => emit('addToNote', text, source)"
        />

        <WorkspaceQuizPanel
          v-else-if="activeTab === 'quiz'"
          :key="'tab-quiz'"
          :tab-anim="tabAnim"
          :active-file-name="activeFileName"
          :active-file-id="activeFileId"
          :quiz-source="quizSource"
        />

        <WorkspaceFloatingTabs :tabs="tabs" :active-tab="activeTab" @change="handleTabChange" />
      </div>
    </div>
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
