<!-- 워크스페이스의 중앙 영역으로, 강의 자료 뷰어, 녹음 조작, 노점 요약 내용을 표시합니다. -->
<script setup>
import { ref, computed, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import WorkspaceWordCard from './MainContent/WorkspaceWordCard.vue'
import WorkspaceHeader from './MainContent/WorkspaceHeader.vue'
import WorkspaceFloatingTabs from './MainContent/WorkspaceFloatingTabs.vue'
import LectureMaterialList from './MainContent/LectureMaterialList.vue'
import LecturePreviewPanel from './MainContent/LecturePreviewPanel.vue'

const { selectedWordData, clearSelectedWord } = useChat()

const props = defineProps({
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingTimeText: String,
  activeFileName: String,
  activeFileId: String,
  materialAttachments: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  summaryNotes: { type: Array, default: () => [] }
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
  'closePreviewMaterial',
  'openStoredMaterial',
  'deleteStoredMaterial'
])

const activeTab = ref('note')
const activeSummaryTab = ref('ai-summary')
const noteContent = ref('')
const isNoteFocused = ref(false)
const tabAnim = ref('tab-slide-right')
const isNoteDragOver = ref(false)
let prevTab = 'note'

const TAB_ORDER = ['note', 'summary-note', 'material', 'summary', 'quiz']

const tabs = computed(() => [
  { key: 'note', label: '메모' },
  { key: 'summary-note', label: '정리' },
  { key: 'material', label: '자료' },
  { key: 'summary', label: '요약' },
  { key: 'quiz', label: '퀴즈' }
])

const noteTitle = computed(() => props.activeFileName || '강의1')

const allowedMaterialTypes = [
  'application/pdf',
  'application/vnd.ms-powerpoint',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation'
]

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

const handleOpenStoredMaterial = (fileId) => {
  emit('openStoredMaterial', fileId)
  handleTabChange('note')
}

const handleDeleteStoredMaterial = (fileId) => {
  emit('deleteStoredMaterial', fileId)
}
</script>

<template>
  <main class="flex-1 flex flex-col gap-[12px] h-full min-w-0" style="flex: 1 1 0%; min-width: 300px;">
    <transition name="word-card">
      <WorkspaceWordCard
        v-if="selectedWordData"
        :word-data="selectedWordData"
        @close="clearSelectedWord"
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
        @start-recording="emit('startRecording')"
        @pause-recording="emit('pauseRecording')"
        @resume-recording="emit('resumeRecording')"
        @stop-recording="emit('stopRecording')"
        @main-sidebar-toggle="emit('mainSidebarToggle')"
        @right-sidebar-toggle="emit('rightSidebarToggle')"
        @material-selected="handleMaterialSelection"
        @close-preview-material="emit('closePreviewMaterial')"
      />

      <div class="flex-1 flex flex-col relative min-h-0 min-w-0">
        <section
          v-if="activeTab === 'note'"
          :key="'tab-note'"
          :class="[
            'tab-content flex-1 flex flex-col relative overflow-hidden note-canvas',
            currentPreviewMaterial ? 'p-6 pt-4' : 'p-10 pt-4',
            tabAnim
          ]"
          @dragover.prevent="isNoteDragOver = true"
          @dragenter.prevent="isNoteDragOver = true"
          @dragleave.prevent="isNoteDragOver = false"
          @drop="handleDroppedMaterial"
        >
          <div :class="[currentPreviewMaterial ? 'w-full h-full flex flex-col' : 'max-w-4xl mx-auto w-full h-full']">
            <h1 v-if="!currentPreviewMaterial" class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">{{ noteTitle }}</h1>

            <div v-if="activeFileId === 'lecture-1' && currentPreviewMaterial" class="preview-panel-wrap">
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

        <section v-else-if="activeTab === 'summary-note'" :key="'tab-summary-note'" :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full h-full overflow-y-auto custom-scrollbar">
            <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">정리 노트</h1>
            <div class="flex flex-col gap-4">
              <div v-if="summaryNotes.length === 0" class="text-[16px] text-[#aeaeb2] leading-relaxed italic">아직 추가된 내용이 없습니다. 전사 내용에서 '노트에 추가'를 눌러보세요.</div>
              <div v-else v-for="note in summaryNotes" :key="note.id" class="workspace-subpanel p-5 rounded-[24px] flex flex-col gap-2 transcription-item-enter">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-[18px] text-blue-500">auto_stories</span>
                    <span class="text-[13px] font-bold text-[#1d1d1f]">추가된 내용</span>
                  </div>
                  <span class="text-[11px] font-medium text-[#aeaeb2]">{{ note.time }}</span>
                </div>
                <p class="text-[15px] leading-[1.6] text-[#3a3a3c] font-medium">{{ note.text }}</p>
                <div class="flex items-center gap-1.5 mt-1 border-t border-black/5 pt-3">
                  <span class="material-symbols-outlined text-[14px] text-[#8e8e93]">link</span>
                  <span class="text-[11px] font-bold text-[#8e8e93] uppercase tracking-wider">Source:</span>
                  <span class="text-[11px] font-bold text-blue-500 cursor-pointer hover:underline decoration-blue-500/50 underline-offset-2">{{ note.source || 'AI 분석 결과' }}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section v-else-if="activeTab === 'material'" :key="'tab-material'" :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full h-full overflow-y-auto custom-scrollbar">
            <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">자료</h1>
            <LectureMaterialList
              :material-attachments="materialAttachments"
              @open="handleOpenStoredMaterial"
              @delete="handleDeleteStoredMaterial"
            />
          </div>
        </section>

        <section v-else-if="activeTab === 'summary'" :key="'tab-summary'" :class="['tab-content flex-1 flex flex-col relative overflow-hidden note-canvas p-10 overflow-y-auto custom-scrollbar pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full">
            <div class="flex items-center justify-between border-b border-[#e5e5ea] mb-5 pb-0">
              <nav class="flex gap-8">
                <div class="relative cursor-pointer summary-subtab-btn group" @click="activeSummaryTab = 'ai-summary'">
                  <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'ai-summary' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">AI 요약&nbsp;&nbsp;</button>
                  <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'ai-summary' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
                </div>
                <div class="relative cursor-pointer summary-subtab-btn group" @click="activeSummaryTab = 'history'">
                  <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'history' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">대화기록&nbsp;&nbsp;</button>
                  <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'history' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
                </div>
              </nav>
            </div>
            <div v-show="activeSummaryTab === 'ai-summary'" class="summary-subcontent space-y-10"></div>
            <div v-show="activeSummaryTab === 'history'" class="summary-subcontent space-y-10"></div>
          </div>
        </section>

        <section v-else-if="activeTab === 'quiz'" :key="'tab-quiz'" :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full h-full">
            <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">퀴즈</h1>
            <div class="text-[16px] text-[#aeaeb2] leading-relaxed">생성된 퀴즈와 테스트가 여기에 표시됩니다.</div>
          </div>
        </section>

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
