<!-- 워크스페이스의 중앙 영역으로, 강의 자료 뷰어, 녹음 조작, 노점 요약 내용을 표시합니다. -->
<script setup>
import { ref, computed, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import WorkspaceWordCard from './MainContent/WorkspaceWordCard.vue'
import WorkspaceHeader from './MainContent/WorkspaceHeader.vue'
import WorkspaceFloatingTabs from './MainContent/WorkspaceFloatingTabs.vue'
import LectureMaterialList from './MainContent/LectureMaterialList.vue'
import LecturePreviewPanel from './MainContent/LecturePreviewPanel.vue'
import VoiceTransferSideTab from './VoiceTransferSideTab.vue'

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

const getDefaultTabByFileType = (fileType) => (fileType === 'meeting' ? 'summary' : 'note')

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

const handleOpenStoredMaterial = (fileId) => {
  emit('openStoredMaterial', fileId)
  handleTabChange('note')
}

const handleDeleteStoredMaterial = (fileId) => {
  emit('deleteStoredMaterial', fileId)
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

const speakerSummaryAccents = [
  { avatar: 'speaker-summary-avatar-blue', dot: 'speaker-summary-dot-blue' },
  { avatar: 'speaker-summary-avatar-amber', dot: 'speaker-summary-dot-amber' },
  { avatar: 'speaker-summary-avatar-rose', dot: 'speaker-summary-dot-rose' },
  { avatar: 'speaker-summary-avatar-green', dot: 'speaker-summary-dot-green' }
]

const getTranscriptText = (transcription) => {
  if (transcription?.segments?.length) {
    return transcription.segments
      .map((segment) => segment.text)
      .filter(Boolean)
      .join(' ')
      .trim()
  }
  return (transcription?.text || '').trim()
}

const getSpeakerSummaryKey = (transcription, index) => {
  if (transcription?.speakerId) return transcription.speakerId
  if (transcription?.speaker) return transcription.speaker
  return props.recordingMode === 'meeting' ? `unknown-speaker-${index}` : 'me'
}

const getSpeakerBadgeText = (speakerLabel) => {
  const label = String(speakerLabel || '화자')
    .replace(/^화자\s*/, '')
    .trim()
  return (label || '화').slice(0, 2)
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

    const speakerLabel = transcription.speaker || (props.recordingMode === 'meeting' ? '화자 미상' : '나')
    const speakerKey = getSpeakerSummaryKey(transcription, index)

    if (!speakerMap.has(speakerKey)) {
      speakerMap.set(speakerKey, {
        key: speakerKey,
        label: speakerLabel,
        firstIndex: index,
        utterances: []
      })
    }

    speakerMap.get(speakerKey).utterances.push({
      text,
      time: transcription.time || '',
      order: index
    })
  })

  return Array.from(speakerMap.values()).map((speaker, index) => {
    const latestUtterance = speaker.utterances[speaker.utterances.length - 1]
    const accent = speakerSummaryAccents[index % speakerSummaryAccents.length]

    return {
      ...speaker,
      accent,
      badgeText: getSpeakerBadgeText(speaker.label),
      utteranceCount: speaker.utterances.length,
      summary: buildMockSpeakerSummary(speaker.utterances),
      latestText: latestUtterance?.text || '',
      lastUpdatedAt: latestUtterance?.time || ''
    }
  })
})

const hasSpeakerSummaries = computed(() => speakerSummaryItems.value.length > 0)
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
            <h1 class="text-[32px] font-heavy-heading text-[#8e8e93] mb-5">정리 노트</h1>
            <div class="flex flex-col gap-4">
              <div v-if="summaryNotes.length === 0" class="text-[16px] text-[#aeaeb2] leading-relaxed italic">아직 추가된 내용이 없습니다. 전사 내용에서 '노트에 추가'를 눌러보세요.</div>
              <div v-else v-for="note in summaryNotes" :key="note.id" class="workspace-subpanel p-5 rounded-[24px] flex flex-col gap-2 transcription-item-enter">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-[18px] text-[#2563eb]">auto_stories</span>
                    <span class="text-[13px] font-bold text-[#1d1d1f]">추가된 내용</span>
                  </div>
                  <span class="text-[11px] font-bold text-[#6b7280]">{{ note.time }}</span>
                </div>
                <p class="text-[15px] leading-[1.6] text-[#1f2937] font-semibold">{{ note.text }}</p>
                <div class="flex items-center gap-1.5 mt-1 border-t border-slate-200 pt-3">
                  <span class="material-symbols-outlined text-[14px] text-[#64748b]">link</span>
                  <span class="text-[11px] font-bold text-[#64748b] uppercase tracking-wider">Source:</span>
                  <span class="text-[11px] font-bold text-[#2563eb] cursor-pointer hover:underline decoration-blue-500/50 underline-offset-2">{{ note.source || 'AI 분석 결과' }}</span>
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
          <div class="max-w-5xl mx-auto w-full h-full flex flex-col min-h-0">
            <div class="flex items-center justify-between border-b border-[#e5e5ea] mb-5 pb-0">
              <nav class="flex gap-8">
                <div class="relative cursor-pointer summary-subtab-btn group" @click="activeSummaryTab = 'transcript'">
                  <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'transcript' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">실시간 전사&nbsp;&nbsp;</button>
                  <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'transcript' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
                </div>
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
            <div v-show="activeSummaryTab === 'transcript'" class="summary-subcontent summary-transcript-wrap flex-1 min-h-0">
              <VoiceTransferSideTab
                :transcriptions="transcriptions"
                :recording-mode="recordingMode"
                variant="content"
                @askAi="emit('askAi', $event)"
                @addToNote="(text, source) => emit('addToNote', text, source)"
              />
            </div>
            <div v-show="activeSummaryTab === 'ai-summary'" class="summary-subcontent ai-summary-panel flex-1 min-h-0">
              <div v-if="!hasSpeakerSummaries" class="ai-summary-empty">
                <span class="material-symbols-outlined text-[42px] text-[#c7c7cc]">summarize</span>
                <p>아직 요약된 발화가 없습니다.</p>
              </div>

              <div v-else class="ai-summary-list">
                <article
                  v-for="speaker in speakerSummaryItems"
                  :key="speaker.key"
                  class="speaker-summary-card transcription-item-enter"
                  :style="{ animationDelay: `${speaker.firstIndex * 0.05}s` }"
                >
                  <div class="speaker-summary-top">
                    <div class="speaker-summary-identity">
                      <div class="speaker-summary-avatar" :class="speaker.accent.avatar">
                        {{ speaker.badgeText }}
                      </div>
                      <div class="min-w-0">
                        <h3>{{ speaker.label }}</h3>
                        <p>
                          발화 {{ speaker.utteranceCount }}개
                          <span v-if="speaker.lastUpdatedAt">· {{ speaker.lastUpdatedAt }}</span>
                        </p>
                      </div>
                    </div>

                    <div class="speaker-summary-status">
                      <span :class="['speaker-summary-dot', speaker.accent.dot]"></span>
                      <span>{{ isRecording && !isRecordingPaused ? '실시간' : '요약' }}</span>
                    </div>
                  </div>

                  <p class="speaker-summary-text">{{ speaker.summary }}</p>

                  <div class="speaker-summary-latest">
                    <span class="material-symbols-outlined">graphic_eq</span>
                    <span>{{ speaker.latestText }}</span>
                  </div>
                </article>
              </div>
            </div>
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

.summary-transcript-wrap {
  padding-bottom: 120px;
}

.ai-summary-panel {
  overflow-y: auto;
  padding: 2px 2px 120px;
}

.ai-summary-empty {
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #8e8e93;
  font-size: 14px;
  font-weight: 700;
}

.ai-summary-list {
  display: grid;
  gap: 14px;
}

.speaker-summary-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid rgba(229, 229, 234, 0.9);
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 16px 34px rgba(148, 163, 184, 0.08);
}

.speaker-summary-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.speaker-summary-identity {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.speaker-summary-avatar {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  font-size: 12px;
  font-weight: 900;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.26);
}

.speaker-summary-avatar-blue {
  background: linear-gradient(135deg, #60a5fa, #2563eb);
}

.speaker-summary-avatar-amber {
  background: linear-gradient(135deg, #f59e0b, #d97706);
}

.speaker-summary-avatar-rose {
  background: linear-gradient(135deg, #fb7185, #e11d48);
}

.speaker-summary-avatar-green {
  background: linear-gradient(135deg, #34d399, #059669);
}

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

.speaker-summary-identity p {
  margin: 4px 0 0;
  color: #8e8e93;
  font-size: 11px;
  font-weight: 800;
  line-height: 1.2;
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

.speaker-summary-text {
  margin: 0;
  color: #1f2937;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.7;
  word-break: keep-all;
  overflow-wrap: anywhere;
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
