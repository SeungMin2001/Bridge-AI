<script setup>
import { ref, computed, watch } from 'vue'
import AnimatedTabs from '../ui/AnimatedTabs.vue'

const props = defineProps({
  isRecording: Boolean,
  recordingTimeText: String,
  activeFileName: String,
  summaryNotes: { type: Array, default: () => [] }
})

const emit = defineEmits(['startRecording', 'stopRecording', 'mainSidebarToggle', 'rightSidebarToggle'])

const activeTab = ref('note')
const activeSummaryTab = ref('ai-summary')
const noteContent = ref('')
const isNoteFocused = ref(false)
const tabAnim = ref('tab-slide-right')
let prevTab = 'note'

const TAB_ORDER = ['note', 'summary-note', 'material', 'summary', 'quiz']

const tabs = computed(() => [
  { key: 'note', label: noteTabName.value },
  { key: 'summary-note', label: '정리 노트' },
  { key: 'material', label: '자료' },
  { key: 'summary', label: '요약' },
  { key: 'quiz', label: '퀴즈' }
])

const handleTabChange = (newTab) => {
  const prevIdx = TAB_ORDER.indexOf(prevTab)
  const nextIdx = TAB_ORDER.indexOf(newTab)
  tabAnim.value = nextIdx > prevIdx ? 'tab-slide-right' : 'tab-slide-left'
  prevTab = newTab
  activeTab.value = newTab
}

// v-model 연동을 위한 watch
watch(activeTab, (newVal) => {
  if (newVal !== prevTab) {
    handleTabChange(newVal)
  }
})

const noteTabName = computed(() => props.activeFileName || '새 노트')

const onNoteBlur = (e) => {
  isNoteFocused.value = false
  noteContent.value = e.target.innerText
}
</script>

<template>
  <main class="flex-1 flex flex-col gap-[12px] h-full min-w-0" style="flex: 1 1 0%; min-width: 300px;">
    <!-- Header Card -->
    <header class="card h-[56px] flex items-center px-5 shrink-0">
      <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] mr-4 shrink-0" title="사이드바 토글" @click="emit('mainSidebarToggle')">
        <span class="material-symbols-outlined text-[20px]">side_navigation</span>
      </button>

      <!-- Tab Navigation -->
      <nav class="flex h-full py-2 items-center" id="main-tabs">
        <AnimatedTabs v-model="activeTab" :tabs="tabs" />
      </nav>

      <div class="ml-auto flex items-center gap-1.5 shrink-0 pl-2">
        <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]">
          <span class="material-symbols-outlined text-[20px]">play_circle</span>
        </button>

        <button v-if="!isRecording" class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]" id="start" @click="emit('startRecording')">
          <span class="material-symbols-outlined text-[20px]">mic</span>
        </button>
        <div
          v-else
          id="recording-timer"
          class="flex items-center gap-2 bg-[#FFF0F3] hover:bg-[#FFE4E9] px-3 py-1.5 rounded-full cursor-pointer transition-colors border border-[#FFD1DA] shrink-0"
          @click="emit('stopRecording')"
        >
          <div class="recording-wave-container w-6 h-6 shrink-0">
            <div class="recording-wave-ring"></div>
            <div class="recording-wave-ring"></div>
            <div class="recording-wave-ring"></div>
            <span class="live-dot" style="position: relative; z-index: 1;"></span>
          </div>
          <span class="text-[13px] font-bold text-[#1d1d1f] tabular-nums" id="recording-time">{{ recordingTimeText }}</span>
        </div>

        <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]" title="우측 사이드바 토글" @click="emit('rightSidebarToggle')">
          <span class="material-symbols-outlined text-[20px] scale-x-[-1]">side_navigation</span>
        </button>
      </div>
    </header>

    <!-- Main Content Area -->
    <div id="tab-contents-container" class="flex-1 flex flex-col relative min-h-0 min-w-0">
      <!-- Note Tab -->
      <section v-if="activeTab === 'note'" :key="'tab-note'" :class="['tab-content card flex-1 flex flex-col relative overflow-hidden note-canvas p-10 pt-12', tabAnim]">
        <div class="max-w-4xl mx-auto w-full h-full">
          <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">{{ noteTabName }}</h1>
          <div
            class="text-[16px] leading-relaxed min-h-[200px] focus:outline-none"
            id="note-body"
            contenteditable="true"
            :style="{ color: isNoteFocused || noteContent ? '#1d1d1f' : '#aeaeb2' }"
            @focus="isNoteFocused = true"
            @blur="onNoteBlur"
          >
            {{ (!noteContent && !isNoteFocused) ? '여기에 타이핑을 시작하거나 파일을 업로드하세요.' : noteContent }}
          </div>
        </div>
        <div class="floating-toolbar absolute bottom-8 left-1/2 -translate-x-1/2 flex p-1.5 gap-1 z-10 bg-white">
          <button class="tool-btn-active w-[48px] h-[48px] flex items-center justify-center rounded-full"><span class="material-symbols-outlined text-[24px]">near_me</span></button>
          <button class="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors"><span class="material-symbols-outlined text-[24px]">ink_pen</span></button>
          <button class="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors"><span class="material-symbols-outlined text-[24px]">history_edu</span></button>
          <button class="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors"><span class="material-symbols-outlined text-[24px]">add_circle</span></button>
        </div>
      </section>

      <!-- Summary Note Tab -->
      <section v-else-if="activeTab === 'summary-note'" :key="'tab-summary-note'" :class="['tab-content card flex-1 flex flex-col relative overflow-hidden p-10 pt-12', tabAnim]">
        <div class="max-w-4xl mx-auto w-full h-full overflow-y-auto custom-scrollbar">
          <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">정리 노트</h1>
          <div class="flex flex-col gap-4">
            <div v-if="summaryNotes.length === 0" class="text-[16px] text-[#aeaeb2] leading-relaxed italic">아직 추가된 내용이 없습니다. 전사 내용에서 '노트에 추가'를 눌러보세요.</div>
            <div v-else v-for="note in summaryNotes" :key="note.id" class="p-5 rounded-2xl bg-[#fbfbfd] border border-gray-100 shadow-sm flex flex-col gap-2 transcription-item-enter">
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

      <!-- Material Tab -->
      <section v-else-if="activeTab === 'material'" :key="'tab-material'" :class="['tab-content card flex-1 flex flex-col relative overflow-hidden p-10 pt-12', tabAnim]">
        <div class="max-w-4xl mx-auto w-full h-full">
          <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">자료</h1>
          <div class="text-[16px] text-[#aeaeb2] leading-relaxed">학습 자료 및 관련 문서가 여기에 표시됩니다.</div>
        </div>
      </section>

      <!-- Summary Tab -->
      <section v-else-if="activeTab === 'summary'" :key="'tab-summary'" :class="['tab-content card flex-1 flex flex-col relative overflow-hidden note-canvas p-10 overflow-y-auto custom-scrollbar pt-[32px]', tabAnim]">
        <div class="max-w-4xl mx-auto w-full">
          <div class="flex items-center justify-between border-b border-[#e5e5ea] mb-8 pb-0">
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

      <!-- Quiz Tab -->
      <section v-else-if="activeTab === 'quiz'" :key="'tab-quiz'" :class="['tab-content card flex-1 flex flex-col relative overflow-hidden p-10 pt-12', tabAnim]">
        <div class="max-w-4xl mx-auto w-full h-full">
          <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">퀴즈</h1>
          <div class="text-[16px] text-[#aeaeb2] leading-relaxed">생성된 퀴즈와 테스트가 여기에 표시됩니다.</div>
        </div>
      </section>
    </div>
  </main>
</template>
