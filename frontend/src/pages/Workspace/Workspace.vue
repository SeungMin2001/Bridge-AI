<script setup>
import { ref } from 'vue'
import LeftSidebar from '../../components/workspace/LeftSidebar.vue'
import MainContent from '../../components/workspace/MainContent.vue'
import RightSidebar from '../../components/workspace/RightSidebar.vue'
import { useChat } from '../../composables/useChat'

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  isRecording: { type: Boolean, default: false },
  recordingTimeText: { type: String, default: '0:00' },
  activeFileName: { type: String, default: '' },
  isRightSidebarVisible: { type: Boolean, default: true },
  summaryNotes: { type: Array, default: () => [] },
  aiInput: { type: String, default: '' }
})

const emit = defineEmits([
  'navigateHome',
  'fileSelect',
  'update:fileTree',
  'update:favorites',
  'update:aiInput',
  'startRecording',
  'stopRecording',
  'rightSidebarToggle',
  'addToNote',
  'askAi'
])

const isLeftSidebarCollapsed = ref(false)
const { showCitePopover, currentCite, citePopoverPos, closeCitePopover } = useChat()

// 팝오버 내 버튼 액션
function askAboutCite(cite) {
  if (!cite) return
  closeCitePopover()
  if (!props.isRightSidebarVisible) {
    emit('rightSidebarToggle')
  }
  const shortened = cite.text.length > 15 ? cite.text.slice(0, 15) + '...' : cite.text
  emit('update:aiInput', `"${shortened}"에 대해 더 자세히 알려줘 `)
}

function noteAddDummy() {
  alert('노트에 추가되었습니다. (데모)')
  closeCitePopover()
}

import { computed } from 'vue'

const highlightedTranscript = computed(() => {
  const cite = currentCite.value
  if (!cite) return ''

  // 전체 전사가 있으면 그것을 쓰고, 없으면 기존 text 사용
  const fullText = cite.full_transcript || cite.text
  // 하이라이팅 대상
  const target = cite.text

  if (cite.full_transcript && fullText.includes(target)) {
    // 찾은 문장을 <mark> 태그로 감싸서 리턴
    return fullText.replace(
      target, 
      `<mark class="bg-[#eff6ff] text-[#1d1d1f] font-bold rounded-[4px] px-1 -mx-1" style="box-decoration-break: clone;">${target}</mark>`
    )
  }
  return fullText
})
</script>

<template>
  <div 
    class="p-[12px] flex relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden transition-all duration-400"
    :class="[
      { 'gap-[12px]': !isLeftSidebarCollapsed || isRightSidebarVisible }
    ]"
  >
    <LeftSidebar
      :isCollapsed="isLeftSidebarCollapsed"
      :transcriptions="transcriptions"
      :fileTree="fileTree"
      :favorites="favorites"
      @toggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
      @navigateHome="emit('navigateHome')"
      @fileSelect="(id, node) => emit('fileSelect', id, node)"
      @update:fileTree="emit('update:fileTree', $event)"
      @update:favorites="emit('update:favorites', $event)"
      @addToNote="(text, source) => emit('addToNote', text, source)"
      @askAi="(word) => emit('askAi', word)"
    />
    
    <MainContent
      :isRecording="isRecording"
      :recordingTimeText="recordingTimeText"
      :activeFileName="activeFileName"
      :summaryNotes="summaryNotes"
      @startRecording="emit('startRecording')"
      @stopRecording="emit('stopRecording')"
      @mainSidebarToggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
      @rightSidebarToggle="emit('rightSidebarToggle')"
      @askAi="(word) => emit('askAi', word)"
      @addToNote="(text, source) => emit('addToNote', text, source)"
    />
    
    <RightSidebar 
      :visible="isRightSidebarVisible" 
      :aiInput="aiInput"
      @update:aiInput="emit('update:aiInput', $event)"
    />
  </div>

  <!-- ═══ 부유형 팝오버 (Workspace 수준 관리) ═══ -->
  <Teleport to="body">
    <transition name="popover-fade">
      <div v-if="showCitePopover" class="cite-popover-overlay" @click.self="closeCitePopover">
        <div 
          class="cite-popover"
          :style="{ left: citePopoverPos.x + 'px', top: citePopoverPos.y + 'px' }"
        >
          <!-- 헤더 -->
          <div class="flex items-center justify-between mb-5 px-1">
            <div class="flex items-center gap-3">
              <div class="w-2.5 h-2.5 bg-[#3b82f6] rounded-full animate-pulse shadow-[0_0_10px_rgba(59,130,246,0.5)]"></div>
              <span class="font-bold text-[#1c1c1e] text-[17px] tracking-tight">근거 정보</span>
            </div>
            <button class="text-[#8e8e93] hover:text-[#1c1c1e] transition-all bg-transparent border-none p-1.5 cursor-pointer flex items-center justify-center rounded-full hover:bg-black/5" @click="closeCitePopover">
              <span class="material-symbols-outlined text-[20px]">close</span>
            </button>
          </div>

          <!-- 본문 (스크롤 영역) -->
          <div class="flex-1 overflow-y-auto mb-6 px-1 custom-scrollbar" style="max-height: 400px;">
            <div 
              class="text-[15px] text-[#3a3a3c] leading-[1.8] whitespace-pre-wrap break-keep font-medium"
              v-html="highlightedTranscript"
            >
            </div>
          </div>

          <!-- 구분선 -->
          <div class="w-full h-[1px] bg-black/5 mb-4 shrink-0"></div>

          <!-- 출처 정보 -->
          <div class="flex flex-col gap-1.5 pt-1 shrink-0">
            <div class="flex items-center gap-1.5">
              <span class="material-symbols-outlined text-[14px] text-[#8e8e93]">link</span>
              <span class="font-bold text-[#8e8e93] text-[11px] uppercase tracking-wider">SOURCE:</span>
              <span class="font-bold text-[#3b82f6] text-[11px] ml-1 truncate hover:underline cursor-pointer">
                {{ currentCite?.session_title || 'AI 분석 결과' }}
              </span>
            </div>
            
            <div v-if="currentCite?.transcript_id" class="flex items-center gap-1.5 ml-[20px]">
              <span class="text-[#8e8e93] text-[9px] font-medium tracking-wide uppercase">REF_ID:</span>
              <span class="text-[#aeaeb2] text-[9px] font-mono select-all">{{ currentCite.transcript_id }}</span>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<style scoped>
.cite-popover-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: transparent;
}

.cite-popover {
  position: fixed;
  /* x, y 좌표는 citePopoverPos에서 인라인 스타일로 제어됨 */
  width: 290px; /* 320px에서 30px 줄임 */
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.6);
  display: flex;
  flex-direction: column;
  padding: 20px;
  transform-origin: right top; /* 우측 정렬이므로 우상단 기준으로 애니메이션 */
}

/* 애니메이션 개선 */
.popover-fade-enter-active {
  transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.popover-fade-leave-active {
  transition: all 0.15s ease;
}
.popover-fade-enter-from {
  opacity: 0;
  transform: scale(0.9) translateY(10px);
}
.popover-fade-leave-to {
  opacity: 0;
  transform: scale(0.95) translateY(5px);
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
