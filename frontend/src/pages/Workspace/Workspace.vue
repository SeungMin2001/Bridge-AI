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
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center gap-2.5">
              <div class="w-2.5 h-2.5 bg-[#3b82f6] rounded-full animate-pulse"></div>
              <span class="font-extrabold text-[#1d1d1f] text-[15px] tracking-tight">근거 정보</span>
            </div>
            <button class="text-[#8e8e93] hover:text-[#1d1d1f] transition-colors bg-transparent border-none p-1 cursor-pointer flex items-center justify-center rounded-full hover:bg-black/5" @click="closeCitePopover">
              <span class="material-symbols-outlined text-[18px]">close</span>
            </button>
          </div>

          <!-- 본문 -->
          <div class="text-[13px] text-[#3a3a3c] leading-[1.6] mb-5 whitespace-pre-wrap break-keep font-medium">
            {{ currentCite?.text || '내용이 없습니다.' }}
          </div>

          <!-- 구분선 -->
          <div class="w-full h-[1px] bg-black/5 mb-4"></div>

          <!-- 출처 정보 -->
          <div class="flex items-center gap-1.5 mb-5 pt-1">
            <span class="material-symbols-outlined text-[14px] text-[#8e8e93]">link</span>
            <span class="font-bold text-[#8e8e93] text-[11px] uppercase tracking-wider">SOURCE:</span>
            <span class="font-bold text-[#3b82f6] text-[11px] ml-1 truncate hover:underline cursor-pointer">
              {{ currentCite?.session_title || 'AI 분석 결과' }}
            </span>
          </div>

          <!-- 하단 버튼들 -->
          <div class="flex gap-2">
            <button class="flex-1 bg-[#3b82f6] hover:bg-blue-600 active:bg-blue-700 text-white font-bold text-[12px] py-2.5 rounded-xl transition-colors border-none cursor-pointer shadow-sm" @click="askAboutCite(currentCite)">
              AI에게 질문
            </button>
            <button class="flex-1 bg-[#f2f2f7] hover:bg-[#e5e5ea] active:bg-[#d1d1d6] text-[#1d1d1f] font-bold text-[12px] py-2.5 rounded-xl transition-colors border-none cursor-pointer" @click="noteAddDummy">
              노트에 추가
            </button>
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
  width: 280px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 20px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.4);
  display: flex;
  flex-direction: column;
  padding: 20px;
  transform-origin: left top;
}

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
</style>
