<script setup>
import { ref } from 'vue'
import LeftSidebar from '../../components/workspace/LeftSidebar.vue'
import MainContent from '../../components/workspace/MainContent.vue'
import RightSidebar from '../../components/workspace/RightSidebar.vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import { useChat } from '../../composables/useChat'

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  isRecording: { type: Boolean, default: false },
  recordingTimeText: { type: String, default: '0:00' },
  activeFileName: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  currentAttachments: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
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
  'askAi',
  'uploadLectureMaterials',
  'closePreviewMaterial',
  'openStoredMaterial'
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
    class="p-[12px] flex relative h-full w-full bg-transparent text-[#1e293b] overflow-hidden transition-all duration-400"
    :class="[
      { 'gap-[12px]': !isLeftSidebarCollapsed || isRightSidebarVisible }
    ]"
  >
    <InfiniteGrid class="absolute inset-0 z-0" />
    <LeftSidebar
      class="relative z-10"
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
      class="relative z-10"
      :isRecording="isRecording"
      :recordingTimeText="recordingTimeText"
      :activeFileName="activeFileName"
      :activeFileId="activeFileId"
      :materialAttachments="currentAttachments"
      :currentPreviewMaterial="currentPreviewMaterial"
      :summaryNotes="summaryNotes"
      @startRecording="emit('startRecording')"
      @stopRecording="emit('stopRecording')"
      @mainSidebarToggle="isLeftSidebarCollapsed = !isLeftSidebarCollapsed"
      @rightSidebarToggle="emit('rightSidebarToggle')"
      @askAi="(word) => emit('askAi', word)"
      @addToNote="(text, source) => emit('addToNote', text, source)"
      @uploadLectureMaterials="emit('uploadLectureMaterials', $event)"
      @closePreviewMaterial="emit('closePreviewMaterial')"
      @openStoredMaterial="emit('openStoredMaterial', $event)"
    />
    
    <RightSidebar 
      class="relative z-10"
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
              <div class="cite-popover-badge">
                <span class="material-symbols-outlined text-[15px]">fact_check</span>
              </div>
              <span class="font-bold text-[#1c1c1e] text-[18px] tracking-tight">근거 정보</span>
            </div>
            <button class="cite-popover-close-btn" @click="closeCitePopover">
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
          <div class="cite-popover-divider"></div>

          <!-- 출처 정보 -->
          <div class="cite-source-wrap shrink-0">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-[15px] text-[#8e8e93]">link</span>
              <span class="font-bold text-[#8e8e93] text-[11px] uppercase tracking-wider">Source</span>
              <span class="font-bold text-[#4b5563] text-[12px] ml-1 truncate hover:underline cursor-pointer">
                {{ currentCite?.session_title || 'AI 분석 결과' }}
              </span>
            </div>
            
            <div v-if="currentCite?.transcript_id" class="flex items-center gap-1.5 ml-[23px] mt-1">
              <span class="text-[#8e8e93] text-[9px] font-medium tracking-wide uppercase">Ref ID</span>
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
  width: 284px;
  background: linear-gradient(160deg, rgba(246, 240, 232, 0.94), rgba(241, 233, 223, 0.72));
  border-radius: 24px;
  box-shadow: 0 24px 48px rgba(148, 163, 184, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(255, 255, 255, 0.82);
  display: flex;
  flex-direction: column;
  padding: 18px;
  transform-origin: right top;
  backdrop-filter: blur(22px) saturate(145%);
  -webkit-backdrop-filter: blur(22px) saturate(145%);
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

.cite-popover-badge {
  width: 30px;
  height: 30px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  background: linear-gradient(180deg, rgba(250,246,240,0.96), rgba(242,235,226,0.76));
  border: 1px solid rgba(255,255,255,0.84);
  box-shadow: 0 12px 24px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255,255,255,0.96);
}

.cite-popover-close-btn {
  color: #8e8e93;
  background: rgba(248,244,238,0.48);
  border: 1px solid rgba(255,255,255,0.72);
  padding: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  transition: all 0.2s ease;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
}

.cite-popover-close-btn:hover {
  background: rgba(250,246,240,0.78);
  color: #1c1c1e;
}

.cite-popover-divider {
  width: 100%;
  height: 1px;
  margin-bottom: 12px;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(206,212,218,0.7), rgba(255,255,255,0));
  flex-shrink: 0;
}

.cite-source-wrap {
  padding: 10px 12px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(249,244,238,0.78), rgba(241,233,224,0.56));
  border: 1px solid rgba(255,255,255,0.78);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.94);
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
