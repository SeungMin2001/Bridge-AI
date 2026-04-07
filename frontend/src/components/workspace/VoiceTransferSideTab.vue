<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import { useChat } from '../../composables/useChat'

const { selectWord } = useChat()

const props = defineProps({
  transcriptions: { type: Array, default: () => [] }
})

const emit = defineEmits(['addToNote', 'askAi'])

const transSearch = ref('')
const scrollContainer = ref(null)

// 최하단으로 스크롤 이동
const scrollToBottom = async () => {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTo({
      top: scrollContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}

// 전사 데이터가 변경될 때마다 스크롤 이동
watch(() => props.transcriptions, () => {
  scrollToBottom()
}, { deep: true })

onMounted(() => {
  scrollToBottom()
})

// 단어 클릭 → 전역 상태로 전달하여 메인 컨텐츠 영역에 카드로 표시
const handleWordClick = (e, word) => {
  e.stopPropagation()
  selectWord(word)
}
</script>

<template>
  <div class="flex flex-col flex-1 overflow-hidden">
    <!-- 검색 창 -->
    <div class="sidebar-search-bg workspace-inset-shell rounded-[24px] px-3 py-3 flex items-center gap-3 mb-6">
      <span class="material-symbols-outlined text-[#8e8e93] text-[20px]">search</span>
      <input
        class="bg-transparent border-none focus:ring-0 p-0 text-[14px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
        placeholder="전사 내용 검색"
        type="text"
        v-model="transSearch"
      />
    </div>

    <!-- 전사 기록 리스트 -->
    <div 
      ref="scrollContainer"
      class="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-4 pb-4"
    >
      <template v-if="transcriptions.filter(t => t.text.toLowerCase().includes(transSearch.toLowerCase())).length === 0">
        <div class="flex flex-col items-center justify-center h-full opacity-40 py-10">
          <span class="material-symbols-outlined text-[48px] mb-2 text-[#aeaeb2]">record_voice_over</span>
          <p class="text-[13px] font-medium text-[#8e8e93]">전사된 데이터가 없습니다.</p>
        </div>
      </template>
      <template v-else>
        <div 
          v-for="(t, idx) in transcriptions.filter(tr => tr.text.toLowerCase().includes(transSearch.toLowerCase()))" 
          :key="idx" 
          class="flex flex-col gap-1.5 mt-2 transcription-item-enter"
          :style="{ animationDelay: `${idx * 0.06}s` }"
        >
          <span class="text-[11px] font-bold text-[#aeaeb2] px-1.5">{{ t.time }}</span>
          <div class="flex items-center gap-2 px-1.5 mb-1">
            <div class="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
              <span class="text-[10px] font-bold text-blue-600">나</span>
            </div>
            <span class="text-[11px] font-bold text-[#1d1d1f]">나</span>
          </div>
          <div class="message-bubble px-3.5 py-3 text-[13px] leading-[1.6]">
            <template v-if="t.segments && t.segments.length">
              <span
                v-for="(seg, sIdx) in t.segments"
                :key="seg.id ?? sIdx"
                class="segment-wrap"
                :class="{ 'segment-pending': seg.status === 'pending', 'segment-confirmed': seg.status === 'confirmed' }"
              >
                <span
                  v-for="(word, wIdx) in seg.text.split(' ')"
                  :key="wIdx"
                  class="clickable-word"
                  @click="(e) => handleWordClick(e, word)"
                >{{ word }}&nbsp;</span>
              </span>
            </template>
            <template v-else>
              <span
                v-for="(word, wIdx) in t.text.split(' ')"
                :key="wIdx"
                class="clickable-word"
                @click="(e) => handleWordClick(e, word)"
              >{{ word }}&nbsp;</span>
            </template>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.popover-enter-active,
.popover-leave-active {
  transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.popover-enter-from,
.popover-leave-to {
  opacity: 0;
  transform: translateY(10px) scale(0.95);
}
.popover-enter-to,
.popover-leave-from {
  opacity: 1;
  transform: translateY(0) scale(1);
}

/* segment-wrap 인라인 표시 */
.segment-wrap {
  display: inline;
}

/* 전사 텍스트 상태 애니메이션 */
.segment-pending {
  opacity: 0.55;
  filter: blur(0.3px);
  transition: opacity 0.5s ease, filter 0.5s ease;
}
.segment-pending .clickable-word {
  color: #aeaeb2;
  transition: color 0.5s ease;
}
.segment-confirmed {
  opacity: 1;
  filter: none;
  animation: confirmSegment 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.segment-confirmed .clickable-word {
  color: #1d1d1f;
  animation: confirmWord 0.5s ease forwards;
}
@keyframes confirmSegment {
  0%   { opacity: 0.55; transform: translateY(2px); }
  60%  { opacity: 1;    transform: translateY(-1px); }
  100% { opacity: 1;    transform: translateY(0); }
}
@keyframes confirmWord {
  0%   { color: #aeaeb2; }
  40%  { color: #3b82f6; }
  100% { color: #1d1d1f; }
}
</style>
