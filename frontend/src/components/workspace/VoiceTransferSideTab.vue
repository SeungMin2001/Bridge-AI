<!-- 실시간 음성 전사 결과를 확인하고 AI에게 질문하거나 노트에 추가하는 사이드 탭 컴포넌트입니다. -->
<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import { useChat } from '../../composables/useChat'

const { selectWord } = useChat()

const props = defineProps({
  transcriptions: { type: Array, default: () => [] },
  recordingMode: { type: String, default: 'lecture' },
  variant: { type: String, default: 'sidebar' }
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

const shouldShowSpeaker = (transcription) => props.recordingMode === 'meeting' || !!transcription.speaker

const getSpeakerLabel = (transcription) => {
  if (shouldShowSpeaker(transcription)) return transcription.speaker || '화자 미상'
  return '나'
}

const getSpeakerAvatarSrc = (transcription) => {
  const label = getSpeakerLabel(transcription)
  if (label === '화자 B' || label === '화자 2') return '/images/man1.png'
  return '/images/woman1.png'
}

const getSpeakerAccent = (transcription) => {
  const label = getSpeakerLabel(transcription)
  if (label === '나') return 'blue'
  if (label === '화자 B' || label === '화자 2') return 'rose'
  if (label === '화자 C' || label === '화자 3') return 'green'
  return 'amber'
}

const getSpeakerAvatarClass = (transcription) => `speaker-avatar-${getSpeakerAccent(transcription)}`
</script>

<template>
  <div class="flex flex-col flex-1 overflow-hidden" :class="{ 'transcript-panel-content': variant === 'content' }">
    <!-- 검색 창 -->
    <div class="sidebar-search-bg workspace-inset-shell transcript-search-shell rounded-[20px] px-3 py-2 flex items-center gap-2.5 mb-4">
      <span class="material-symbols-outlined text-[#8e8e93] text-[19px]">search</span>
      <input
        class="bg-transparent border-none focus:ring-0 p-0 text-[13px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
        placeholder="전사 내용 검색"
        type="text"
        v-model="transSearch"
      />
    </div>

    <!-- 전사 기록 리스트 -->
    <div 
      ref="scrollContainer"
      class="transcript-list flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-4 pb-4"
    >
      <template v-if="transcriptions.filter(t => t.text.toLowerCase().includes(transSearch.toLowerCase())).length === 0">
        <div class="empty-transcript-state flex flex-col items-center justify-center h-full py-10">
          <img
            class="empty-transcript-image"
            src="/images/novoice.png"
            alt=""
            aria-hidden="true"
          />
          <p class="text-[13px] font-medium text-[#8e8e93]">
            {{ recordingMode === 'meeting' ? '화자 분리된 회의 스크립트가 여기에 표시됩니다.' : '전사된 데이터가 없습니다.' }}
          </p>
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
            <div class="speaker-avatar" :class="getSpeakerAvatarClass(t)">
              <img :src="getSpeakerAvatarSrc(t)" :alt="getSpeakerLabel(t)" />
            </div>
            <span class="text-[11px] font-bold text-[#1d1d1f]">{{ getSpeakerLabel(t) }}</span>
          </div>
          <div class="message-bubble voice-message-bubble px-3.5 py-3 text-[15px] leading-[1.6]" :class="{ 'is-content': variant === 'content', 'is-meeting': shouldShowSpeaker(t) }">
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

.voice-message-bubble .clickable-word:hover {
  background-color: rgba(191, 165, 128, 0.46);
  color: #1d1d1f;
}

.voice-message-bubble .clickable-word:active {
  background-color: rgba(148, 130, 106, 0.42);
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

.voice-message-bubble {
  position: relative;
  width: fit-content;
  max-width: min(calc(100% - 18px), 440px);
  background: #f4ede4;
  border: 1px solid rgba(255, 255, 255, 0.82);
  box-shadow: none;
  overflow: hidden;
}

.transcript-list {
  padding-right: 14px;
}

.empty-transcript-state {
  gap: 12px;
  color: #8e8e93;
  text-align: center;
}

.empty-transcript-image {
  width: min(72%, 178px);
  height: auto;
  opacity: 0.5;
  filter: grayscale(1);
  user-select: none;
  pointer-events: none;
}

.speaker-avatar {
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  padding: 3px;
  overflow: hidden;
  border-radius: 999px;
  border: 1.5px solid #1d1d1f;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.85),
    0 1px 0 rgba(255, 255, 255, 0.9);
}

.speaker-avatar img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center bottom;
  border-radius: 999px;
  display: block;
}

.speaker-avatar-blue {
  background: #dbeafe;
}

.speaker-avatar-amber {
  background: #fff1d6;
}

.speaker-avatar-rose {
  background: #ffe4ea;
}

.speaker-avatar-green {
  background: #dcfce7;
}

.transcript-panel-content .voice-message-bubble.is-content {
  max-width: min(100%, 860px);
}

.transcript-panel-content .transcript-list {
  padding-right: 0;
}

.transcript-panel-content .empty-transcript-image {
  width: min(42%, 260px);
  opacity: 0.46;
}

.voice-message-bubble.is-meeting {
  background: #f8f4ee;
  border-color: rgba(222, 205, 182, 0.72);
}

.transcript-search-shell {
  position: relative;
  background: #f4ede4;
  border: 1px solid rgba(255, 255, 255, 0.82);
  box-shadow: none;
  overflow: hidden;
}

.transcript-search-shell::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: radial-gradient(circle at top left, rgba(255, 255, 255, 0.58), transparent 42%);
  pointer-events: none;
}

.voice-message-bubble::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background:
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.18), transparent 34%);
  pointer-events: none;
}

</style>
