<!-- 홈 화면에서 선택한 강의 자료나 AI 분석 결과를 상세하게 보여주는 오른쪽 사이드바입니다. -->
<script setup>
import { computed } from 'vue'

const props = defineProps({
  isOpen: {
    type: Boolean,
    default: false
  },
  referenceData: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['close', 'openFile'])

const title = computed(() => {
  const rawTitle = props.referenceData?.title || '스크립트'
  const parts = String(rawTitle)
    .split('>')
    .map((part) => part.trim())
    .filter(Boolean)

  return parts.length >= 2 ? parts[1] : rawTitle
})
const script = computed(() => props.referenceData?.script || '스크립트 내용이 없습니다.')
const canOpenFile = computed(() => Boolean(props.referenceData?.raw?.session_id))

const escapeHtml = (value = '') => (
  String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
)

const highlightedScript = computed(() => {
  const fullText = script.value
  const target = String(props.referenceData?.raw?.text || '').trim()

  if (target && fullText.includes(target)) {
    const highlightedTarget = `<mark class="home-reference-highlight">${escapeHtml(target)}</mark>`
    return fullText.split(target).map((part) => escapeHtml(part)).join(highlightedTarget)
  }

  return escapeHtml(fullText)
})
</script>

<template>
  <div 
    :class="[
      'home-reference-sidebar fixed right-4 top-4 bottom-4 h-[calc(100%-32px)] w-[400px] transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] z-[100] flex flex-col',
      isOpen ? 'translate-x-0' : 'translate-x-[120%]'
    ]"
  >
    <!-- Header -->
    <div class="h-16 shrink-0 flex items-center justify-between px-6 border-b border-slate-200 bg-white/70">
      <h2 class="text-[16px] font-bold text-[#1d1d1f] flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center">
          <span class="material-symbols-outlined text-[16px] text-indigo-600">article</span>
        </div>
        <span class="truncate pr-4">{{ title }}</span>
      </h2>
      <button 
        @click="emit('close')" 
        class="w-8 h-8 flex items-center justify-center rounded-full hover:bg-black/5 text-gray-400 hover:text-gray-700 transition-colors"
      >
        <span class="material-symbols-outlined text-[20px]">close</span>
      </button>
    </div>

    <!-- Content -->
    <div class="flex-1 overflow-y-auto p-6 custom-scrollbar">
      <div class="home-reference-script-card rounded-2xl p-6 min-h-full">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">전사 내용 스크립트</h3>
        <div
          class="text-[15px] leading-relaxed text-gray-700 whitespace-pre-wrap"
          v-html="highlightedScript"
        ></div>
      </div>
    </div>

    <div class="home-reference-footer shrink-0 p-5 border-t border-slate-200 bg-white/55">
      <button
        type="button"
        class="home-reference-open-file-btn"
        :disabled="!canOpenFile"
        @click="emit('openFile', referenceData)"
      >
        <span>파일로 이동</span>
        <span class="material-symbols-outlined text-[18px]">arrow_forward</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 10px;
}

.home-reference-sidebar {
  background: rgba(248, 250, 252, 0.94);
  border: 1px solid rgba(226, 232, 240, 0.96);
  border-radius: 24px;
  box-shadow: none;
  overflow: hidden;
}

.home-reference-script-card {
  background: rgba(255, 255, 255, 0.54);
  border: 1px solid rgba(226, 232, 240, 0.9);
  box-shadow: none;
}

:deep(.home-reference-highlight) {
  background: #ffeb3b;
  color: #111827;
  font-weight: 900;
  border-radius: 5px;
  padding: 1px 5px;
  margin: 0 -2px;
  box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.34);
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

.home-reference-footer {
  box-shadow: none;
}

.home-reference-open-file-btn {
  width: 100%;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-radius: 999px;
  background: #1f2937;
  color: #fff;
  font-size: 14px;
  font-weight: 800;
  border: 1px solid #1f2937;
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.home-reference-open-file-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}

.home-reference-open-file-btn:disabled {
  cursor: not-allowed;
  opacity: 0.35;
}
</style>
