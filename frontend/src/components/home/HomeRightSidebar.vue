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

const titleParts = computed(() => {
  const raw = props.referenceData?.raw || {}
  const rawTitle = raw.session_title || raw.recording_title || props.referenceData?.title || '스크립트'
  const parts = String(rawTitle)
    .split('>')
    .map((part) => part.trim())
    .filter(Boolean)

  if (parts.length >= 2) {
    return {
      title: parts.slice(0, -1).join(' > '),
      subtitle: parts.at(-1)
    }
  }

  const citationParts = String(raw.citation || '')
    .split('>')
    .map((part) => part.trim())
    .filter(Boolean)

  return {
    title: rawTitle,
    subtitle: citationParts.length >= 2 ? citationParts.at(-1) : ''
  }
})
const title = computed(() => titleParts.value.title)
const subtitle = computed(() => titleParts.value.subtitle)
const script = computed(() => (
  props.referenceData?.script
  || props.referenceData?.raw?.full_transcript
  || props.referenceData?.raw?.transcript
  || props.referenceData?.raw?.source_text
  || props.referenceData?.raw?.text
  || '스크립트 내용이 없습니다.'
))
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
      'home-reference-sidebar fixed right-3 top-[10px] bottom-[10px] h-[calc(100%-20px)] w-[416px] transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] z-[100] flex flex-col',
      isOpen ? 'translate-x-0' : 'translate-x-[120%]'
    ]"
  >
    <!-- Header -->
    <div class="h-16 shrink-0 flex items-center justify-between px-6 border-b border-[#eeeaf3] bg-white">
      <h2 class="text-[16px] font-bold text-[#1d1d1f] flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center">
          <span class="material-symbols-outlined text-[16px] text-indigo-600">article</span>
        </div>
        <span class="min-w-0 flex flex-col">
          <span class="truncate pr-4">{{ title }}</span>
          <span v-if="subtitle" class="home-reference-subtitle truncate pr-4">{{ subtitle }}</span>
        </span>
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
        <div class="home-reference-script-text" v-html="highlightedScript"></div>
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
  background: #f7f4fa;
  border: 1px solid rgba(255, 255, 255, 0.92);
  border-radius: var(--copy-radius-lg);
  box-shadow: var(--copy-card-shadow);
  overflow: hidden;
}

.home-reference-subtitle {
  margin-top: 2px;
  color: #8f8b98;
  font-size: 12px;
  font-weight: 800;
  line-height: 1.1;
}

.home-reference-script-card {
  background: #fff;
  border: 1px solid var(--copy-line);
  box-shadow: 0 18px 44px rgba(24, 28, 35, 0.05);
}

.home-reference-script-text {
  color: #3d4048;
  font-size: 14px;
  font-weight: 750;
  line-height: 1.78;
  white-space: pre-wrap;
  word-break: keep-all;
}

:deep(.home-reference-highlight) {
  background: #fff0a8;
  color: #111827;
  font-weight: 900;
  border-radius: 6px;
  padding: 1px 4px;
  margin: 0 -1px;
  box-shadow: none;
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
  background: var(--copy-black);
  color: #fff;
  font-size: 14px;
  font-weight: 800;
  border: 1px solid var(--copy-black);
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
