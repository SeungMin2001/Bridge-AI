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

const emit = defineEmits(['close'])

const title = computed(() => props.referenceData?.title || '스크립트')
const script = computed(() => props.referenceData?.script || '스크립트 내용이 없습니다.')
</script>

<template>
  <div 
    :class="[
      'fixed right-0 top-0 h-full w-[400px] bg-white/95 backdrop-blur-2xl border-l border-black/5 shadow-2xl transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] z-[100] flex flex-col',
      isOpen ? 'translate-x-0' : 'translate-x-full'
    ]"
  >
    <!-- Header -->
    <div class="h-16 shrink-0 flex items-center justify-between px-6 border-b border-black/5 bg-white/50">
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
    <div class="flex-1 overflow-y-auto p-6 custom-scrollbar bg-gray-50/50">
      <div class="bg-white rounded-2xl p-6 shadow-sm border border-black/5 min-h-full">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">전사 내용 스크립트</h3>
        <p class="text-[15px] leading-relaxed text-gray-700 whitespace-pre-wrap">
          {{ script }}
        </p>
      </div>
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
</style>
