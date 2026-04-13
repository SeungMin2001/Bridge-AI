<!-- 워크스페이스 중앙 영역 상단에서 녹음 제어와 자료 업로드, 사이드바 토글을 담당하는 헤더입니다. -->
<script setup>
import { ref } from 'vue'

defineProps({
  isRecording: Boolean,
  recordingTimeText: String
})

const emit = defineEmits([
  'start-recording',
  'stop-recording',
  'main-sidebar-toggle',
  'right-sidebar-toggle',
  'material-selected'
])

const fileInputRef = ref(null)

const triggerMaterialPicker = () => {
  fileInputRef.value?.click()
}

const handleMaterialInputChange = (event) => {
  const [file] = Array.from(event.target.files || [])
  if (file) {
    emit('material-selected', file)
  }
  event.target.value = ''
}
</script>

<template>
  <header class="workspace-embedded-header h-[56px] flex items-center px-6 shrink-0">
    <div class="flex items-center gap-1.5 shrink-0">
      <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] shrink-0" title="사이드바 토글" @click="emit('main-sidebar-toggle')">
        <span class="material-symbols-outlined text-[20px]">side_navigation</span>
      </button>

      <button v-if="!isRecording" id="start" class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] shrink-0" @click="emit('start-recording')">
        <span class="material-symbols-outlined text-[20px]">mic</span>
      </button>
      <div
        v-else
        id="recording-timer"
        class="flex items-center gap-2 bg-[#FFF4F6] hover:bg-[#FFECEE] px-3 py-1.5 rounded-full cursor-pointer transition-colors border border-white/80 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_8px_18px_rgba(148,163,184,0.12)] shrink-0"
        @click="emit('stop-recording')"
      >
        <div class="recording-wave-container w-6 h-6 shrink-0">
          <div class="recording-wave-ring"></div>
          <div class="recording-wave-ring"></div>
          <div class="recording-wave-ring"></div>
          <span class="live-dot" style="position: relative; z-index: 1;"></span>
        </div>
        <span id="recording-time" class="text-[13px] font-bold text-[#1d1d1f] tabular-nums">{{ recordingTimeText }}</span>
      </div>
    </div>

    <div class="ml-auto flex items-center gap-1.5 shrink-0 pl-2">
      <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]">
        <span class="material-symbols-outlined text-[20px]">play_circle</span>
      </button>

      <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]" title="강의 자료 열기" @click="triggerMaterialPicker">
        <span class="material-symbols-outlined text-[20px]">folder_open</span>
      </button>

      <input
        ref="fileInputRef"
        type="file"
        accept=".pdf,.ppt,.pptx,application/pdf,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"
        class="hidden"
        @change="handleMaterialInputChange"
      />

      <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]" title="우측 사이드바 토글" @click="emit('right-sidebar-toggle')">
        <span class="material-symbols-outlined text-[20px] scale-x-[-1]">side_navigation</span>
      </button>
    </div>
  </header>

  <div class="workspace-embedded-divider shrink-0"></div>
</template>

<style scoped>
.workspace-embedded-header {
  background: #ffffff;
}

.workspace-embedded-divider {
  height: 1px;
  margin: 0 24px;
  background: rgba(0, 0, 0, 0.06);
}
</style>
