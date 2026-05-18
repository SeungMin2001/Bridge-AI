<!-- 워크스페이스 중앙 영역 상단에서 녹음 제어와 자료 업로드, 사이드바 토글을 담당하는 헤더입니다. -->
<script setup>
import { ref } from 'vue'

const props = defineProps({
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingTimeText: String,
  recordingAudioLevel: { type: Number, default: 0 },
  tabs: { type: Array, default: () => [] },
  activeTab: { type: String, default: 'materials' },
  showClosePreview: Boolean,
  hasWordInsight: Boolean,
  wordInsightVisible: Boolean
})

const emit = defineEmits([
  'start-recording',
  'pause-recording',
  'resume-recording',
  'stop-recording',
  'tab-change',
  'main-sidebar-toggle',
  'right-sidebar-toggle',
  'material-selected',
  'word-insight-click',
  'close-preview-material'
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
  <header class="workspace-embedded-header flex items-center px-6 shrink-0">
    <div class="flex items-center gap-2.5 shrink-0 min-w-0">
      <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] shrink-0" title="사이드바 토글" @click="emit('main-sidebar-toggle')">
        <span class="material-symbols-outlined text-[20px]">side_navigation</span>
      </button>

      <nav class="workspace-header-tabs" aria-label="워크스페이스 탭">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          type="button"
          class="workspace-header-tab"
          :class="{ 'is-active': activeTab === tab.key }"
          @click="emit('tab-change', tab.key)"
        >
          {{ tab.label }}
        </button>
      </nav>

    </div>

    <div class="ml-auto flex items-center gap-1.5 shrink-0 pl-3 self-center">
      <button
        v-if="showClosePreview"
        class="preview-close-header-btn"
        title="자료 닫기"
        @click="emit('close-preview-material')"
      >
        <span class="material-symbols-outlined text-[18px]">close</span>
        <span>닫기</span>
      </button>

      <button
        v-if="hasWordInsight"
        class="btn-ghost-icon p-2 rounded-lg shrink-0 word-insight-btn text-[#8e8e93]"
        :aria-label="wordInsightVisible ? 'AI 결과 카드 접기' : 'AI 결과 카드 다시 보기'"
        :title="wordInsightVisible ? 'AI 결과 카드 접기' : 'AI 결과 카드 다시 보기'"
        @click="emit('word-insight-click')"
      >
        <span class="material-symbols-outlined text-[20px]">
          {{ wordInsightVisible ? 'unfold_less' : 'unfold_more' }}
        </span>
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
  position: relative;
  background: #ffffff;
  min-height: 56px;
}

.workspace-embedded-divider {
  height: 1px;
  margin: 0 24px;
  background: rgba(0, 0, 0, 0.06);
}

.workspace-header-tabs {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  display: inline-flex;
  align-items: center;
  gap: 26px;
  height: 42px;
  margin-left: 0;
  z-index: 1;
}

.workspace-header-tab {
  position: relative;
  height: 42px;
  border: 0;
  background: transparent;
  color: #8e8e93;
  font-size: 15px;
  font-weight: 900;
  letter-spacing: -0.02em;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.18s ease;
}

.workspace-header-tab:hover,
.workspace-header-tab.is-active {
  color: #1d1d1f;
}

.workspace-header-tab.is-active::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: -8px;
  height: 3px;
  border-radius: 999px;
  background: #1d1d1f;
}

.preview-close-header-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 36px;
  padding: 0 13px;
  border-radius: 999px;
  border: 1px solid rgba(226, 232, 240, 0.95);
  background: rgba(255, 255, 255, 0.92);
  color: #374151;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: -0.01em;
  box-shadow: 0 10px 22px rgba(148, 163, 184, 0.08);
  transition: transform 0.2s ease, border-color 0.2s ease, background-color 0.2s ease;
}

.preview-close-header-btn:hover {
  border-color: rgba(148, 163, 184, 0.42);
  background: #ffffff;
}

.preview-close-header-btn:active {
  transform: scale(0.98);
}

.recording-control-bar {
  position: relative;
  min-height: 36px;
  display: flex;
  align-items: center;
}

.word-insight-btn {
  transition: color 0.2s ease, transform 0.2s ease;
}

.word-insight-btn:hover {
  color: #1d1d1f;
}

.word-insight-btn:active {
  transform: scale(0.98);
}

.recording-control-inner {
  display: inline-flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
  white-space: nowrap;
}

.recording-time-text {
  font-size: 14px;
  font-weight: 800;
  color: #1d1d1f;
  letter-spacing: 0.02em;
}

.recording-icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 999px;
  background: #e5e5ea;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.recording-icon-btn:hover {
  background: #dbdbe2;
}

.recording-icon-btn:active {
  transform: scale(0.98);
}

.recording-icon-btn-sm {
  width: 36px;
  height: 36px;
}

.recording-icon-btn.is-paused {
  background: #fff0f1;
}

.recording-icon-btn.is-paused:hover {
  background: #ffe4e7;
}

.recording-pause-bars {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.recording-pause-bars span {
  display: block;
  width: 5px;
  height: 18px;
  border-radius: 999px;
  background: #5f6472;
}

.recording-play-triangle {
  width: 0;
  height: 0;
  margin-left: 2px;
  border-top: 9px solid transparent;
  border-bottom: 9px solid transparent;
  border-left: 14px solid #ef4444;
}

.recording-primary-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 88px;
  height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  background: #111111;
  color: #ffffff;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: -0.01em;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.recording-primary-btn:hover {
  background: #1f1f1f;
}

.recording-primary-btn:active {
  transform: scale(0.98);
}

.recording-control-enter-active,
.recording-control-leave-active {
  transition: opacity 0.26s ease, transform 0.32s cubic-bezier(0.22, 1, 0.36, 1);
}

.recording-control-move {
  transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1);
}

.recording-control-enter-from {
  opacity: 0;
  transform: translateY(8px) scale(0.96);
}

.recording-control-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.96);
}

.recording-control-leave-active {
  position: absolute;
}

.recording-voice-dots.is-paused .recording-voice-dot {
  animation-play-state: paused;
}

.recording-voice-dots.is-paused .recording-voice-dot {
  opacity: 0.82;
  transform: scaleY(0.85);
}

.recording-voice-dot {
  animation: none;
  transition: transform 90ms ease-out, opacity 90ms ease-out;
}
</style>
