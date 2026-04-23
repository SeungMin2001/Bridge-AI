<!-- 워크스페이스 중앙 영역 상단에서 녹음 제어와 자료 업로드, 사이드바 토글을 담당하는 헤더입니다. -->
<script setup>
import { ref } from 'vue'

defineProps({
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingTimeText: String,
  showClosePreview: Boolean
})

const emit = defineEmits([
  'start-recording',
  'pause-recording',
  'resume-recording',
  'stop-recording',
  'main-sidebar-toggle',
  'right-sidebar-toggle',
  'material-selected',
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

      <div
        id="recording-control-bar"
        class="recording-control-bar shrink-0"
      >
        <transition-group name="recording-control" tag="div" class="recording-control-inner">
          <button
            v-if="!isRecording"
            key="start"
            class="recording-primary-btn"
            @click="emit('start-recording')"
          >
            녹음시작
          </button>
          <template v-else>
            <div
              key="voice-dots"
              class="recording-voice-dots shrink-0"
              :class="{ 'is-paused': isRecordingPaused }"
              aria-hidden="true"
            >
              <span class="recording-voice-dot"></span>
              <span class="recording-voice-dot"></span>
              <span class="recording-voice-dot"></span>
              <span class="recording-voice-dot"></span>
              <span class="recording-voice-dot"></span>
            </div>

            <span key="time" id="recording-time" class="recording-time-text tabular-nums">
              {{ recordingTimeText }}
            </span>

            <button
              key="pause-toggle"
              class="recording-icon-btn recording-icon-btn-sm"
              :class="{ 'is-paused': isRecordingPaused }"
              :aria-label="isRecordingPaused ? '녹음 재개' : '일시정지'"
              @click="isRecordingPaused ? emit('resume-recording') : emit('pause-recording')"
            >
              <span v-if="!isRecordingPaused" class="recording-pause-bars" aria-hidden="true">
                <span></span>
                <span></span>
              </span>
              <span v-else class="recording-play-triangle" aria-hidden="true"></span>
            </button>

            <button
              key="stop"
              class="recording-primary-btn"
              @click="emit('stop-recording')"
            >
              녹음종료
            </button>
          </template>
        </transition-group>
      </div>
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
  min-height: 56px;
}

.workspace-embedded-divider {
  height: 1px;
  margin: 0 24px;
  background: rgba(0, 0, 0, 0.06);
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
</style>
