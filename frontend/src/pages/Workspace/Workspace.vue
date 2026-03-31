<script setup>
import LeftSidebar from '../../components/workspace/LeftSidebar.vue'
import MainContent from '../../components/workspace/MainContent.vue'
import RightSidebar from '../../components/workspace/RightSidebar.vue'

defineProps({
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
</script>

<template>
  <div class="p-[12px] flex gap-[12px] relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
    <LeftSidebar
      :transcriptions="transcriptions"
      :fileTree="fileTree"
      :favorites="favorites"
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
      @rightSidebarToggle="emit('rightSidebarToggle')"
    />
    
    <RightSidebar 
      :visible="isRightSidebarVisible" 
      :aiInput="aiInput"
      @update:aiInput="emit('update:aiInput', $event)"
    />
  </div>
</template>
