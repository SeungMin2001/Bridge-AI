<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import FolderSideTab from './FolderSideTab.vue'
import VoiceTransferSideTab from './VoiceTransferSideTab.vue'

defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  transcriptions: { type: Array, default: () => [] },
  isCollapsed: { type: Boolean, default: false }
})

const emit = defineEmits([
  'navigateHome',
  'fileSelect',
  'update:fileTree',
  'update:favorites',
  'addToNote',
  'askAi',
  'toggle'
])

const activeTab = ref('folders')
const width = ref(280)
const toastMsg = ref('')
const isResizing = ref(false)

const handleMouseMove = (e) => {
  if (!isResizing.value) return
  const newWidth = e.clientX - 12
  if (newWidth > 160 && newWidth < 600) width.value = newWidth
}

const handleMouseUp = () => {
  if (!isResizing.value) return
  isResizing.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.body.classList.remove('is-resizing')
}

onMounted(() => {
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})

const handleResizerMouseDown = () => {
  isResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
}

const showToast = (msg) => {
  toastMsg.value = msg
  setTimeout(() => toastMsg.value = '', 2000)
}
</script>

<template>
  <aside
    :class="[{ 'sidebar-collapsed': isCollapsed }]"
    id="sidebar"
    class="transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden rounded-[24px]"
    :style="{ width: isCollapsed ? '0px' : width + 'px', flexShrink: 0 }"
  >
    <div class="card workspace-sidebar-card h-full bg-white flex flex-col p-5 overflow-hidden min-w-[280px]">
      <!-- Header -->
      <div class="flex items-center justify-between mb-5">
        <div class="flex items-center gap-2 font-extrabold tracking-tight">
          <div class="w-8 h-8 bg-[#1d1d1f] rounded-[10px] flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px] text-white">menu_book</span>
          </div>
          <span class="text-[20px] collapsible-content">LectoAI</span>
        </div>
        <div class="flex gap-1">
          <button class="btn-ghost-icon p-1.5 rounded-[10px]" title="사이드바 접기" @click="emit('toggle')">
            <span class="material-symbols-outlined text-[22px] text-[#8e8e93]">menu_open</span>
          </button>
        </div>
      </div>

      <!-- Tab Buttons -->
      <div class="bg-gray-100/50 p-1 rounded-lg flex gap-1 mb-4 collapsible-content">
        <button
          class="flex-1 py-1.5 rounded-md text-[12px] font-bold transition-all"
          :class="activeTab === 'folders' ? 'bg-white shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-black' : 'text-gray-500 hover:text-gray-700'"
          @click="activeTab = 'folders'"
        >폴더</button>
        <button
          class="flex-1 py-1.5 rounded-md text-[12px] font-bold transition-all"
          :class="activeTab === 'voice' ? 'bg-white shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-black' : 'text-gray-500 hover:text-gray-700'"
          @click="activeTab = 'voice'"
        >전사 내용</button>
      </div>

      <!-- Tab Content -->
      <div class="flex-1 flex flex-col overflow-hidden">
        <FolderSideTab 
          v-if="activeTab === 'folders'"
          :fileTree="fileTree"
          :favorites="favorites"
          @update:fileTree="emit('update:fileTree', $event)"
          @update:favorites="emit('update:favorites', $event)"
          @fileSelect="(id, node) => emit('fileSelect', id, node)"
          @showToast="showToast"
          class="sidebar-content-animate"
        />
        <VoiceTransferSideTab 
          v-else
          :transcriptions="transcriptions"
          @addToNote="(text, source) => emit('addToNote', text, source)"
          @askAi="emit('askAi', $event)"
          class="sidebar-content-animate"
        />
      </div>

      <!-- Home Button -->
      <footer class="mt-auto pt-4 flex items-center justify-center">
        <button class="btn-ghost-icon p-2.5 rounded-xl cursor-pointer flex items-center text-[#aeaeb2] justify-center" @click="emit('navigateHome')">
          <span class="material-symbols-outlined text-[24px]" style="font-variation-settings: 'FILL' 1">home</span>
        </button>
      </footer>
    </div>
  </aside>

  <!-- Resizer -->
  <div
    v-show="!isCollapsed"
    class="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
    id="resizer-left"
    :class="{ 'is-collapsed': isCollapsed }"
    @mousedown="handleResizerMouseDown"
  >
    <div class="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
  </div>



  <!-- Toast -->
  <div :class="['toast', { show: toastMsg }]" id="toast">{{ toastMsg }}</div>
</template>

<style scoped>
.workspace-sidebar-card {
  box-shadow: 0 14px 34px rgba(31, 41, 55, 0.08), 0 2px 8px rgba(15, 23, 42, 0.04);
}
</style>
