<!-- 워크스페이스의 왼쪽 사이드바 본체로, 폴더 탐색기와 음성 전사 탭을 전환하며 보여줍니다. -->
<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import FolderSideTab from './FolderSideTab.vue'
import VoiceTransferSideTab from './VoiceTransferSideTab.vue'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  transcriptions: { type: Array, default: () => [] },
  recordingMode: { type: String, default: 'lecture' },
  activeFileId: { type: String, default: '' },
  isCollapsed: { type: Boolean, default: false }
})

const emit = defineEmits([
  'navigateHome',
  'fileSelect',
  'update:fileTree',
  'update:favorites',
  'addToNote',
  'askAi',
  'openStoredMaterial',
  'toggle'
])

const activeTab = ref('voice')
const width = ref(340)
const toastMsg = ref('')
const isResizing = ref(false)
const selectedTranscriptSource = ref(null)

const visibleTranscriptions = computed(() => selectedTranscriptSource.value?.transcriptions || props.transcriptions)

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

const handleShowLiveTranscripts = () => {
  selectedTranscriptSource.value = null
  activeTab.value = 'voice'
}

const handleOpenMaterial = ({ fileId, node, materialId, recording }) => {
  if (fileId && node) {
    emit('fileSelect', fileId, node)
  }

  if (recording) {
    selectedTranscriptSource.value = {
      title: recording.title || '연결된 녹음',
      transcriptions: recording.transcriptions || []
    }
  } else {
    selectedTranscriptSource.value = {
      title: '연결된 전사 없음',
      transcriptions: []
    }
  }

  activeTab.value = 'voice'
  emit('openStoredMaterial', materialId)
}

const handleOpenRecording = ({ fileId, node, recording }) => {
  if (fileId && node) {
    emit('fileSelect', fileId, node)
  }

  selectedTranscriptSource.value = {
    title: recording?.title || '저장된 녹음',
    transcriptions: recording?.transcriptions || []
  }
  activeTab.value = 'voice'
}
</script>

<template>
  <aside
    :class="[{ 'sidebar-collapsed': isCollapsed }]"
    id="sidebar"
    class="transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden rounded-[24px]"
    :style="{ width: isCollapsed ? '0px' : width + 'px', flexShrink: 0 }"
  >
    <div class="card workspace-sidebar-card h-full flex flex-col p-5 overflow-hidden min-w-[280px]">
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
      <div class="workspace-inset-shell p-1.5 rounded-[22px] flex gap-1.5 mb-4 collapsible-content">
        <button
          class="workspace-inset-pill flex-1 py-3 rounded-[18px] text-[12px] font-bold text-gray-500"
          :class="{ 'is-active text-black': activeTab === 'voice' }"
          @click="handleShowLiveTranscripts"
        >전사 내용</button>
        <button
          class="workspace-inset-pill flex-1 py-3 rounded-[18px] text-[12px] font-bold text-gray-500"
          :class="{ 'is-active text-black': activeTab === 'folders' }"
          @click="activeTab = 'folders'"
        >폴더</button>
      </div>

      <!-- Tab Content -->
      <div class="flex-1 flex flex-col overflow-hidden">
        <FolderSideTab 
          v-if="activeTab === 'folders'"
          :fileTree="fileTree"
          :favorites="favorites"
          :active-file-id="activeFileId"
          @update:fileTree="emit('update:fileTree', $event)"
          @update:favorites="emit('update:favorites', $event)"
          @fileSelect="(id, node) => emit('fileSelect', id, node)"
          @openMaterial="handleOpenMaterial"
          @openRecording="handleOpenRecording"
          @showToast="showToast"
          class="sidebar-content-animate"
        />
        <div v-else class="flex flex-col flex-1 min-h-0 sidebar-content-animate">
          <div v-if="selectedTranscriptSource" class="selected-transcript-source">
            <div class="min-w-0">
              <p>{{ selectedTranscriptSource.title }}</p>
              <span>저장된 전사 스크립트</span>
            </div>
            <button type="button" title="실시간 전사로 돌아가기" @click="handleShowLiveTranscripts">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>
          <VoiceTransferSideTab
            :transcriptions="visibleTranscriptions"
            :recording-mode="recordingMode"
            @addToNote="(text, source) => emit('addToNote', text, source)"
            @askAi="emit('askAi', $event)"
          />
        </div>
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
  background: var(--workspace-sidebar-card-bg);
  border: 1px solid var(--workspace-sidebar-card-border);
  box-shadow: var(--workspace-sidebar-card-shadow);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.workspace-sidebar-card::before {
  background: var(--workspace-sidebar-card-overlay);
}

.workspace-sidebar-card::after {
  border-color: var(--workspace-sidebar-card-inner-border);
}

.selected-transcript-source {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(229, 229, 234, 0.82);
}

.selected-transcript-source p {
  margin: 0;
  overflow: hidden;
  color: #1d1d1f;
  font-size: 12px;
  font-weight: 900;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.selected-transcript-source span {
  display: block;
  margin-top: 3px;
  color: #8e8e93;
  font-size: 10px;
  font-weight: 800;
}

.selected-transcript-source button {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 999px;
  color: #8e8e93;
  background: rgba(242, 242, 247, 0.92);
}

.selected-transcript-source button .material-symbols-outlined {
  font-size: 15px;
}
</style>
