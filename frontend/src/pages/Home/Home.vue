<!-- 애플리케이션의 메인 홈 대시보드 페이지 컴포넌트입니다. -->
<script setup>
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import HomeBanner from '../../components/home/HomeBanner.vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import HomeRightSidebar from '../../components/home/HomeRightSidebar.vue'
import HomeModals from '../../components/home/HomeModals.vue'
import { useHome } from '../../composables/useHome'
import { ref } from 'vue'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  recentFiles: { type: Array, default: () => [] }
})

const emit = defineEmits(['navigate', 'fileSelect', 'update:fileTree', 'update:favorites'])

const isSidebarCollapsed = ref(false)
const hasStartedChat = ref(false)
const activeReference = ref(null)
const isRightSidebarOpen = ref(false)
const {
  isFolderModalOpen,
  isFileModalOpen,
  isEditItemModalOpen,
  selectedColor,
  selectedTag,
  selectedFileIcon,
  editingItemType,
  editingFileKind,
  newFolderName,
  newFileName,
  FOLDER_COLORS,
  LECTURE_FILE_COLORS,
  MEETING_FILE_COLORS,
  FILE_TAGS,
  FILE_ICONS,
  openFileCreateModal,
  handleFileTagChange,
  handleCreateFolder,
  handleCreateFile,
  closeEditItemModal,
  handleUpdateItem,
  handleDeleteEditingItem
} = useHome(props, emit)

const handleMessageSent = (params) => {
  console.log('Message sent:', params)
  hasStartedChat.value = true
}

const openReferenceHandler = (refData) => {
  activeReference.value = refData
  isRightSidebarOpen.value = true
}

const findNodeById = (nodes = [], id = '') => {
  for (const node of nodes) {
    if (node?.id === id) return node
    if (Array.isArray(node?.children)) {
      const found = findNodeById(node.children, id)
      if (found) return found
    }
  }
  return null
}

const openReferenceFileHandler = (refData) => {
  const sessionId = refData?.raw?.session_id
  if (!sessionId) return

  const node = findNodeById(props.fileTree, sessionId)
  if (node) {
    emit('fileSelect', sessionId, node)
  }

  emit('navigate', 'workspace')
}

const openRecentFileHandler = (file) => {
  if (!file?.id || !file?.node) return
  emit('fileSelect', file.id, file.node)
  emit('navigate', 'workspace')
}
</script>

<template>
  <div class="copy-app-frame flex relative h-full w-full text-[#1d1d1f] overflow-hidden">
    <InfiniteGrid />
    
    <HomeSidebar 
      class="relative z-10"
      :isCollapsed="isSidebarCollapsed"
      activeView="home"
      :fileTree="fileTree"
      :favorites="favorites"
      @toggle="isSidebarCollapsed = !isSidebarCollapsed"
      @navigate="emit('navigate', $event)"
      @openScheduleSource="openReferenceHandler"
      @openFileCreate="openFileCreateModal"
    />

    <main
      id="home-main-content"
      :class="[
        'home-main-shell flex-1 relative z-10 transition-all duration-700 overflow-hidden',
        isRightSidebarOpen ? 'home-main-shell-with-reference' : ''
      ]"
    >
      <button class="copy-home-bell-btn" type="button" aria-label="알림">
        <span class="material-symbols-outlined">notifications</span>
      </button>
      
      <!-- Unified Content Wrapper for seamless transition -->
      <div :class="[
        'absolute top-0 bottom-0 left-0 flex flex-col transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]',
        'right-0',
        hasStartedChat ? '' : 'items-center justify-start'
      ]">
        
        <HomeBanner 
          :recentFiles="recentFiles"
          @sendMessage="handleMessageSent" 
          @openReference="openReferenceHandler"
          @openRecentFile="openRecentFileHandler"
          :class="['transition-all duration-700 w-full', hasStartedChat ? 'h-full' : 'max-w-[800px]']" 
        />
        
      </div>

    <!-- Navigation Button to All Folders (Fixed at viewport) -->
    <button 
      @click="emit('navigate', 'workfolder')"
      class="hidden fixed bottom-8 right-8 neo-active-btn text-white px-6 py-4 rounded-full items-center gap-3 hover:scale-105 active:scale-95 transition-all duration-300 z-[60] group">
      <span class="font-bold tracking-tight">전체 폴더 가기</span>
      <div class="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center group-hover:bg-white/30 transition-colors">
        <span class="material-symbols-outlined text-[18px]">arrow_forward</span>
      </div>
    </button>

    </main>
    
    <!-- Right Sidebar for References -->
    <HomeRightSidebar 
      :isOpen="isRightSidebarOpen"
      :referenceData="activeReference"
      @close="isRightSidebarOpen = false"
      @openFile="openReferenceFileHandler"
    />

    <HomeModals
      placement="rail"
      :isFolderModalOpen="isFolderModalOpen"
      :isFileModalOpen="isFileModalOpen"
      :isEditItemModalOpen="isEditItemModalOpen"
      :selectedColor="selectedColor"
      :selectedTag="selectedTag"
      :selectedFileIcon="selectedFileIcon"
      :newFolderName="newFolderName"
      :newFileName="newFileName"
      :editingItemType="editingItemType"
      :editingFileKind="editingFileKind"
      :folderColors="FOLDER_COLORS"
      :lectureFileColors="LECTURE_FILE_COLORS"
      :meetingFileColors="MEETING_FILE_COLORS"
      :fileTags="FILE_TAGS"
      :fileIcons="FILE_ICONS"
      @update:isFolderModalOpen="isFolderModalOpen = $event"
      @update:isFileModalOpen="isFileModalOpen = $event"
      @update:isEditItemModalOpen="!$event && closeEditItemModal()"
      @update:selectedColor="selectedColor = $event"
      @update:selectedTag="handleFileTagChange"
      @update:selectedFileIcon="selectedFileIcon = $event"
      @update:newFolderName="newFolderName = $event"
      @update:newFileName="newFileName = $event"
      @createFolder="handleCreateFolder"
      @createFile="handleCreateFile()"
      @updateItem="handleUpdateItem"
      @deleteEditingItem="handleDeleteEditingItem"
    />
  </div>
</template>

<style scoped>
.copy-app-frame {
  background: var(--copy-bg);
}

#home-main-content.home-main-shell {
  min-width: 0;
  height: calc(100% - 20px);
  max-height: calc(100% - 20px);
  min-height: calc(100% - 20px);
  align-self: flex-start;
  margin: 10px 12px 10px 0;
  box-sizing: border-box;
  border-radius: 32px;
  background: var(--copy-surface);
  border: 0;
  box-shadow: -10px 0 34px rgba(48, 42, 58, 0.05);
}

.home-main-shell-with-reference {
  flex: 0 0 calc(100% - var(--copy-rail-width) - 450px);
  max-width: calc(100% - var(--copy-rail-width) - 450px);
  margin-right: 14px;
  border-radius: 32px;
}

.home-main-shell::before {
  content: '';
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-color: #fff;
  background-image:
    linear-gradient(rgba(21, 22, 26, 0.052) 1px, transparent 1px),
    linear-gradient(90deg, rgba(21, 22, 26, 0.052) 1px, transparent 1px);
  background-size: 40px 40px;
  background-position: 0 0;
  opacity: 0.78;
  animation: homeCardGridDrift 5.2s linear infinite;
}

.copy-home-bell-btn {
  position: absolute;
  top: 28px;
  right: 34px;
  z-index: 20;
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 21px;
  background: #fff;
  color: var(--copy-black);
  box-shadow: 0 10px 24px rgba(48, 42, 58, 0.05);
}

.copy-home-bell-btn .material-symbols-outlined {
  font-size: 22px;
}

@keyframes homeCardGridDrift {
  to {
    background-position: 40px 40px, 40px 40px;
  }
}

.copy-app-frame > :deep(.infinite-grid-container) {
  opacity: 0;
}
</style>
