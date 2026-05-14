<!-- 사용자의 폴더 구조를 관리하고 파일들을 탐색할 수 있는 워크폴더 페이지 컴포넌트입니다. -->
<script setup>
import { computed } from 'vue'
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import HomeGrid from '../../components/home/HomeGrid.vue'
import HomeModals from '../../components/home/HomeModals.vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import { useHome } from '../../composables/useHome'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['navigate', 'update:fileTree', 'update:favorites', 'fileSelect'])

const {
  isSidebarCollapsed,
  isFolderModalOpen,
  isFileModalOpen,
  isEditItemModalOpen,
  selectedColor,
  selectedTag,
  selectedFileIcon,
  navigationStack,
  editingItemType,
  editingFileKind,
  newFolderName,
  newFileName,
  FOLDER_COLORS,
  LECTURE_FILE_COLORS,
  MEETING_FILE_COLORS,
  FILE_TAGS,
  FILE_ICONS,
  toggleStar,
  openFileCreateModal,
  handleFileTagChange,
  handleCreateFolder,
  handleCreateFile,
  openItemEditModal,
  closeEditItemModal,
  handleUpdateItem,
  handleDeleteEditingItem,
  handleEnterFolder,
  handleGoBack
} = useHome(props, emit)

const currentTitle = computed(() => {
  return navigationStack.value.length > 0
    ? navigationStack.value[navigationStack.value.length - 1].name
    : '전체 폴더'
})

const currentItems = computed(() => {
  if (navigationStack.value.length === 0) return props.fileTree
  
  const currentFolderId = navigationStack.value[navigationStack.value.length - 1].id
  const findFolder = (nodes, id) => {
    for (const node of nodes) {
      if (node.id === id) return node
      if (node.children) {
        const found = findFolder(node.children, id)
        if (found) return found
      }
    }
    return null
  }
  const folder = findFolder(props.fileTree, currentFolderId)
  return folder ? (folder.children || []) : []
})
</script>

<template>
  <div class="p-[12px] flex gap-[12px] relative h-full w-full bg-transparent text-[#1e293b] overflow-hidden">
    <InfiniteGrid />
    <HomeSidebar 
      class="relative z-10"
      :isCollapsed="isSidebarCollapsed"
      :fileTree="fileTree"
      :favorites="favorites"
      @toggle="isSidebarCollapsed = !isSidebarCollapsed"
      @navigate="emit('navigate', $event)"
    />

    <main id="home-main-content" class="custom-scrollbar flex-1 overflow-y-auto relative z-10">
      <HomeGrid 
        :currentItems="currentItems"
        :currentTitle="currentTitle"
        :navigationStack="navigationStack"
        :favorites="favorites"
        @goBack="handleGoBack"
        @enterFolder="handleEnterFolder"
        @openFolderModal="isFolderModalOpen = true"
        @openFileModal="openFileCreateModal"
        @toggleStar="toggleStar"
        @openItemEditModal="openItemEditModal"
        @navigate="emit('navigate', $event)"
        @openFile="(item) => { emit('fileSelect', item.id, item); emit('navigate', 'workspace') }"
      />
    </main>

    <HomeModals 
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
<style src="./Workfolder.css"></style>
