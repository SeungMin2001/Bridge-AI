<script setup>
import { computed } from 'vue'
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import HomeBanner from '../../components/home/HomeBanner.vue'
import HomeGrid from '../../components/home/HomeGrid.vue'
import HomeModals from '../../components/home/HomeModals.vue'
import AiAssistant from '../../components/home/AiAssistant.vue'
import { useHome } from '../../composables/useHome'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['navigate', 'update:fileTree', 'update:favorites'])

const {
  isSidebarCollapsed,
  setIsSidebarCollapsed,
  isAiChatOpen,
  setIsAiChatOpen,
  isFolderModalOpen,
  isFileModalOpen,
  selectedColor,
  navigationStack,
  aiWinRef,
  aiBtnRef,
  newFolderName,
  newFileName,
  toggleStar,
  handleCreateFolder,
  handleCreateFile,
  handleEnterFolder,
  handleGoBack
} = useHome(props, emit)

const currentTitle = computed(() => {
  return navigationStack.value.length > 0
    ? navigationStack.value[navigationStack.value.length - 1].name
    : '내 폴더'
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
  <div class="p-[12px] flex gap-[12px] relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
    <HomeSidebar 
      :isCollapsed="isSidebarCollapsed"
      :fileTree="fileTree"
      :favorites="favorites"
      @toggle="isSidebarCollapsed = !isSidebarCollapsed"
      @navigate="emit('navigate', $event)"
    />

    <main id="home-main-content" class="custom-scrollbar flex-1 overflow-y-auto">
      <HomeBanner />
      
      <HomeGrid 
        :currentItems="currentItems"
        :currentTitle="currentTitle"
        :navigationStack="navigationStack"
        :favorites="favorites"
        @goBack="handleGoBack"
        @enterFolder="handleEnterFolder"
        @openFolderModal="isFolderModalOpen = true"
        @openFileModal="isFileModalOpen = true"
        @toggleStar="toggleStar"
        @navigate="emit('navigate', $event)"
      />
    </main>

    <AiAssistant 
      :isOpen="isAiChatOpen"
      :aiWinRef="aiWinRef"
      :aiBtnRef="aiBtnRef"
      @update:isOpen="isAiChatOpen = $event"
    />

    <HomeModals 
      :isFolderModalOpen="isFolderModalOpen"
      :isFileModalOpen="isFileModalOpen"
      :selectedColor="selectedColor"
      :newFolderName="newFolderName"
      :newFileName="newFileName"
      @update:isFolderModalOpen="isFolderModalOpen = $event"
      @update:isFileModalOpen="isFileModalOpen = $event"
      @update:selectedColor="selectedColor = $event"
      @update:newFolderName="newFolderName = $event"
      @update:newFileName="newFileName = $event"
      @createFolder="handleCreateFolder"
      @createFile="handleCreateFile"
    />
  </div>
</template>
<style src="./Home.css"></style>
