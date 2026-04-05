<script setup>
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import HomeBanner from '../../components/home/HomeBanner.vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import HomeRightSidebar from '../../components/home/HomeRightSidebar.vue'
import { ref } from 'vue'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['navigate'])

const isSidebarCollapsed = ref(false)
const hasStartedChat = ref(false)
const activeReference = ref(null)
const isRightSidebarOpen = ref(false)

const handleMessageSent = (params) => {
  console.log('Message sent:', params)
  hasStartedChat.value = true
}

const openReferenceHandler = (refData) => {
  activeReference.value = refData
  isRightSidebarOpen.value = true
}
</script>

<template>
  <div class="p-[12px] flex gap-[12px] relative h-full w-full text-[#1d1d1f] overflow-hidden">
    <InfiniteGrid />
    
    <HomeSidebar 
      class="relative z-10"
      :isCollapsed="isSidebarCollapsed"
      :fileTree="fileTree"
      :favorites="favorites"
      @toggle="isSidebarCollapsed = !isSidebarCollapsed"
      @navigate="emit('navigate', $event)"
    />

    <main id="home-main-content" class="flex-1 relative z-10 transition-all duration-700 overflow-hidden">
      
      <!-- Unified Content Wrapper for seamless transition -->
      <div :class="[
        'absolute top-0 bottom-0 left-0 flex flex-col transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]',
        isRightSidebarOpen ? 'right-[420px]' : 'right-0',
        hasStartedChat ? '' : 'items-center justify-center -mt-20'
      ]">
        
        <HomeBanner 
          @sendMessage="handleMessageSent" 
          @openReference="openReferenceHandler"
          :class="['transition-all duration-700 w-full', hasStartedChat ? 'h-full' : 'max-w-[800px]']" 
        />
        
      </div>

    <!-- Navigation Button to All Folders (Fixed at viewport) -->
    <button 
      @click="emit('navigate', 'workfolder')"
      class="fixed bottom-8 right-8 neo-active-btn text-white px-6 py-4 rounded-full flex items-center gap-3 hover:scale-105 active:scale-95 transition-all duration-300 z-[60] group">
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
    />
  </div>
</template>

<style scoped>
</style>
