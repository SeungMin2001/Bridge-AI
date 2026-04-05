<script setup>
defineProps({
  isCollapsed: Boolean,
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['toggle', 'navigate'])
</script>

<template>
  <aside
    id="sidebar"
    :class="[
      isCollapsed ? 'w-16' : 'w-[280px]',
      'home-sidebar neo-sidebar flex flex-col h-[calc(100%-24px)] shrink-0 overflow-hidden transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] relative z-10 my-3 ml-3 text-white',
      { 'sidebar-collapsed': isCollapsed }
    ]"
  >
    <div class="sidebar-main-card flex flex-col h-full bg-transparent border-none">
      <div class="sidebar-header transition-all">
        <div class="sidebar-logo-section">
          <div class="sidebar-logo-box">
            <span class="material-symbols-outlined text-white text-[20px]">menu_book</span>
          </div>
          <span class="collapsible-content sidebar-logo-text font-extrabold">LectoAI</span>
        </div>
        <div class="sidebar-btn-group">
          <button class="sidebar-icon-btn" @click="emit('toggle')">
            <span class="material-symbols-outlined">
              {{ isCollapsed ? 'menu' : 'side_navigation' }}
            </span>
          </button>
          <button
            class="sidebar-icon-btn"
            @click="emit('navigate', 'workspace')"
          >
            <span class="material-symbols-outlined">edit_note</span>
          </button>
        </div>
      </div>
      
      <div class="sidebar-search-container collapsible-content">
        <span class="material-symbols-outlined">search</span>
        <span class="sidebar-search-text">제목으로 검색</span>
      </div>

      <div class="sidebar-content collapsible-content custom-scrollbar">
        <div class="sidebar-section-title">즐겨찾기</div>
        <div id="favorites-list" class="flex flex-col gap-1">
          <div 
            v-for="fav in fileTree.filter(item => favorites.has(item.id))" 
            :key="`fav-${fav.id}`" 
            class="sidebar-nav-item" 
            @click="emit('navigate', 'workspace')"
          >
            <span class="material-symbols-outlined nav-icon" :style="{ color: fav.color, fontVariationSettings: `'FILL' ${fav.type === 'folder' ? 1 : 0}` }">
              {{ fav.type === 'folder' ? 'folder' : 'description' }}
            </span>
            <span class="nav-text truncate">{{ fav.name }}</span>
            <span v-if="fav.type === 'folder'" class="material-symbols-outlined text-[#8e8e93] text-[18px]">expand_more</span>
          </div>
        </div>
      </div>

      <div class="mt-auto pt-5 flex justify-center w-full">
        <button class="sidebar-icon-btn !p-3 rounded-full text-[#1d1d1f]" @click="emit('navigate', 'home')">
          <span class="material-symbols-outlined !text-[24px]" style="font-variation-settings: 'FILL' 1">home</span>
        </button>
      </div>
    </div>
  </aside>
</template>
