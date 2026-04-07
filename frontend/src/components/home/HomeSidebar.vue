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
      'home-sidebar flex flex-col h-[calc(100%-24px)] shrink-0 overflow-hidden transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] relative z-10 my-3 ml-3 rounded-[24px]',
      { 'sidebar-collapsed': isCollapsed }
    ]"
  >
    <div class="sidebar-main-card card flex flex-col h-full p-5 home-left-sidebar-card">
      <div class="sidebar-header transition-all">
        <div class="sidebar-logo-section">
          <div class="sidebar-logo-box">
            <span class="material-symbols-outlined text-white text-[20px]">menu_book</span>
          </div>
          <span class="collapsible-content sidebar-logo-text font-extrabold">LectoAI</span>
        </div>
        <div class="sidebar-btn-group">
          <button class="sidebar-icon-btn home-sidebar-icon-btn" @click="emit('toggle')">
            <span class="material-symbols-outlined">
              {{ isCollapsed ? 'menu' : 'side_navigation' }}
            </span>
          </button>
          <button
            class="sidebar-icon-btn home-sidebar-icon-btn"
            @click="emit('navigate', 'workspace')"
          >
            <span class="material-symbols-outlined">edit_note</span>
          </button>
        </div>
      </div>
      
      <div class="sidebar-search-container collapsible-content workspace-inset-shell rounded-[24px]">
        <span class="material-symbols-outlined">search</span>
        <span class="sidebar-search-text">제목으로 검색</span>
      </div>

      <div class="sidebar-content collapsible-content custom-scrollbar">
        <div class="sidebar-section-title">즐겨찾기</div>
        <div id="favorites-list" class="flex flex-col gap-1">
          <div 
            v-for="fav in fileTree.filter(item => favorites.has(item.id))" 
            :key="`fav-${fav.id}`" 
            class="sidebar-nav-item home-sidebar-nav-item" 
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
        <button class="sidebar-icon-btn home-sidebar-icon-btn !p-3 rounded-full text-[#1d1d1f]" @click="emit('navigate', 'home')">
          <span class="material-symbols-outlined !text-[24px]" style="font-variation-settings: 'FILL' 1">home</span>
        </button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.home-left-sidebar-card {
  background: linear-gradient(160deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.58));
  border: 1px solid rgba(255, 255, 255, 0.86);
  box-shadow: 0 28px 56px rgba(148, 163, 184, 0.14), 0 10px 26px rgba(255, 255, 255, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.98);
}

.home-left-sidebar-card::before {
  background:
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.96), transparent 40%),
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.72), transparent 30%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0));
}

.home-left-sidebar-card::after {
  border-color: rgba(255, 255, 255, 0.42);
}

.home-sidebar-icon-btn {
  border: 1px solid rgba(255, 255, 255, 0.58);
  background: rgba(255, 255, 255, 0.32);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.84);
}

.home-sidebar-nav-item {
  border: 1px solid rgba(255, 255, 255, 0.44);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.46), rgba(255, 255, 255, 0.22));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
}
</style>
