<!-- 워크폴더 화면에서 폴더와 파일 목록을 그리드 형태로 시각화하는 컴포넌트입니다. -->
<script setup>
import { ref, computed } from 'vue'

const FOLDER_COLORS = {
  '#3b82f6': { body: 'fc-blue', tab: 'fc-blue-tab' },
  '#2dd4bf': { body: 'fc-teal', tab: 'fc-teal-tab' },
  '#ef4444': { body: 'fc-coral', tab: 'fc-coral-tab' },
  '#f87171': { body: 'fc-coral', tab: 'fc-coral-tab' },
  '#f59e0b': { body: 'fc-amber', tab: 'fc-amber-tab' },
  '#10b981': { body: 'fc-teal', tab: 'fc-teal-tab' },
  '#8b5cf6': { body: 'fc-purple', tab: 'fc-purple-tab' },
  '#a78bfa': { body: 'fc-purple', tab: 'fc-purple-tab' },
}

const props = defineProps({
  currentItems: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  navigationStack: { type: Array, default: () => [] },
  currentTitle: { type: String, default: '' }
})

const emit = defineEmits([
  'toggleStar',
  'enterFolder',
  'navigate',
  'goBack',
  'openFolderModal',
  'openFileModal'
])

const filterType = ref('all')
const isFilterOpen = ref(false)

const filterLabels = {
  all: '모두 보기',
  folder: '폴더만 보기',
  file: '파일만 보기'
}

const filteredItems = computed(() => {
  if (filterType.value === 'all') return props.currentItems
  return props.currentItems.filter(item => item.type === filterType.value)
})

const toggleFilter = () => {
  isFilterOpen.value = !isFilterOpen.value
}

const selectFilter = (type) => {
  filterType.value = type
  isFilterOpen.value = false
}

// Close dropdown on outside click
import { onMounted, onUnmounted } from 'vue'
const handleGlobalClick = (e) => {
  if (!e.target.closest('.filter-dropdown-wrapper')) {
    isFilterOpen.value = false
  }
}
onMounted(() => window.addEventListener('click', handleGlobalClick))
onUnmounted(() => window.removeEventListener('click', handleGlobalClick))
</script>

<template>
  <div class="flex-1 mt-6"> 
    <div class="shrink-0">
      <div class="grid-header flex items-center justify-between">
        <div class="flex items-center gap-3">
          <button 
            v-if="navigationStack.length > 0"
            @click="emit('goBack')"
            class="flex items-center justify-center p-2 bg-white border-none text-[#1d1d1f] cursor-pointer rounded-xl hover:bg-[#f2f2f7] transition-colors shadow-sm"
          >
            <span class="material-symbols-outlined text-[20px]">arrow_back</span>
          </button>
          <span class="section-title !m-0 transition-all duration-300">{{ currentTitle }}</span>
        </div>

        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2">
            <button class="header-action-btn group" @click="emit('openFolderModal')">
              <span class="material-symbols-outlined group-hover:scale-110 transition-transform">create_new_folder</span>
              <span>새 폴더</span>
            </button>
            <button class="header-action-btn group" @click="emit('openFileModal')">
              <span class="material-symbols-outlined group-hover:scale-110 transition-transform">description</span>
              <span>새 파일</span>
            </button>
          </div>

          <div class="w-[1px] h-4 bg-black/10 mx-1"></div>

          <div class="flex items-center">
            <div class="filter-dropdown-wrapper">
              <button 
                @click.stop="toggleFilter"
                :class="['filter-trigger-btn shadow-sm', { active: isFilterOpen }]"
              >
                <span>{{ filterLabels[filterType] }}</span>
                <span :class="['material-symbols-outlined dropdown-icon', { rotate: isFilterOpen }]">expand_more</span>
              </button>
              
              <Transition name="dropdown">
                <div v-if="isFilterOpen" class="filter-menu shadow-xl">
                  <div 
                    v-for="(label, type) in filterLabels" 
                    :key="type"
                    @click="selectFilter(type)"
                    :class="['filter-item', { selected: filterType === type }]"
                  >
                    {{ label }}
                    <span v-if="filterType === type" class="material-symbols-outlined check-icon">check</span>
                  </div>
                </div>
              </Transition>
            </div>
          </div>
        </div>
      </div>
      
      <div class="folder-grid">
        <template v-for="item in filteredItems" :key="item.id">
          <!-- Folder Card -->
          <div v-if="item.type === 'folder'" class="folder-card" @click="emit('enterFolder', $event, item)">
            <div :class="['folder-back', FOLDER_COLORS[item.color]?.body || 'fc-blue']">
              <div :class="['folder-tab', FOLDER_COLORS[item.color]?.tab || 'fc-blue-tab']" style="width: 45%;"></div>
            </div>
            <div class="folder-paper"></div>
            <div :class="['folder-body', FOLDER_COLORS[item.color]?.body || 'fc-blue']">
              <button
                :class="['star-btn', { starred: favorites.has(item.id) }]"
                @click="emit('toggleStar', $event, item.id)"
              >
                <span class="material-symbols-outlined" :style="{ fontSize: '16px', fontVariationSettings: `'FILL' ${favorites.has(item.id) ? 1 : 0}` }">star</span>
              </button>
              <div class="folder-icon-area">
                <span class="material-symbols-outlined" style="font-size: 22px; color: #fff; font-variation-settings: 'FILL' 1">folder</span>
              </div>
              <div class="folder-card-name">{{ item.name }}</div>
              <div class="folder-card-date">{{ item.date || '' }}</div>
            </div>
          </div>
          
          <!-- File Card -->
          <div v-else class="folder-card file-card" @click="emit('navigate', 'workspace')" style="display: flex; flex-direction: column; height: 160px;">
            <div style="height: 10px; flex-shrink: 0;"></div>
            <div style="background: #fff; border-radius: 14px; padding: 0; flex: 1; position: relative; overflow: hidden; box-shadow: 2px 3px 0px #e0e0e8; border: 1.5px solid #e5e5ea; display: flex; flex-direction: column;">
              <div :style="{ height: '6px', background: item.color || '#6366f1', borderRadius: '12px 12px 0 0' }"></div>
              <div style="position: absolute; top: 30px; left: 0; right: 0; bottom: 0; background-image: repeating-linear-gradient(transparent, transparent 22px, #f0f0f5 22px, #f0f0f5 23px); opacity: 0.6;"></div>
              <div style="position: relative; z-index: 1; padding: 14px; display: flex; flex-direction: column; flex: 1;">
                <button
                  :class="['star-btn', { starred: favorites.has(item.id) }]"
                  @click="emit('toggleStar', $event, item.id)"
                  style="position: absolute; top: 14px; right: 10px; background: rgba(0,0,0,0.04); color: #d1d1d6;"
                >
                  <span class="material-symbols-outlined" :style="{ fontSize: '16px', fontVariationSettings: `'FILL' ${favorites.has(item.id) ? 1 : 0}` }">star</span>
                </button>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                  <div :style="{ width: '36px', height: '36px', background: `${item.color}15` || '#ede9fe', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }">
                    <span class="material-symbols-outlined" :style="{ fontSize: '20px', color: item.color || '#6366f1', fontVariationSettings: `'FILL' 1` }">article</span>
                  </div>
                  <span :style="{ fontSize: '10px', fontWeight: '700', color: item.color || '#6366f1', background: `${item.color}15` || '#ede9fe', padding: '2px 8px', borderRadius: '100px', letterSpacing: '0.04em' }">FILE</span>
                </div>
                <div style="margin-top: auto;">
                  <div class="folder-card-name" style="color: #1d1d1f; font-size: 13px;">{{ item.name }}</div>
                  <div class="folder-card-date" style="color: #8e8e93;">{{ item.date || '' }}</div>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>

    <div class="h-[60px] shrink-0"></div>
  </div>
</template>

<style scoped>
.filter-dropdown-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.filter-trigger-btn {
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.05);
  border-radius: 12px;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13.5px;
  font-weight: 700;
  color: #1d1d1f;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.filter-trigger-btn:hover {
  background: #f9f9fb;
  border-color: rgba(0, 0, 0, 0.1);
}

.filter-trigger-btn.active {
  background: #f2f2f7;
  border-color: #1d1d1f;
}

.dropdown-icon {
  font-size: 18px;
  transition: transform 0.3s ease;
}

.dropdown-icon.rotate {
  transform: rotate(180deg);
}

.filter-menu {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 160px;
  background: white;
  border-radius: 16px;
  padding: 6px;
  z-index: 100;
  border: 1px solid rgba(0,0,0,0.06);
  transform-origin: top right;
}

.filter-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  color: #3a3a3c;
  cursor: pointer;
  transition: all 0.2s;
}

.filter-item:hover {
  background: #f2f2f7;
  color: #1d1d1f;
}

.filter-item.selected {
  background: #f2f2f7;
  color: #3b82f6;
}

.check-icon {
  margin-left: auto;
  font-size: 16px;
  color: #3b82f6;
}

/* Transition styles */
.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(0.95);
}
</style>
