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
  'openItemEditModal',
  'enterFolder',
  'navigate',
  'openFile',
  'goBack',
  'openFolderModal',
  'openFileModal'
])

const filterType = ref('all')
const isFilterOpen = ref(false)
const selectedIds = ref(new Set())

const filterLabels = {
  all: '모두 보기',
  folder: '폴더만 보기',
  file: '파일만 보기'
}

const filteredItems = computed(() => {
  if (filterType.value === 'all') return props.currentItems
  return props.currentItems.filter(item => item.type === filterType.value)
})

const selectedCount = computed(() => selectedIds.value.size)
const allVisibleSelected = computed(() => (
  filteredItems.value.length > 0 && filteredItems.value.every((item) => selectedIds.value.has(item.id))
))

const isItemSelected = (item) => selectedIds.value.has(item.id)

const toggleItemSelected = (item) => {
  const next = new Set(selectedIds.value)
  if (next.has(item.id)) next.delete(item.id)
  else next.add(item.id)
  selectedIds.value = next
}

const toggleAllVisible = () => {
  if (allVisibleSelected.value) {
    selectedIds.value = new Set()
    return
  }
  selectedIds.value = new Set(filteredItems.value.map((item) => item.id))
}

const toggleFilter = () => {
  isFilterOpen.value = !isFilterOpen.value
}

const selectFilter = (type) => {
  filterType.value = type
  isFilterOpen.value = false
}

const colorWithAlpha = (color = '#6366f1', alpha = 0.12) => {
  const hex = String(color).trim()
  const fullHex = /^#[0-9a-fA-F]{6}$/.test(hex)
    ? hex
    : (/^#[0-9a-fA-F]{3}$/.test(hex)
        ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
        : '#6366f1')
  const value = fullHex.slice(1)
  const red = parseInt(value.slice(0, 2), 16)
  const green = parseInt(value.slice(2, 4), 16)
  const blue = parseInt(value.slice(4, 6), 16)
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}

const isMeetingCard = (item) => item?.fileKind === 'meeting' || item?.tag === '회의'

const getDefaultFileIcon = (item) => {
  if (item?.tag === '프로젝트') return 'workspaces'
  if (item?.tag === '개인') return 'person'
  if (item?.tag === '중요') return 'priority_high'
  return isMeetingCard(item) ? 'groups_2' : 'article'
}

const getFileCardIcon = (item) => item?.fileIcon || getDefaultFileIcon(item)

const getFileIconBoxStyle = (item) => ({
  width: '36px',
  height: '36px',
  background: colorWithAlpha(item?.color, 0.14),
  borderRadius: '10px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flex: '0 0 auto'
})

const getFileTagStyle = (item) => ({
  fontSize: '10px',
  fontWeight: '700',
  color: item?.color || '#6366f1',
  background: colorWithAlpha(item?.color, 0.12),
  padding: '2px 8px',
  borderRadius: '100px',
  letterSpacing: '0.04em'
})

const openItem = (event, item) => {
  if (item.type === 'folder') {
    emit('enterFolder', event, item)
    return
  }
  emit('openFile', item)
}

const getItemIcon = (item) => item.type === 'folder' ? 'folder' : getFileCardIcon(item)

const getItemTag = (item) => {
  if (item.type === 'folder') return '폴더'
  return item.tag || (item.fileKind === 'meeting' ? '회의' : '수업')
}

const getRowStyle = (item) => ({
  background: colorWithAlpha(item?.color, item.type === 'folder' ? 0.12 : 0.14),
  '--row-accent': item?.color || '#6366f1',
})

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
  <div class="flex-1 mt-0"> 
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

        <div class="grid-header-actions flex items-center gap-4">
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
      
      <div class="work-list-shell">
        <div class="work-list-head">
          <div class="work-head-main">
            <button
              type="button"
              :class="['work-check-placeholder', { 'is-checked': allVisibleSelected }]"
              aria-label="전체 선택"
              @click="toggleAllVisible"
            ></button>
            <span>보드 이름</span>
            <div v-if="selectedCount" class="work-selection-toolbar">
              <button class="work-selection-action delete" type="button">
                <span class="material-symbols-outlined">delete</span>
                삭제하기
              </button>
              <button class="work-selection-action" type="button">
                <span class="material-symbols-outlined">drive_file_move</span>
                이동하기
              </button>
            </div>
          </div>
          <button class="work-sort-btn" type="button">
            <span>생성일</span>
            <span class="material-symbols-outlined">south</span>
          </button>
        </div>

        <div class="work-row-list">
        <template v-for="item in filteredItems" :key="item.id">
          <article
            class="work-row-card"
            :class="{ 'is-selected': isItemSelected(item) }"
            :style="getRowStyle(item)"
            @click="openItem($event, item)"
          >
            <button
              type="button"
              :class="['work-row-check', { 'is-checked': isItemSelected(item) }]"
              aria-label="선택"
              @click.stop="toggleItemSelected(item)"
            ></button>
            <button
              :class="['work-row-star', { starred: favorites.has(item.id) }]"
              type="button"
              @click.stop="emit('toggleStar', $event, item.id)"
            >
              <span class="material-symbols-outlined" :style="{ fontVariationSettings: `'FILL' ${favorites.has(item.id) ? 1 : 0}` }">star</span>
            </button>
            <div class="work-row-icon">
              <span class="material-symbols-outlined">{{ getItemIcon(item) }}</span>
            </div>
            <div class="work-row-title">
              <strong>{{ item.name }}</strong>
              <span class="work-row-tag" :style="{ color: item.color || '#6366f1' }">{{ getItemTag(item) }}</span>
            </div>
            <time class="work-row-date">{{ item.date || '-' }}</time>
            <button class="work-row-more" title="설정" type="button" @click.stop="emit('openItemEditModal', item.id)">
              <span class="material-symbols-outlined">more_horiz</span>
            </button>
          </article>
        </template>
        </div>
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

.file-card-actions {
  position: absolute;
  top: 14px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 3;
}

.folder-card-actions {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 3;
}

.folder-card-action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.22);
  color: rgba(255, 255, 255, 0.9);
  border: none;
  cursor: pointer;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.folder-card-action-btn:hover {
  background: rgba(255, 255, 255, 0.34);
}

.folder-card-action-btn:active {
  transform: scale(0.96);
}

.folder-card-star-btn {
  position: static !important;
}

.file-card-action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.04);
  color: #8e8e93;
  border: none;
  cursor: pointer;
  transition: background-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.file-card-action-btn:hover {
  background: rgba(0, 0, 0, 0.08);
  color: #1d1d1f;
}

.file-card-action-btn:active {
  transform: scale(0.96);
}

.file-card-star-btn {
  position: static !important;
  background: rgba(255, 202, 40, 0.18) !important;
  color: #f4b400 !important;
}

.file-card-star-btn:hover {
  background: rgba(255, 202, 40, 0.28) !important;
}

.file-card-star-btn.starred {
  background: rgba(255, 202, 40, 0.24) !important;
  color: #f4b400 !important;
  box-shadow: 0 8px 18px rgba(244, 180, 0, 0.16);
}

.work-list-shell {
  width: 100%;
  margin: 0 auto;
}

.work-list-head,
.work-row-card {
  display: grid;
  grid-template-columns: 30px 30px 36px minmax(0, 1fr) 250px 30px;
  align-items: center;
  gap: 11px;
}

.work-list-head {
  min-height: 36px;
  padding: 0 20px;
  color: var(--copy-muted);
  font-size: 12px;
  font-weight: 900;
}

.work-head-main {
  grid-column: 1 / 5;
  display: flex;
  align-items: center;
  gap: 14px;
}

.work-check-placeholder,
.work-row-check {
  position: relative;
  width: 17px;
  height: 17px;
  border-radius: 5px;
  border: 2px solid #c7c9d1;
  background: rgba(255, 255, 255, 0.72);
  cursor: pointer;
}

.work-check-placeholder::after,
.work-row-check::after {
  content: '';
  position: absolute;
  left: 5px;
  top: 1px;
  width: 5px;
  height: 9px;
  border: solid #fff;
  border-width: 0 2px 2px 0;
  opacity: 0;
  transform: rotate(45deg) scale(0.8);
  transition: opacity 0.12s ease, transform 0.12s ease;
}

.work-check-placeholder.is-checked,
.work-row-check.is-checked {
  border-color: var(--copy-blue);
  background: var(--copy-blue);
  box-shadow: 0 0 0 3px rgba(47, 128, 237, 0.16);
}

.work-check-placeholder.is-checked::after,
.work-row-check.is-checked::after {
  opacity: 1;
  transform: rotate(45deg) scale(1);
}

.work-selection-toolbar {
  display: inline-flex;
  align-items: center;
  gap: 14px;
}

.work-selection-action {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 0;
  background: transparent;
  color: #5b6069;
  font-size: 14px;
  font-weight: 850;
  white-space: nowrap;
  cursor: pointer;
}

.work-selection-action.delete {
  color: #ff453d;
}

.work-selection-action .material-symbols-outlined {
  font-size: 18px;
}

.work-sort-btn {
  grid-column: 5 / 7;
  display: inline-flex;
  align-items: center;
  justify-self: start;
  gap: 9px;
  color: #515866;
  font-size: 12px;
  font-weight: 900;
}

.work-sort-btn .material-symbols-outlined {
  font-size: 17px;
}

.work-row-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.work-row-card {
  position: relative;
  min-height: 68px;
  overflow: hidden;
  border-radius: 19px;
  padding: 0 20px;
  border: 1px solid rgba(255, 255, 255, 0.68);
  cursor: pointer;
  box-shadow: 0 18px 42px rgba(24, 28, 35, 0.05);
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.work-row-card.is-selected {
  filter: saturate(1.08);
}

.work-row-card::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 80px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.34), rgba(255, 255, 255, 0));
  pointer-events: none;
}

.work-row-card:hover {
  transform: translateY(-2px);
  filter: brightness(1.01);
  box-shadow: 0 24px 52px rgba(24, 28, 35, 0.08);
}

.work-row-star,
.work-row-more {
  width: 30px;
  height: 30px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #9ba0ab;
  background: rgba(255, 255, 255, 0.42);
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
  position: relative;
  z-index: 1;
}

.work-row-star:hover,
.work-row-more:hover {
  transform: translateY(-1px);
  background: rgba(255, 255, 255, 0.82);
  color: var(--copy-text);
}

.work-row-star.starred {
  color: var(--copy-text);
  background: rgba(255, 255, 255, 0.82);
}

.work-row-star .material-symbols-outlined,
.work-row-more .material-symbols-outlined {
  font-size: 18px;
}

.work-row-icon {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.72);
  color: var(--copy-text);
  position: relative;
  z-index: 1;
}

.work-row-icon .material-symbols-outlined {
  font-size: 18px;
  font-variation-settings: 'FILL' 0;
}

.work-row-title {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  position: relative;
  z-index: 1;
}

.work-row-title strong {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--copy-text);
  font-size: 14px;
  font-weight: 900;
  letter-spacing: -0.02em;
}

.work-row-tag {
  align-self: flex-start;
  height: 18px;
  min-width: 0;
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0 8px;
  background: rgba(255, 255, 255, 0.64);
  font-size: 10px;
  font-weight: 950;
}

.work-row-date {
  color: var(--copy-text);
  font-size: 13px;
  font-weight: 900;
  white-space: nowrap;
  position: relative;
  z-index: 1;
}

@media (max-width: 900px) {
  .work-list-head,
  .work-row-card {
    grid-template-columns: 28px 34px 38px minmax(0, 1fr) 30px;
  }

  .work-row-date,
  .work-sort-btn {
    display: none;
  }

  .work-row-more {
    grid-column: 5;
  }
}

</style>
