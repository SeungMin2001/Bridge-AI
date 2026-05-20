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
  currentTitle: { type: String, default: '' },
  sortType: { type: String, default: 'latest' },
  viewMode: { type: String, default: 'list' }
})

const emit = defineEmits([
  'toggleStar',
  'openItemEditModal',
  'enterFolder',
  'navigate',
  'openFile',
  'goBack',
  'openFolderModal',
  'openFileModal',
  'deleteSelectedItems'
])

const filterType = ref('all')
const selectedIds = ref(new Set())

const filterLabels = {
  all: '모두 보기',
  folder: '폴더만 보기',
  file: '파일만 보기'
}

const parseKoreanDate = (value) => {
  if (!value) return 0
  const directTime = new Date(value).getTime()
  if (!Number.isNaN(directTime)) return directTime

  const normalized = String(value).replace(/\s+/g, ' ').trim()
  const match = normalized.match(/(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(오전|오후)?\s*(\d{1,2})?:?(\d{1,2})?/)
  if (!match) return 0

  const [, year, month, day, meridiem, rawHour = '0', rawMinute = '0'] = match
  let hour = Number(rawHour)
  const minute = Number(rawMinute)
  if (meridiem === '오후' && hour < 12) hour += 12
  if (meridiem === '오전' && hour === 12) hour = 0
  return new Date(Number(year), Number(month) - 1, Number(day), hour, minute).getTime()
}

const getItemTime = (item) => (
  parseKoreanDate(item?.updatedAt || item?.updated_at || item?.createdAt || item?.created_at || item?.date)
)

const getSourceCount = (item) => {
  const materials = Array.isArray(item?.attachments) ? item.attachments.length : 0
  const recordings = Array.isArray(item?.recordings) ? item.recordings.length : 0
  const weekMaterials = Array.isArray(item?.weeks)
    ? item.weeks.reduce((count, week) => count + (Array.isArray(week?.materials) ? week.materials.length : 0), 0)
    : 0
  const weekRecordings = Array.isArray(item?.weeks)
    ? item.weeks.reduce((count, week) => count + (Array.isArray(week?.recordings) ? week.recordings.length : 0), 0)
    : 0

  return Math.max(materials + recordings, weekMaterials + weekRecordings)
}

const filteredItems = computed(() => {
  const items = filterType.value === 'all'
    ? props.currentItems
    : props.currentItems.filter(item => item.type === filterType.value)

  return [...items].sort((a, b) => {
    if (props.sortType === 'title') {
      return String(a?.name || '').localeCompare(String(b?.name || ''), 'ko-KR', {
        numeric: true,
        sensitivity: 'base'
      })
    }

    return getItemTime(b) - getItemTime(a)
  })
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

const handleDeleteSelected = () => {
  const ids = Array.from(selectedIds.value)
  if (!ids.length) return
  if (!confirm(`선택한 보드 ${ids.length}개를 삭제할까요?`)) return
  emit('deleteSelectedItems', ids)
  selectedIds.value = new Set()
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

const getFolderLocation = (item) => {
  if (item.type === 'folder') return '-'
  return item.folderLocationName || item.folderName || item.courseTitle || '기본폴더'
}

const getRowStyle = (item) => ({
  background: '#ffffff',
  '--row-accent': item?.color || '#6366f1',
  '--row-accent-soft': colorWithAlpha(item?.color, item.type === 'folder' ? 0.12 : 0.14),
})

</script>

<template>
  <div class="flex-1 mt-0"> 
    <div class="shrink-0">
      <div class="grid-header flex items-center">
        <div class="work-title-row">
          <button 
            v-if="navigationStack.length > 0"
            @click="emit('goBack')"
            class="flex items-center justify-center p-2 bg-white border-none text-[#1d1d1f] cursor-pointer rounded-xl hover:bg-[#f2f2f7] transition-colors shadow-sm"
          >
            <span class="material-symbols-outlined text-[20px]">arrow_back</span>
          </button>
          <span class="section-title !m-0 transition-all duration-300">{{ currentTitle }}</span>
        </div>
      </div>
      
      <div v-if="props.viewMode === 'list'" class="work-list-shell">
        <div class="work-list-head">
          <div class="work-head-main">
            <button
              type="button"
              :class="['work-check-placeholder', { 'is-checked': allVisibleSelected }]"
              aria-label="전체 선택"
              @click="toggleAllVisible"
            ></button>
            <span>파일 이름</span>
            <div v-if="selectedCount" class="work-selection-toolbar">
              <button class="work-selection-action delete" type="button" @click="handleDeleteSelected">
                <span class="material-symbols-outlined">delete</span>
                삭제하기
              </button>
              <button class="work-selection-action" type="button">
                <span class="material-symbols-outlined">drive_file_move</span>
                이동하기
              </button>
            </div>
          </div>
          <div class="work-source-head">소스</div>
          <div class="work-folder-head">폴더 위치</div>
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
            <div class="work-row-source">소스 {{ getSourceCount(item) }}개</div>
            <div class="work-row-folder-location">
              <span class="material-symbols-outlined">folder</span>
              <span>{{ getFolderLocation(item) }}</span>
            </div>
            <time class="work-row-date">{{ item.date || '-' }}</time>
            <button class="work-row-more" title="설정" type="button" @click.stop="emit('openItemEditModal', item.id)">
              <span class="material-symbols-outlined">more_horiz</span>
            </button>
          </article>
        </template>
        </div>
      </div>

      <div v-else class="work-card-grid">
        <button
          class="work-grid-create-card"
          type="button"
          @click="emit('openFileModal')"
        >
          <span class="work-grid-create-icon">
            <span class="material-symbols-outlined">add</span>
          </span>
          <span class="work-grid-create-label">새 파일 만들기</span>
        </button>

        <article
          v-for="item in filteredItems"
          :key="item.id"
          class="work-grid-card"
          :style="{ '--grid-card-accent': item.color || '#6366f1', '--grid-card-accent-soft': colorWithAlpha(item.color, 0.13) }"
          @click="openItem($event, item)"
        >
          <button
            :class="['work-grid-star', { starred: favorites.has(item.id) }]"
            type="button"
            aria-label="즐겨찾기"
            @click.stop="emit('toggleStar', $event, item.id)"
          >
            <span class="material-symbols-outlined" :style="{ fontVariationSettings: `'FILL' ${favorites.has(item.id) ? 1 : 0}` }">star</span>
          </button>
          <button
            class="work-grid-more"
            title="설정"
            type="button"
            @click.stop="emit('openItemEditModal', item.id)"
          >
            <span class="material-symbols-outlined">more_vert</span>
          </button>
          <div class="work-grid-icon">
            <span class="material-symbols-outlined">{{ getItemIcon(item) }}</span>
          </div>
          <div class="work-grid-body">
            <strong>{{ item.name }}</strong>
            <span>{{ item.date || '-' }} · 소스 {{ getSourceCount(item) }}개</span>
          </div>
          <div class="work-grid-tag" :style="{ color: item.color || '#6366f1' }">
            {{ getItemTag(item) }}
          </div>
        </article>
      </div>
    </div>

    <div class="h-[60px] shrink-0"></div>
  </div>
</template>

<style scoped>
.grid-header {
  padding-right: 430px;
}

.work-title-row {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 18px;
  flex-wrap: wrap;
}

.work-view-toolbar {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.filter-dropdown-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.sort-dropdown-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.filter-trigger-btn {
  height: 42px;
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.05);
  border-radius: 12px;
  padding: 0 16px;
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

.sort-trigger-btn {
  height: 42px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0 16px;
  border: 1px solid rgba(25, 25, 31, 0.12);
  border-radius: 999px;
  background: #fff;
  color: #1d1d1f;
  box-shadow: 0 10px 22px rgba(31, 34, 43, 0.05);
  font-size: 13.5px;
  font-weight: 850;
  cursor: pointer;
  transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.sort-trigger-btn:hover,
.sort-trigger-btn.active {
  background: #f8f8fb;
  border-color: rgba(25, 25, 31, 0.26);
  box-shadow: 0 12px 26px rgba(31, 34, 43, 0.08);
}

.work-view-segment {
  height: 42px;
  display: inline-flex;
  align-items: center;
  padding: 3px;
  border: 1px solid rgba(25, 25, 31, 0.14);
  border-radius: 999px;
  background: #fff;
  box-shadow: 0 10px 22px rgba(31, 34, 43, 0.05);
}

.work-view-mode-btn {
  width: 36px;
  height: 34px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: #5f6571;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.work-view-mode-btn:hover {
  color: #1d1d1f;
  transform: translateY(-1px);
}

.work-view-mode-btn.active {
  background: #edf0fb;
  color: #1d1d1f;
}

.work-view-mode-btn .material-symbols-outlined {
  font-size: 21px;
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

.sort-menu {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 142px;
  padding: 6px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 16px;
  background: #fff;
  z-index: 100;
  transform-origin: top right;
}

.filter-item,
.sort-item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 0;
  background: transparent;
  font-size: 13px;
  font-weight: 600;
  color: #3a3a3c;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.filter-item:hover,
.sort-item:hover {
  background: #f2f2f7;
  color: #1d1d1f;
}

.filter-item.selected,
.sort-item.selected {
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

.work-card-grid {
  --work-grid-card-height: 210px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 260px));
  grid-auto-rows: var(--work-grid-card-height);
  align-items: start;
  justify-content: center;
  gap: 14px;
}

.work-grid-create-card,
.work-grid-card {
  height: var(--work-grid-card-height);
  min-height: var(--work-grid-card-height);
  box-sizing: border-box;
}

.work-grid-create-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 18px;
  padding: 22px 20px 18px;
  border: 1px solid rgba(207, 215, 229, 0.88);
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 18px 42px rgba(24, 28, 35, 0.035);
  color: var(--copy-text);
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.work-grid-create-card:hover {
  transform: translateY(-3px);
  border-color: rgba(47, 128, 237, 0.36);
  box-shadow: 0 24px 56px rgba(24, 28, 35, 0.07);
}

.work-grid-create-icon {
  width: 56px;
  height: 56px;
  display: grid;
  place-items: center;
  border-radius: 999px;
  background: #eef1ff;
  color: #355cff;
}

.work-grid-create-icon .material-symbols-outlined {
  font-size: 24px;
}

.work-grid-create-label {
  font-size: 18px;
  font-weight: 850;
  letter-spacing: 0;
}

.work-grid-card {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  overflow: hidden;
  padding: 22px 20px 18px;
  border: 1px solid rgba(226, 232, 240, 0.76);
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.62), rgba(255, 255, 255, 0.26)),
    var(--grid-card-accent-soft);
  box-shadow: 0 18px 42px rgba(24, 28, 35, 0.05);
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.work-grid-card:hover {
  transform: translateY(-3px);
  border-color: color-mix(in srgb, var(--grid-card-accent) 32%, #d8deea);
  box-shadow: 0 24px 56px rgba(24, 28, 35, 0.09);
}

.work-grid-star,
.work-grid-more {
  position: absolute;
  top: 14px;
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: #7d8490;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.work-grid-star {
  right: 48px;
}

.work-grid-more {
  right: 14px;
}

.work-grid-star:hover,
.work-grid-more:hover {
  transform: translateY(-1px);
  background: transparent;
  color: #1d1d1f;
}

.work-grid-star.starred {
  color: #f4b400;
  background: transparent;
}

.work-grid-star .material-symbols-outlined,
.work-grid-more .material-symbols-outlined {
  font-size: 18px;
}

.work-grid-icon {
  width: 48px;
  height: 48px;
  display: grid;
  place-items: center;
  margin-bottom: auto;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.68);
  color: var(--grid-card-accent);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.58);
}

.work-grid-icon .material-symbols-outlined {
  font-size: 26px;
  font-variation-settings: 'FILL' 0;
}

.work-grid-body {
  min-width: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 34px;
}

.work-grid-body strong {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--copy-text);
  font-size: 20px;
  font-weight: 850;
  letter-spacing: 0;
}

.work-grid-body span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #686f7b;
  font-size: 13px;
  font-weight: 700;
}

.work-grid-tag {
  max-width: 100%;
  height: 22px;
  display: inline-flex;
  align-items: center;
  margin-top: 14px;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.62);
  font-size: 11px;
  font-weight: 900;
}

.work-list-shell {
  width: 100%;
  margin: 0 auto;
}

.work-list-head,
.work-row-card {
  display: grid;
  grid-template-columns: 30px 30px 36px minmax(0, 1fr) 112px 170px 230px 30px;
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

.work-folder-head {
  grid-column: 6;
  justify-self: start;
  color: #515866;
  font-size: 12px;
  font-weight: 900;
}

.work-source-head {
  grid-column: 5;
  justify-self: start;
  color: #515866;
  font-size: 12px;
  font-weight: 900;
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
  grid-column: 7 / 9;
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
  border: 1px solid rgba(226, 232, 240, 0.78);
  cursor: pointer;
  box-shadow: 0 18px 42px rgba(24, 28, 35, 0.05);
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.work-row-card.is-selected {
  border-color: rgba(148, 163, 184, 0.48);
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
  color: #f4b400;
  background: rgba(255, 255, 255, 0.88);
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
  background: var(--row-accent-soft);
  color: var(--row-accent);
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

.work-row-folder-location {
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 9px;
  color: #6f7682;
  font-size: 13px;
  font-weight: 850;
  white-space: nowrap;
  position: relative;
  z-index: 1;
}

.work-row-source {
  color: #6f7682;
  font-size: 13px;
  font-weight: 850;
  white-space: nowrap;
  position: relative;
  z-index: 1;
}

.work-row-folder-location .material-symbols-outlined {
  flex: 0 0 auto;
  color: #b6bac2;
  font-size: 19px;
  font-variation-settings: 'FILL' 1;
}

.work-row-folder-location span:last-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
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
  .grid-header {
    padding-right: 0;
  }

  .work-title-row {
    align-items: flex-start;
    flex-direction: column;
    gap: 14px;
  }

  .work-view-toolbar {
    width: 100%;
    flex-wrap: wrap;
  }

  .work-card-grid {
    grid-template-columns: repeat(auto-fit, minmax(180px, 260px));
  }

  .work-list-head,
  .work-row-card {
    grid-template-columns: 28px 34px 38px minmax(0, 1fr) 30px;
  }

  .work-folder-head,
  .work-source-head,
  .work-row-source,
  .work-row-folder-location,
  .work-row-date,
  .work-sort-btn {
    display: none;
  }

  .work-row-more {
    grid-column: 5;
  }
}

</style>
