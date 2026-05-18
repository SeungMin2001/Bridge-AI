<!-- 사용자의 폴더 구조를 관리하고 파일들을 탐색할 수 있는 워크폴더 페이지 컴포넌트입니다. -->
<script setup>
import { computed, nextTick, ref } from 'vue'
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
  deleteItemById,
  deleteItemsByIds,
  handleEnterFolder,
  handleGoBack
} = useHome(props, emit)

const workViewMode = ref('all')
const isCreatingFolder = ref(false)
const isFolderNameComposing = ref(false)
const shouldCommitFolderAfterComposition = ref(false)
const draftFolderName = ref('')
const folderCreateInputRef = ref(null)
const openFolderMenuId = ref(null)

const isDefaultFolder = (item) => item?.isDefaultFolder || item?.name === '기본폴더' || item?.name === '기본파일'
const getFolderDisplayName = (folder) => isDefaultFolder(folder) ? '기본폴더' : folder.name

const withFolderLocation = (item, folderName = '') => (
  item?.type === 'file'
    ? { ...item, folderLocationName: folderName || item.folderLocationName || '기본폴더' }
    : item
)

const flattenItems = (nodes = [], parentFolderName = '') => nodes.flatMap((node) => {
  const currentFolderName = node.type === 'folder' ? getFolderDisplayName(node) : parentFolderName
  return [
    withFolderLocation(node, parentFolderName),
    ...flattenItems(node.children || [], currentFolderName)
  ]
})

const attachCurrentFolderLocation = (items = [], folderName = '') => (
  items.map((item) => withFolderLocation(item, folderName))
)

const allFiles = computed(() => flattenItems(props.fileTree).filter((item) => item.type === 'file'))
const favoriteItems = computed(() => flattenItems(props.fileTree).filter((item) => props.favorites.has(item.id)))

const currentTitle = computed(() => {
  if (navigationStack.value.length > 0) return navigationStack.value[navigationStack.value.length - 1].name
  return workViewMode.value === 'favorites' ? '즐겨찾기' : '전체 파일'
})

const currentItems = computed(() => {
  if (navigationStack.value.length === 0) {
    return workViewMode.value === 'favorites' ? favoriteItems.value : allFiles.value
  }
  
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
  return folder ? attachCurrentFolderLocation(folder.children || [], getFolderDisplayName(folder)) : []
})

const rootFolders = computed(() => (
  props.fileTree
    .filter((item) => item.type === 'folder')
    .sort((a, b) => Number(isDefaultFolder(b)) - Number(isDefaultFolder(a)))
))
const sidebarFolderEntries = computed(() => {
  const defaultFolder = rootFolders.value.find((folder) => isDefaultFolder(folder))
  const userFolders = rootFolders.value.filter((folder) => !isDefaultFolder(folder))
  const entries = []

  if (defaultFolder) {
    entries.push({ type: 'folder', key: defaultFolder.id, folder: defaultFolder })
  }

  if (isCreatingFolder.value) {
    entries.push({ type: 'create', key: '__folder-create' })
  }

  userFolders.forEach((folder) => entries.push({ type: 'folder', key: folder.id, folder }))
  return entries
})
const activeFolderId = computed(() => navigationStack.value.at(-1)?.id || null)

const showAllFiles = () => {
  navigationStack.value = []
  workViewMode.value = 'all'
}

const showFavorites = () => {
  navigationStack.value = []
  workViewMode.value = 'favorites'
}

const enterFolderFromSidebar = (event, folder) => {
  event.stopPropagation()
  workViewMode.value = 'folder'
  openFolderMenuId.value = null
  navigationStack.value = [{ id: folder.id, name: getFolderDisplayName(folder) }]
}

const startInlineFolderCreate = async () => {
  isCreatingFolder.value = true
  draftFolderName.value = ''
  await nextTick()
  folderCreateInputRef.value?.focus()
}

const cancelInlineFolderCreate = () => {
  isCreatingFolder.value = false
  isFolderNameComposing.value = false
  shouldCommitFolderAfterComposition.value = false
  draftFolderName.value = ''
}

const commitInlineFolderCreate = async (event) => {
  if (!isCreatingFolder.value) return

  await nextTick()
  const folderName = (event?.target?.value || draftFolderName.value).trim()
  isCreatingFolder.value = false
  isFolderNameComposing.value = false
  shouldCommitFolderAfterComposition.value = false
  draftFolderName.value = ''
  if (!folderName) return

  const previousStack = [...navigationStack.value]
  navigationStack.value = []
  newFolderName.value = folderName
  selectedColor.value = FOLDER_COLORS[0]
  try {
    await handleCreateFolder()
  } finally {
    navigationStack.value = previousStack
  }
}

const handleFolderNameCompositionStart = () => {
  isFolderNameComposing.value = true
}

const handleFolderNameCompositionEnd = async (event) => {
  isFolderNameComposing.value = false
  draftFolderName.value = event.target.value

  if (shouldCommitFolderAfterComposition.value) {
    await commitInlineFolderCreate(event)
  }
}

const handleFolderNameEnter = async (event) => {
  if (event.isComposing || event.keyCode === 229 || isFolderNameComposing.value) {
    shouldCommitFolderAfterComposition.value = true
    return
  }

  event.preventDefault()
  await commitInlineFolderCreate(event)
}

const handleFolderNameBlur = async (event) => {
  if (isFolderNameComposing.value) return
  await commitInlineFolderCreate(event)
}

const toggleFolderMenu = (event, folderId) => {
  event.stopPropagation()
  openFolderMenuId.value = openFolderMenuId.value === folderId ? null : folderId
}

const openFolderRename = (folder) => {
  openFolderMenuId.value = null
  openItemEditModal(folder.id)
}

const deleteFolderFromSidebar = async (folder) => {
  openFolderMenuId.value = null
  if (activeFolderId.value === folder.id) {
    navigationStack.value = []
    workViewMode.value = 'all'
  }
  await deleteItemById(folder.id)
}
</script>

<template>
  <div class="copy-app-frame flex relative h-full w-full text-[#1e293b] overflow-hidden" @click="openFolderMenuId = null">
    <InfiniteGrid />
    <HomeSidebar 
      class="relative z-10"
      :isCollapsed="isSidebarCollapsed"
      activeView="workfolder"
      :fileTree="fileTree"
      :favorites="favorites"
      @toggle="isSidebarCollapsed = !isSidebarCollapsed"
      @navigate="emit('navigate', $event)"
      @openFileCreate="openFileCreateModal"
    />

    <aside class="copy-work-sidebar relative z-10">
      <button class="copy-collapse-btn" type="button" aria-label="사이드바 접기">
        <span class="material-symbols-outlined">keyboard_double_arrow_left</span>
      </button>

      <div class="copy-work-inner">
        <button
          :class="['copy-work-item', { 'is-selected': workViewMode === 'all' && navigationStack.length === 0 }]"
          type="button"
          @click="showAllFiles"
        >
          <span class="material-symbols-outlined">view_sidebar</span>
          <span>전체 파일</span>
        </button>
        <button
          :class="['copy-work-item', { 'is-selected': workViewMode === 'favorites' && navigationStack.length === 0 }]"
          type="button"
          @click="showFavorites"
        >
          <span class="material-symbols-outlined">star</span>
          <span>즐겨찾기</span>
        </button>

        <div class="copy-work-section-row">
          <span>폴더</span>
          <button class="copy-folder-add" type="button" @click="startInlineFolderCreate">
            <span class="material-symbols-outlined">add</span>
          </button>
        </div>

        <div class="copy-folder-list">
          <template v-for="entry in sidebarFolderEntries" :key="entry.key">
            <div v-if="entry.type === 'create'" class="copy-folder-create-row">
              <span class="material-symbols-outlined">folder</span>
              <input
                ref="folderCreateInputRef"
                v-model="draftFolderName"
                type="text"
                aria-label="새 폴더 이름"
                @compositionstart="handleFolderNameCompositionStart"
                @compositionend="handleFolderNameCompositionEnd"
                @keydown.enter="handleFolderNameEnter"
                @keydown.esc.prevent="cancelInlineFolderCreate"
                @blur="handleFolderNameBlur"
              />
            </div>
            <div
              v-else
              :class="[
                'copy-folder-item',
                {
                  'is-selected': activeFolderId === entry.folder.id,
                  'has-menu': openFolderMenuId === entry.folder.id
                }
              ]"
              role="button"
              tabindex="0"
              @click="enterFolderFromSidebar($event, entry.folder)"
              @keydown.enter.prevent="enterFolderFromSidebar($event, entry.folder)"
            >
              <span class="material-symbols-outlined">folder</span>
              <span class="copy-folder-name">{{ getFolderDisplayName(entry.folder) }}</span>
              <button
                v-if="!isDefaultFolder(entry.folder)"
                class="copy-folder-more"
                type="button"
                aria-label="폴더 메뉴"
                @click="toggleFolderMenu($event, entry.folder.id)"
              >
                <span class="material-symbols-outlined">more_horiz</span>
              </button>
              <div v-if="openFolderMenuId === entry.folder.id" class="copy-folder-menu" @click.stop>
                <button class="copy-folder-menu-action" type="button" @click="openFolderRename(entry.folder)">
                  <span class="material-symbols-outlined">edit</span>
                  <span>이름 변경하기</span>
                </button>
                <button class="copy-folder-menu-action is-danger" type="button" @click="deleteFolderFromSidebar(entry.folder)">
                  <span class="material-symbols-outlined">delete</span>
                  <span>삭제하기</span>
                </button>
              </div>
            </div>
          </template>
        </div>
      </div>
    </aside>

    <main id="home-main-content" class="work-main-shell custom-scrollbar flex-1 overflow-y-auto relative z-10">
      <div class="copy-work-topbar">
        <button class="copy-new-file-btn" type="button" @click="openFileCreateModal">
          <span class="material-symbols-outlined">add</span>
          <span>새 파일</span>
        </button>
        <button class="copy-bell-btn" type="button" aria-label="알림">
          <span class="material-symbols-outlined">notifications</span>
        </button>
      </div>

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
        @deleteSelectedItems="deleteItemsByIds"
        @navigate="emit('navigate', $event)"
        @openFile="(item) => { emit('fileSelect', item.id, item); emit('navigate', 'workspace') }"
      />
    </main>

    <HomeModals 
      placement="work-top"
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
<style scoped>
.copy-app-frame {
  background: var(--copy-bg);
}

#home-main-content.work-main-shell {
  min-width: 0;
  height: calc(100% - 20px);
  max-height: calc(100% - 20px);
  min-height: calc(100% - 20px);
  align-self: flex-start;
  margin: 10px 12px 10px 0;
  box-sizing: border-box;
  padding: 34px 58px 0;
  border-radius: 32px;
  background: #fff;
  border: 0;
  box-shadow: -10px 0 34px rgba(48, 42, 58, 0.05);
}

.copy-work-topbar {
  position: absolute;
  top: 34px;
  right: 34px;
  display: inline-flex;
  align-items: center;
  gap: 14px;
}

.copy-new-file-btn {
  height: 44px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0 19px;
  border: 0;
  border-radius: 15px;
  background: var(--copy-black);
  color: #fff;
  box-shadow: 0 14px 28px rgba(21, 22, 26, 0.14);
  font-size: 14px;
  font-weight: 900;
}

.copy-new-file-btn .material-symbols-outlined {
  font-size: 20px;
}

.copy-bell-btn {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 19px;
  background: #fff;
  color: var(--copy-black);
  box-shadow: 0 10px 24px rgba(48, 42, 58, 0.05);
}

.copy-bell-btn .material-symbols-outlined {
  font-size: 20px;
}

.copy-app-frame > :deep(.infinite-grid-container) {
  opacity: 0;
}

.copy-work-sidebar {
  position: relative;
  z-index: 30;
  width: 260px;
  min-width: 260px;
  height: 100vh;
  background: var(--copy-bg);
  color: var(--copy-text);
}

.copy-collapse-btn {
  position: absolute;
  top: 34px;
  right: 20px;
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border: 0;
  background: transparent;
  color: var(--copy-black);
}

.copy-collapse-btn .material-symbols-outlined {
  font-size: 20px;
}

.copy-work-inner {
  height: 100%;
  padding: 82px 18px 28px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.copy-work-item,
.copy-folder-item {
  position: relative;
  width: 100%;
  height: 44px;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 0 12px;
  border: 0;
  border-radius: 18px;
  background: transparent;
  color: #1f2026;
  font-size: 14px;
  font-weight: 850;
  text-align: left;
  cursor: pointer;
  transition: background 0.18s ease, box-shadow 0.18s ease;
}

.copy-work-item:hover,
.copy-work-item.is-selected,
.copy-folder-item:hover,
.copy-folder-item.is-selected {
  background: #fff;
  box-shadow: 0 14px 28px rgba(48, 42, 58, 0.08);
}

.copy-work-item .material-symbols-outlined,
.copy-folder-item .material-symbols-outlined {
  font-size: 20px;
}

.copy-folder-name {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.copy-folder-more {
  flex: 0 0 auto;
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  margin-left: auto;
  border: 0;
  border-radius: 12px;
  background: rgba(227, 228, 234, 0.78);
  color: #5d626c;
  opacity: 0;
  transform: scale(0.94);
  transition: opacity 0.16s ease, transform 0.16s ease, background 0.16s ease;
}

.copy-folder-more .material-symbols-outlined {
  font-size: 19px;
}

.copy-folder-item:hover .copy-folder-more,
.copy-folder-item.is-selected .copy-folder-more,
.copy-folder-item.has-menu .copy-folder-more {
  opacity: 1;
  transform: scale(1);
}

.copy-folder-item.has-menu .copy-folder-more {
  background: #dfe0e6;
}

.copy-folder-menu {
  position: absolute;
  top: 4px;
  left: calc(100% + 10px);
  z-index: 60;
  width: 166px;
  padding: 8px;
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 18px 42px rgba(35, 31, 45, 0.14);
}

.copy-folder-menu::before {
  content: '';
  position: absolute;
  left: -6px;
  top: 18px;
  width: 14px;
  height: 14px;
  border-radius: 3px;
  background: #fff;
  transform: rotate(45deg);
}

.copy-folder-menu-action {
  position: relative;
  z-index: 1;
  width: 100%;
  height: 38px;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 0 12px;
  border: 0;
  border-radius: 12px;
  background: transparent;
  color: #515866;
  font-size: 13px;
  font-weight: 850;
  text-align: left;
  white-space: nowrap;
}

.copy-folder-menu-action:hover {
  background: #f3f3f6;
  color: var(--copy-text);
}

.copy-folder-menu-action.is-danger {
  color: #ff453d;
}

.copy-folder-menu-action .material-symbols-outlined {
  font-size: 18px;
}

.copy-work-section-row {
  margin-top: 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: rgba(21, 22, 26, 0.45);
  font-size: 13px;
  font-weight: 900;
}

.copy-folder-add {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 13px;
  background: #fff;
  color: var(--copy-black);
  box-shadow: 0 10px 22px rgba(48, 42, 58, 0.07);
}

.copy-folder-add .material-symbols-outlined {
  font-size: 20px;
}

.copy-folder-list {
  display: grid;
  gap: 8px;
  min-width: 0;
}

.copy-folder-create-row {
  width: 100%;
  max-width: 100%;
  height: 44px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 12px;
  border: 0;
  border-radius: 18px;
  background: transparent;
  color: var(--copy-text);
  overflow: hidden;
  box-shadow: none;
}

.copy-folder-create-row .material-symbols-outlined {
  flex: 0 0 auto;
  font-size: 20px;
}

.copy-folder-create-row input {
  min-width: 0;
  flex: 1;
  width: 100%;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--copy-text);
  font-size: 14px;
  font-weight: 850;
  font-family: inherit;
  caret-color: var(--copy-text);
  appearance: none;
  box-shadow: none;
}

.copy-folder-create-row input:focus,
.copy-folder-create-row input:focus-visible {
  outline: none;
  box-shadow: none;
}
</style>
