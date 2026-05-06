<!-- 워크스페이스 왼쪽에서 파일 구조를 탐색하고 파일을 선택할 수 있게 돕는 폴더 탐색기 탭입니다. -->
<script setup>
import { ref, computed } from 'vue'
import {
  deleteWorkspaceFile,
  deleteWorkspaceFolder,
  deleteWorkspaceRecordingData,
  isWorkspaceUuid,
  saveSessionResources
} from '../../api/workspaceApi.js'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  activeFileId: { type: String, default: '' }
})

const emit = defineEmits([
  'update:fileTree',
  'update:favorites',
  'fileSelect',
  'openMaterial',
  'openRecording',
  'showToast'
])

// --- 헬퍼 함수 ---
function findNode(id, nodes) {
  for (const n of nodes) {
    if (n.id === id) return n
    if (n.children) {
      const found = findNode(id, n.children)
      if (found) return found
    }
  }
  return null
}

function deleteNode(id, nodes) {
  const idx = nodes.findIndex(n => n.id === id)
  if (idx !== -1) { nodes.splice(idx, 1); return true }
  for (const n of nodes) {
    if (n.children && deleteNode(id, n.children)) return true
  }
  return false
}

function flattenAll(nodes) {
  const result = []
  for (const n of nodes) {
    result.push(n)
    if (n.children) result.push(...flattenAll(n.children))
  }
  return result
}

function genId() { return 'n' + Date.now() + Math.random().toString(36).slice(2, 6) }

const FOLDER_COLORS = ['#3b82f6', '#5856d6', '#ff9500', '#34c759', '#ff3b30', '#af52de']
let colorIdx = 0

// --- 상태 ---
const localActiveFileId = ref(null)
const searchQuery = ref('')
const ctxMenu = ref({ visible: false, x: 0, y: 0, targetId: null })
const materialMenu = ref({ visible: false, x: 0, y: 0, fileId: null, weekId: null, materialId: null, recordingId: null })

// --- Computed ---
const selectedFileId = computed(() => props.activeFileId || localActiveFileId.value)
const activeNode = computed(() => findNode(selectedFileId.value, props.fileTree))
const scopedTree = computed(() => activeNode.value ? [activeNode.value] : props.fileTree)
const allFlat = computed(() => flattenAll(scopedTree.value))
const searchResults = computed(() => {
  if (!searchQuery.value.trim()) return null
  return allFlat.value.filter(n => n.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
})
const ctxTargetNode = computed(() => ctxMenu.value.targetId ? findNode(ctxMenu.value.targetId, props.fileTree) : null)
const isCtxFavorite = computed(() => ctxMenu.value.targetId ? props.favorites.has(ctxMenu.value.targetId) : false)
const materialMenuTarget = computed(() => {
  const node = materialMenu.value.fileId ? findNode(materialMenu.value.fileId, props.fileTree) : null
  const week = Array.isArray(node?.weeks) ? node.weeks.find((item) => item.id === materialMenu.value.weekId) : null
  const material = Array.isArray(week?.materials) ? week.materials.find((item) => item.id === materialMenu.value.materialId) : null
  const recording = Array.isArray(week?.recordings) ? week.recordings.find((item) => item.id === materialMenu.value.recordingId) : null
  return { node, week, material, recording }
})

const getNodeIcon = (node) => {
  if (!node) return 'description'
  if (node.type === 'folder') return 'folder'
  return node.fileKind === 'meeting' ? 'groups_2' : 'description'
}

const getNodeIconStyle = (node, isSelected = false) => {
  if (!node) return {}
  if (node.type === 'folder') {
    return { color: node.color || '#8e8e93', fontSize: '18px', fontVariationSettings: "'FILL' 1" }
  }

  if (node.fileKind === 'meeting') {
    return { color: isSelected ? '#be185d' : (node.color || '#ec4899'), fontSize: '18px', fontVariationSettings: "'FILL' 1" }
  }

  return { color: isSelected ? '#1d1d1f' : (node.color || '#8e8e93'), fontSize: '18px', fontVariationSettings: "'FILL' 0" }
}

// --- 핸들러 ---
const handleSelectFile = (id) => {
  localActiveFileId.value = id
  emit('fileSelect', id, findNode(id, props.fileTree))
}

const handleOpenMaterial = (payload) => {
  emit('openMaterial', payload)
}

const handleShowMaterialMenu = ({ fileId, weekId, materialId, recordingId }, x, y) => {
  const menuWidth = 190
  const menuHeight = 112
  const nextX = Math.min(x, window.innerWidth - menuWidth - 16)
  const nextY = Math.min(y, window.innerHeight - menuHeight - 16)
  materialMenu.value = { visible: true, x: nextX, y: nextY, fileId, weekId, materialId, recordingId }
}

const handleCloseMaterialMenu = () => {
  materialMenu.value = { visible: false, x: 0, y: 0, fileId: null, weekId: null, materialId: null, recordingId: null }
}

const handleMaterialAction = async (action) => {
  const { fileId, weekId, materialId, recordingId } = materialMenu.value
  const { material, recording } = materialMenuTarget.value
  handleCloseMaterialMenu()
  if (!fileId || !weekId) return
  if (!materialId && !recordingId) return

  const copy = JSON.parse(JSON.stringify(props.fileTree))
  const node = findNode(fileId, copy)
  const week = Array.isArray(node?.weeks) ? node.weeks.find((item) => item.id === weekId) : null
  if (!node || !week) return

  if (materialId && material && Array.isArray(week.materials)) {
    if (action === 'rename') {
      const nextName = prompt('새 파일 이름을 입력하세요:', material.name)
      if (!nextName?.trim()) return

      week.materials = week.materials.map((item) => (
        item.id === materialId ? { ...item, name: nextName.trim() } : item
      ))
      node.attachments = Array.isArray(node.attachments)
        ? node.attachments.map((item) => (item.id === materialId ? { ...item, name: nextName.trim() } : item))
        : node.attachments
      emit('showToast', `"${nextName.trim()}"으로 이름 변경됨`)
    }

    if (action === 'delete') {
      if (!confirm(`"${material.name}" 파일을 삭제할까요?`)) return

      week.materials = week.materials.filter((item) => item.id !== materialId)
      node.attachments = Array.isArray(node.attachments)
        ? node.attachments.filter((item) => item.id !== materialId)
        : node.attachments
      emit('showToast', `"${material.name}" 삭제됨`)
    }
  } else if (recordingId && recording && Array.isArray(week.recordings)) {
    if (action === 'rename') {
      const nextName = prompt('새 녹음본 이름을 입력하세요:', recording.title || '녹음본')
      if (!nextName?.trim()) return

      week.recordings = week.recordings.map((item) => (
        item.id === recordingId ? { ...item, title: nextName.trim() } : item
      ))
      node.recordings = Array.isArray(node.recordings)
        ? node.recordings.map((item) => (item.id === recordingId ? { ...item, title: nextName.trim() } : item))
        : node.recordings
      emit('showToast', `"${nextName.trim()}"으로 이름 변경됨`)
    }

    if (action === 'delete') {
      if (!confirm(`"${recording.title || '녹음본'}"을(를) 삭제할까요?`)) return

      week.recordings = week.recordings.filter((item) => item.id !== recordingId)
      node.recordings = Array.isArray(node.recordings)
        ? node.recordings.filter((item) => item.id !== recordingId)
        : node.recordings
      if (isWorkspaceUuid(fileId)) {
        try {
          await deleteWorkspaceRecordingData(fileId, recordingId)
        } catch (error) {
          console.error('[workspace] recording related data cleanup failed:', error)
          emit('showToast', '녹음본 관련 데이터 정리 실패')
        }
      }
      emit('showToast', `녹음본 삭제됨`)
    }
  }

  emit('update:fileTree', copy)

  if (isWorkspaceUuid(fileId) && Array.isArray(node.weeks)) {
    try {
      await saveSessionResources(fileId, node.weeks)
    } catch (error) {
      console.error('[workspace] session resources save failed:', error)
      emit('showToast', 'DB 저장 실패')
    }
  }
}

const handleOpenRecording = (payload) => {
  emit('openRecording', payload)
}

const handleToggleFolder = (id) => {
  const copy = JSON.parse(JSON.stringify(props.fileTree))
  const node = findNode(id, copy)
  if (node) node.expanded = !node.expanded
  emit('update:fileTree', copy)
}

const handleShowContextMenu = (id, x, y) => {
  const menuWidth = 236
  const menuHeight = 308
  const nextX = Math.min(x, window.innerWidth - menuWidth - 16)
  const nextY = Math.min(y, window.innerHeight - menuHeight - 16)
  ctxMenu.value = { visible: true, x: nextX, y: nextY, targetId: id }
}

const handleCloseContextMenu = () => {
  ctxMenu.value = { visible: false, x: 0, y: 0, targetId: null }
}

const handleContextAction = async (action) => {
  const targetId = ctxMenu.value.targetId
  handleCloseContextMenu()

  const copy = JSON.parse(JSON.stringify(props.fileTree))
  const node = findNode(targetId, copy)
  if (!node) return

  switch (action) {
    case 'open':
      if (node.type === 'file') handleSelectFile(targetId)
      else node.expanded = true
      break
    case 'rename': {
      const newName = prompt('새 이름을 입력하세요:', node.name)
      if (newName && newName.trim()) {
        node.name = newName.trim()
        emit('showToast', `"${newName.trim()}"으로 이름 변경됨`)
      }
      break
    }
    case 'new-file': {
      if (node.type !== 'folder') break
      const newFile = { id: genId(), type: 'file', fileKind: 'lecture', name: '새 파일', content: '' }
      if (!node.children) node.children = []
      node.children.push(newFile)
      node.expanded = true
      emit('showToast', '새 파일이 추가되었습니다')
      break
    }
    case 'new-folder': {
      if (node.type !== 'folder') break
      const color = FOLDER_COLORS[colorIdx++ % FOLDER_COLORS.length]
      const newFolder = { id: genId(), type: 'folder', name: '새 폴더', color, expanded: false, children: [] }
      if (!node.children) node.children = []
      node.children.push(newFolder)
      node.expanded = true
      emit('showToast', '새 폴더가 추가되었습니다')
      break
    }
    case 'favorite': {
      const nextFavs = new Set(props.favorites)
      if (nextFavs.has(targetId)) {
        nextFavs.delete(targetId)
        emit('showToast', '즐겨찾기에서 제거됨')
      } else {
        nextFavs.add(targetId)
        emit('showToast', '즐겨찾기에 추가됨')
      }
      emit('update:favorites', nextFavs)
      return // favorite은 fileTree 변경 불필요
    }
    case 'delete':
      // 신창영 : 워크스페이스 UI에서 삭제할 때도 백엔드 삭제 API를 호출하여 DB/RAG 찌꺼기 생성을 방지
      if (isWorkspaceUuid(targetId)) {
        try {
          if (node.type === 'folder') {
            await deleteWorkspaceFolder(targetId)
          } else {
            await deleteWorkspaceFile(targetId)
          }
        } catch (error) {
          console.error('[workspace] node delete failed:', error)
          emit('showToast', 'DB 삭제 실패')
          return
        }
      }
      deleteNode(targetId, copy)
      emit('showToast', `"${node.name}" 삭제됨`)
      break
  }
  emit('update:fileTree', copy)
}

</script>

<template>
  <div class="flex flex-col flex-1 overflow-hidden">
    <!-- 검색 창 -->
    <div class="sidebar-search-bg workspace-inset-shell rounded-[24px] px-3 py-2.5 flex items-center gap-3 mb-6">
      <span class="material-symbols-outlined text-[#8e8e93] text-[20px]">search</span>
      <input
        class="bg-transparent border-none focus:ring-0 p-0 text-[14px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
        placeholder="제목으로 검색"
        type="text"
        v-model="searchQuery"
      />
    </div>

    <div class="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
      <!-- 현재 파일 구조 -->
      <section>
        <div class="flex items-center justify-between px-1 mb-2">
          <span class="text-[13px] font-bold text-[#3a3a3c]">현재 파일</span>
          <span v-if="activeNode" class="current-file-chip">{{ activeNode.name }}</span>
        </div>
        
        <div id="file-tree">
          <!-- 검색 결과 -->
          <template v-if="searchResults !== null">
            <template v-if="searchResults.length > 0">
              <div 
                v-for="n in searchResults" 
                :key="n.id" 
                :class="['tree-item', { 'is-selected': n.id === selectedFileId }]"
                @click="n.type === 'file' ? handleSelectFile(n.id) : null"
              >
                <span 
                  class="material-symbols-outlined" 
                  :style="getNodeIconStyle(n, n.id === selectedFileId)"
                >
                  {{ getNodeIcon(n) }}
                </span>
                <span class="text-[13px] flex-1 truncate">{{ n.name }}</span>
                <span v-if="n.fileKind === 'meeting'" class="folder-side-kind-badge">회의</span>
              </div>
            </template>
            <div v-else class="text-[12px] text-[#aeaeb2] px-2 py-2">검색 결과 없음</div>
          </template>
          
          <!-- 현재 파일만 렌더링 -->
          <template v-else>
            <TreeItemComponent
              v-for="node in scopedTree"
              :key="node.id"
              :node="node"
              :depth="0"
              :activeFileId="selectedFileId"
              @selectFile="handleSelectFile"
              @toggleFolder="handleToggleFolder"
              @showContextMenu="handleShowContextMenu"
              @showMaterialMenu="handleShowMaterialMenu"
              @openMaterial="handleOpenMaterial"
              @openRecording="handleOpenRecording"
            />
          </template>
        </div>
      </section>
    </div>
  </div>

  <Teleport to="body">
    <div 
      v-if="materialMenu.visible" 
      style="position: fixed; inset: 0; z-index: 9998;" 
      @click="handleCloseMaterialMenu"
    ></div>

    <div
      v-if="materialMenu.visible"
      class="dropdown-menu"
      :style="{ display: 'block', left: materialMenu.x + 'px', top: materialMenu.y + 'px', zIndex: 9999 }"
    >
      <div class="dropdown-item" @click="handleMaterialAction('rename')">
        <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">edit</span>이름 변경
      </div>
      <div class="dropdown-divider"></div>
      <div class="dropdown-item danger" @click="handleMaterialAction('delete')">
        <span class="material-symbols-outlined text-[16px]">delete</span>삭제
      </div>
    </div>

    <div 
      v-if="ctxMenu.visible" 
      style="position: fixed; inset: 0; z-index: 9998;" 
      @click="handleCloseContextMenu"
    ></div>

    <div 
      v-if="ctxMenu.visible" 
      class="dropdown-menu" 
      :style="{ display: 'block', left: ctxMenu.x + 'px', top: ctxMenu.y + 'px', zIndex: 9999 }"
    >
      <div class="dropdown-item" @click="handleContextAction('open')">
        <span class="material-symbols-outlined text-[16px] text-[#3b82f6]">open_in_new</span>열기
      </div>
      <div class="dropdown-item" @click="handleContextAction('rename')">
        <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">edit</span>이름 변경
      </div>
      <template v-if="ctxTargetNode?.type === 'folder'">
        <div class="dropdown-item" @click="handleContextAction('new-file')">
          <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">note_add</span>새 파일
        </div>
        <div class="dropdown-item" @click="handleContextAction('new-folder')">
          <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">create_new_folder</span>새 폴더
        </div>
      </template>
      <div class="dropdown-item" @click="handleContextAction('favorite')">
        <span class="material-symbols-outlined text-[16px] text-[#ff9500]">
          {{ isCtxFavorite ? 'star_border' : 'star' }}
        </span>
        {{ isCtxFavorite ? '즐겨찾기 제거' : '즐겨찾기 추가' }}
      </div>
      <div class="dropdown-divider"></div>
      <div class="dropdown-item danger" @click="handleContextAction('delete')">
        <span class="material-symbols-outlined text-[16px]">delete</span>삭제
      </div>
    </div>
  </Teleport>
</template>

<!-- Vue 컴포넌트 내에 재귀 호출을 위한 내부 트리 아이템 정의 -->
<script>
import { computed, defineComponent, h, ref, resolveComponent } from 'vue'

const getAttachmentIcon = (name = '') => {
  return name.toLowerCase().endsWith('.pdf') ? 'picture_as_pdf' : 'slideshow'
}

const getRelatedRecordings = (recordings, materialId) => {
  return recordings.filter((recording) => Array.isArray(recording.materialIds) && recording.materialIds.includes(materialId))
}

const getWeekMaterials = (week) => Array.isArray(week?.materials) ? week.materials : []

const getWeekRecordings = (week) => Array.isArray(week?.recordings) ? week.recordings : []

const KOREAN_WEEKDAYS_SHORT = ['일', '월', '화', '수', '목', '금', '토']

const formatRecordingDateTime = (endedAt) => {
  if (!endedAt) return '저장 시간 없음'
  const date = new Date(endedAt)
  if (Number.isNaN(date.getTime())) return '저장 시간 없음'

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const weekday = KOREAN_WEEKDAYS_SHORT[date.getDay()]
  const hour = String(date.getHours()).padStart(2, '0')
  const minute = String(date.getMinutes()).padStart(2, '0')

  return `${year}.${month}.${day} · ${weekday} · ${hour}:${minute}`
}

const formatRecordingDuration = (durationText = '00:00:00') => {
  const parts = String(durationText).split(':').map((part) => Number(part))
  if (parts.some((part) => Number.isNaN(part))) return '녹음 시간 없음'

  const [hours = 0, minutes = 0, seconds = 0] = parts.length === 3
    ? parts
    : [0, parts[0] || 0, parts[1] || 0]

  const totalSeconds = (hours * 3600) + (minutes * 60) + seconds
  const nextMinutes = Math.floor(totalSeconds / 60)
  const nextSeconds = totalSeconds % 60

  if (nextMinutes <= 0) return `${nextSeconds}초`
  return `${nextMinutes}분 ${nextSeconds}초`
}

const getRecordingMeta = (recording) => {
  return `${formatRecordingDateTime(recording.endedAt)} · ${formatRecordingDuration(recording.durationText)}`
}

const buildWeekList = (node) => {
  if (Array.isArray(node.weeks) && node.weeks.length) return node.weeks

  const materials = Array.isArray(node.attachments) ? node.attachments : []
  const recordings = Array.isArray(node.recordings) ? node.recordings : []

  return [
    {
      id: `${node.id}-week-fallback`,
      label: '1주차',
      dateLabel: node.date || '',
      expanded: true,
      materialFolderExpanded: true,
      recordingFolderExpanded: true,
      materials,
      recordings
    }
  ]
}

const TreeItemComponent = defineComponent({
  name: 'TreeItemComponent',
  props: {
    node: { type: Object, required: true },
    depth: { type: Number, required: true },
    activeFileId: { type: String, default: null }
  },
  emits: ['selectFile', 'toggleFolder', 'showContextMenu', 'showMaterialMenu', 'openMaterial', 'openRecording'],
  setup(props, { emit }) {
    const isFile = computed(() => props.node.type === 'file')
    const isSelected = computed(() => props.node.id === props.activeFileId)
    const isMeetingFile = computed(() => props.node.fileKind === 'meeting')
    const resourceOpen = ref(props.node.id === props.activeFileId)
    const selectedResources = ref(new Set())
    const collapsedWeeks = ref(new Set())
    const collapsedMaterialFolders = ref(new Set())
    const collapsedRecordingFolders = ref(new Set())
    const weeks = computed(() => buildWeekList(props.node))
    const hasResources = computed(() => isFile.value && weeks.value.length > 0)
    const shouldShowResources = computed(() => hasResources.value && (resourceOpen.value || (isFile.value && props.depth === 0)))

    const handleClick = (e) => {
      e.stopPropagation()
      if (isFile.value) {
        resourceOpen.value = true
        emit('selectFile', props.node.id)
      } else {
        emit('toggleFolder', props.node.id)
      }
    }

    const handleDotsClick = (e) => {
      e.stopPropagation()
      const rect = e.currentTarget.getBoundingClientRect()
      emit('showContextMenu', props.node.id, rect.right + 4, rect.top)
    }

    const handleResourceToggle = (e) => {
      e.stopPropagation()
      if (!hasResources.value) return
      resourceOpen.value = !resourceOpen.value
    }

    const toggleSet = (targetSet, id) => {
      const nextSet = new Set(targetSet.value)
      if (nextSet.has(id)) nextSet.delete(id)
      else nextSet.add(id)
      targetSet.value = nextSet
    }

    const handleWeekToggle = (e, weekId) => {
      e.stopPropagation()
      toggleSet(collapsedWeeks, weekId)
    }

    const handleMaterialFolderToggle = (e, weekId) => {
      e.stopPropagation()
      toggleSet(collapsedMaterialFolders, weekId)
    }

    const handleRecordingFolderToggle = (e, weekId) => {
      e.stopPropagation()
      toggleSet(collapsedRecordingFolders, weekId)
    }

    const handleMaterialClick = (e, material, weekRecordings = []) => {
      e.stopPropagation()
      emit('selectFile', props.node.id)
      emit('openMaterial', {
        fileId: props.node.id,
        node: props.node,
        materialId: material.id,
        recording: getRelatedRecordings(weekRecordings, material.id)[0]
      })
    }

    const handleRecordingClick = (e, recording) => {
      e.stopPropagation()
      emit('selectFile', props.node.id)
      emit('openRecording', {
        fileId: props.node.id,
        node: props.node,
        recording
      })
    }

    const renderResourceSection = () => {
      if (!shouldShowResources.value) return null

      const allIds = []
      weeks.value.forEach(w => {
        getWeekMaterials(w).forEach(m => allIds.push(m.id))
        getWeekRecordings(w).forEach(r => allIds.push(r.id))
      })
      const allSelected = allIds.length > 0 && selectedResources.value.size === allIds.length

      const renderTreeRow = ({
        key,
        id,
        type,
        icon,
        hoverIcon,
        title,
        meta,
        count,
        depth = 0,
        isOpen = false,
        hasChevron = false,
        selectable = false,
        onClick,
        onActionClick
      }) => {
        const isSelected = id && selectedResources.value.has(id)

        return h('button', {
          key,
          type: 'button',
          class: `week-tree-row ${type} ${selectable ? 'selectable' : ''}`,
          style: { '--tree-indent': `${depth * 22}px` },
          title,
          onClick
        }, [
          h('span', {
            class: `week-tree-icon-wrapper ${type}`
          }, [
            h('span', {
              class: `material-symbols-outlined week-tree-icon default-icon ${type} ${hoverIcon ? 'has-hover' : ''}`
            }, icon),
            ...(hoverIcon ? [
              h('span', {
                class: `material-symbols-outlined week-tree-icon hover-icon ${type}`,
                onClick: (e) => { e.stopPropagation(); onActionClick?.(e); }
              }, hoverIcon)
            ] : [])
          ]),
          h('span', {
            class: 'week-tree-copy'
          }, [
            h('span', { class: 'week-tree-title-line' }, [
              h('span', { class: 'week-tree-title' }, title),
              ...(count ? [h('span', { class: 'week-tree-count' }, count)] : [])
            ]),
            ...(meta ? [h('span', { class: 'week-tree-meta' }, meta)] : [])
          ]),
          ...(selectable ? [
            h('span', {
              class: `material-symbols-outlined week-tree-checkbox ${isSelected ? 'checked' : ''}`,
              onClick: (e) => {
                e.stopPropagation()
                toggleSet(selectedResources, id)
              }
            }, isSelected ? 'check_box' : 'check_box_outline_blank')
          ] : []),
          ...(hasChevron ? [
            h('span', {
              class: `material-symbols-outlined week-tree-chevron ${isOpen ? 'open' : ''}`
            }, 'expand_more')
          ] : [])
        ])
      }

      const renderFolder = ({ week, type, title, icon, items, isOpen, onToggle, renderRows }) => {
        const countText = `${items.length}개`

        return h('div', {
          class: `week-tree-folder ${type}`
        }, [
          renderTreeRow({
            key: `${week.id}-${type}-folder`,
            type: `folder ${type}`,
            icon,
            title,
            count: countText,
            depth: 1,
            isOpen,
            hasChevron: true,
            onClick: onToggle
          }),
          isOpen
            ? h('div', { class: 'week-tree-children folder-children' }, items.length
                ? renderRows()
                : [
                    h('div', {
                      class: 'week-tree-empty',
                      style: { '--tree-indent': '44px' }
                    }, type === 'materials' ? '저장된 강의자료가 없습니다.' : '저장된 녹음본이 없습니다.')
                  ])
            : null
        ])
      }

      const headerRow = allIds.length > 0 ? h('div', {
        class: 'select-all-row'
      }, [
        h('span', { class: 'select-all-text' }, '모든 소스 선택'),
        h('span', {
          class: `material-symbols-outlined week-tree-checkbox ${allSelected ? 'checked' : ''}`,
          onClick: () => {
            if (allSelected) {
              selectedResources.value.clear()
            } else {
              allIds.forEach(id => selectedResources.value.add(id))
            }
            selectedResources.value = new Set(selectedResources.value)
          }
        }, allSelected ? 'check_box' : 'check_box_outline_blank')
      ]) : null

      return h('div', {
        class: 'week-tree-list'
      }, [
        ...(headerRow ? [headerRow] : []),
        ...weeks.value.map((week, weekIndex) => {
        const materials = getWeekMaterials(week)
        const weekRecordings = getWeekRecordings(week)
        const weekOpen = !collapsedWeeks.value.has(week.id) && week.expanded !== false
        const materialFolderOpen = !collapsedMaterialFolders.value.has(week.id) && week.materialFolderExpanded !== false
        const recordingFolderOpen = !collapsedRecordingFolders.value.has(week.id) && week.recordingFolderExpanded !== false

        return h('div', {
          key: week.id,
          class: 'week-tree-branch',
          style: { '--week-index': weekIndex }
        }, [
          renderTreeRow({
            key: `${week.id}-row`,
            type: 'week',
            icon: 'folder',
            title: week.label || `${weekIndex + 1}주차`,
            meta: week.dateLabel || '',
            count: `${materials.length + weekRecordings.length}개`,
            depth: 0,
            isOpen: weekOpen,
            hasChevron: true,
            onClick: (e) => handleWeekToggle(e, week.id)
          }),
          weekOpen
            ? h('div', { class: 'week-tree-children' }, [
                renderFolder({
                  week,
                  type: 'materials',
                  title: '강의자료',
                  icon: 'folder',
                  items: materials,
                  isOpen: materialFolderOpen,
                  onToggle: (e) => handleMaterialFolderToggle(e, week.id),
                  renderRows: () => materials.map((material) => renderTreeRow({
                    key: `material-${material.id}`,
                    id: material.id,
                    type: 'material file',
                    icon: getAttachmentIcon(material.name),
                    title: material.name,
                    meta: getRelatedRecordings(weekRecordings, material.id).length ? '연결된 녹음 있음' : '',
                    depth: 2,
                    selectable: true,
                    hoverIcon: 'more_vert',
                    onClick: (e) => handleMaterialClick(e, material, weekRecordings),
                    onActionClick: (e) => {
                      e.stopPropagation()
                      const rect = e.currentTarget.getBoundingClientRect()
                      emit('showMaterialMenu', {
                        fileId: props.node.id,
                        weekId: week.id,
                        materialId: material.id
                      }, rect.right + 4, rect.top)
                    }
                  }))
                }),
                renderFolder({
                  week,
                  type: 'recordings',
                  title: '녹음본',
                  icon: 'graphic_eq',
                  items: weekRecordings,
                  isOpen: recordingFolderOpen,
                  onToggle: (e) => handleRecordingFolderToggle(e, week.id),
                  renderRows: () => weekRecordings.map((recording, recordingIndex) => renderTreeRow({
                    key: `recording-${recording.id}`,
                    id: recording.id,
                    type: 'recording file',
                    icon: 'graphic_eq',
                    title: recording.title || `녹음본 ${recordingIndex + 1}`,
                    meta: getRecordingMeta(recording),
                    depth: 2,
                    selectable: true,
                    hoverIcon: 'more_vert',
                    onClick: (e) => handleRecordingClick(e, recording),
                    onActionClick: (e) => {
                      e.stopPropagation()
                      const rect = e.currentTarget.getBoundingClientRect()
                      emit('showMaterialMenu', {
                        fileId: props.node.id,
                        weekId: week.id,
                        recordingId: recording.id
                      }, rect.right + 4, rect.top)
                    }
                  }))
                })
              ])
            : null
        ])
      })
      ])
    }

    return () => {
      const children = []
      const resourceSection = renderResourceSection()

      if (isFile.value && props.depth === 0) {
        return h('div', { class: 'week-tree-root' }, resourceSection ? [resourceSection] : [])
      }
      
      // Node Header
      children.push(h('div', {
        class: `tree-item ${isSelected.value ? 'is-selected' : ''}`,
        style: { paddingLeft: `${8 + props.depth * 12}px` },
        onClick: handleClick
      }, [
        h('span', {
          class: 'material-symbols-outlined shrink-0',
          style: {
            fontSize: '18px',
            color: isFile.value
              ? (isMeetingFile.value ? (isSelected.value ? '#be185d' : (props.node.color || '#ec4899')) : (isSelected.value ? '#1d1d1f' : '#8e8e93'))
              : (props.node.color || '#3b82f6'),
            fontVariationSettings: isFile.value
              ? (isMeetingFile.value ? '"FILL" 1' : undefined)
              : '"FILL" 1'
          }
        }, isFile.value ? (isMeetingFile.value ? 'groups_2' : 'description') : (props.node.expanded ? 'folder_open' : 'folder')),
        h('span', {
          class: `text-[13px] flex-1 truncate ${isSelected.value ? 'font-semibold text-[#1d1d1f]' : 'font-medium text-[#3a3a3c]'}`
        }, props.node.name),
        ...(isMeetingFile.value ? [
          h('span', {
            class: 'folder-side-kind-badge'
          }, '회의')
        ] : []),
        ...(hasResources.value ? [
          h('button', {
            class: 'resource-toggle-btn',
            title: resourceOpen.value ? '파일 자료 접기' : '파일 자료 펼치기',
            onClick: handleResourceToggle
          }, [
            h('span', {
              class: `material-symbols-outlined chevron ${shouldShowResources.value ? 'open' : ''}`,
              style: { fontSize: '16px', pointerEvents: 'none' }
            }, 'chevron_right')
          ])
        ] : []),
        ...(!isFile.value ? [
          h('span', {
            class: `material-symbols-outlined chevron ${props.node.expanded ? 'open' : ''}`,
            style: { fontSize: '16px' }
          }, 'chevron_right')
        ] : []),
        h('button', {
          class: 'dots-btn',
          title: '더 보기',
          onClick: handleDotsClick
        }, [
          h('span', {
            class: 'material-symbols-outlined',
            style: { fontSize: '16px', pointerEvents: 'none' }
          }, 'more_horiz')
        ])
      ]))

      // Node Children
      if (!isFile.value && props.node.children) {
        children.push(h('div', {
          class: `tree-children ${props.node.expanded ? 'expanded' : 'collapsed'}`,
          style: { maxHeight: props.node.expanded ? '9999px' : '0' }
        }, props.node.children.map(child => h(resolveComponent('TreeItemComponent'), {
          key: child.id,
          node: child,
          depth: props.depth + 1,
          activeFileId: props.activeFileId,
          onSelectFile: (id) => emit('selectFile', id),
          onToggleFolder: (id) => emit('toggleFolder', id),
          onShowContextMenu: (id, x, y) => emit('showContextMenu', id, x, y),
          onShowMaterialMenu: (payload, x, y) => emit('showMaterialMenu', payload, x, y),
          onOpenMaterial: (payload) => emit('openMaterial', payload),
          onOpenRecording: (payload) => emit('openRecording', payload)
        }))))
      }

      if (resourceSection) {
        children.push(resourceSection)
      }

      return h('div', null, children)
    }
  }
})

export default {
  components: { TreeItemComponent }
}
</script>

<style scoped>
.folder-side-kind-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(236, 72, 153, 0.12);
  color: #db2777;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: -0.01em;
  white-space: nowrap;
}

.current-file-chip {
  max-width: 150px;
  overflow: hidden;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(244, 237, 228, 0.78);
  color: #6b5b45;
  font-size: 11px;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.resource-toggle-btn {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 7px;
  color: #8e8e93;
}
</style>

<style>
@keyframes weekBodyDrop {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.week-tree-root {
  width: 100%;
}

.week-tree-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0 0 16px;
  padding: 0;
}

.week-tree-branch {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 4px;
  animation: weekBodyDrop 0.22s ease both;
  animation-delay: calc(var(--week-index, 0) * 0.035s);
}

.week-tree-children {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding-left: 14px;
  animation: weekBodyDrop 0.22s ease both;
}

.week-tree-children::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 6px;
  left: 13px;
  width: 2px;
  border-radius: 999px;
  background: rgba(222, 205, 182, 0.58);
}

.week-tree-children.folder-children {
  gap: 8px;
  padding-top: 8px;
  padding-left: 0;
  padding-bottom: 4px;
}

.week-tree-children.folder-children::before {
  display: none;
}

.week-tree-row {
  --tree-indent: 0px;
  width: calc(100% - var(--tree-indent));
  min-height: 40px;
  display: flex;
  align-items: center;
  gap: 9px;
  margin-left: var(--tree-indent);
  padding: 8px 12px;
  border-radius: 16px;
  text-align: left;
  color: #1f2937;
  background: rgba(244, 237, 228, 0.76);
  transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
}

.week-tree-row:hover {
  background: rgba(244, 237, 228, 0.96);
}

.week-tree-row:active {
  transform: scale(0.995);
}

.week-tree-row.week {
  min-height: 46px;
  border: 1px solid rgba(222, 205, 182, 0.78);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.74);
}

.week-tree-row.folder {
  min-height: 42px;
  border: 1px solid rgba(222, 205, 182, 0.78);
  border-radius: 16px;
  background: rgba(244, 237, 228, 0.76);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.74);
}

.week-tree-row.folder:hover {
  background: rgba(244, 237, 228, 0.96);
}

.week-tree-row.file {
  min-height: 40px;
  padding: 7px 10px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(229, 229, 234, 0.82);
  border-radius: 14px;
}

.week-tree-row.file:hover {
  border-color: rgba(191, 165, 128, 0.48);
  background: #ffffff;
}

.week-tree-icon {
  flex: 0 0 auto;
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #8e8e93;
  font-size: 17px;
}

.week-tree-icon.week {
  color: #8a6f4b;
  font-variation-settings: 'FILL' 1;
}

.week-tree-icon.folder {
  color: #2563eb;
  font-variation-settings: 'FILL' 1;
}

.week-tree-icon.folder.recordings {
  color: #b7791f;
  font-variation-settings: 'FILL' 0;
}

.week-tree-icon.file,
.week-tree-icon.material {
  background: rgba(239, 246, 255, 0.92);
  color: #2563eb;
  font-size: 16px;
}

.week-tree-icon.recording {
  background: rgba(255, 247, 237, 0.92);
  color: #b7791f;
  font-size: 16px;
}

.week-tree-copy {
  min-width: 0;
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 2px;
}

.week-tree-title-line {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
}

.week-tree-title {
  overflow: hidden;
  color: #1f2937;
  font-size: 12px;
  font-weight: 900;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.week-tree-row.week .week-tree-title {
  font-size: 13px;
  font-weight: 950;
}

.week-tree-row.folder .week-tree-title {
  font-size: 13px;
  font-weight: 950;
}

.week-tree-meta {
  overflow: hidden;
  color: #8e8e93;
  font-size: 10px;
  font-weight: 800;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.week-tree-count {
  flex: 0 0 auto;
  padding: 3px 7px;
  border-radius: 999px;
  background: rgba(29, 29, 31, 0.06);
  color: #6b7280;
  font-size: 10px;
  font-weight: 950;
  line-height: 1;
}

.week-tree-chevron,
.week-tree-open,
.week-tree-action {
  flex: 0 0 auto;
  color: #9ca3af;
  font-size: 17px;
  transition: transform 0.18s ease, color 0.18s ease;
}

.week-tree-chevron.open {
  color: #6b5b45;
  transform: rotate(180deg);
}

.week-tree-open {
  font-size: 16px;
}

.week-tree-action {
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 7px;
  font-size: 18px;
}

.week-tree-action:hover {
  background: rgba(148, 163, 184, 0.12);
  color: #4b5563;
}

.week-tree-empty {
  --tree-indent: 0px;
  min-height: 38px;
  display: flex;
  align-items: center;
  margin-left: var(--tree-indent);
  padding: 9px 12px;
  border: 1px dashed rgba(191, 165, 128, 0.52);
  border-radius: 13px;
  color: #9ca3af;
  font-size: 11px;
  font-weight: 850;
  background: rgba(250, 247, 242, 0.58);
}

.week-tree-icon-wrapper {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
}
.week-tree-icon-wrapper .hover-icon {
  position: absolute;
  inset: 0;
  display: none;
  background: rgba(239, 246, 255, 0.92);
  border-radius: 8px;
  cursor: pointer;
  z-index: 10;
  align-items: center;
  justify-content: center;
}
.week-tree-row:hover .hover-icon {
  display: inline-flex;
}
.week-tree-row:hover .default-icon.has-hover {
  opacity: 0;
}
.week-tree-checkbox {
  flex: 0 0 auto;
  color: #d1d1d6;
  font-size: 20px;
  cursor: pointer;
  transition: color 0.18s ease;
}
.week-tree-checkbox:hover {
  color: #8e8e93;
}
.week-tree-checkbox.checked {
  color: #1f2937;
  font-variation-settings: 'FILL' 1;
}
.select-all-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  margin-bottom: 4px;
}
.select-all-text {
  color: #6b7280;
  font-size: 13px;
  font-weight: 800;
}
</style>
