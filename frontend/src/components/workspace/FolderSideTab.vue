<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits([
  'update:fileTree',
  'update:favorites',
  'fileSelect',
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
const activeFileId = ref(null)
const searchQuery = ref('')
const ctxMenu = ref({ visible: false, x: 0, y: 0, targetId: null })

// --- Computed ---
const allFlat = computed(() => flattenAll(props.fileTree))
const searchResults = computed(() => {
  if (!searchQuery.value.trim()) return null
  return allFlat.value.filter(n => n.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
})
const favoriteNodes = computed(() => allFlat.value.filter(n => props.favorites.has(n.id)))
const activeNode = computed(() => findNode(activeFileId.value, props.fileTree))
const ctxTargetNode = computed(() => ctxMenu.value.targetId ? findNode(ctxMenu.value.targetId, props.fileTree) : null)
const isCtxFavorite = computed(() => ctxMenu.value.targetId ? props.favorites.has(ctxMenu.value.targetId) : false)

// --- 핸들러 ---
const handleSelectFile = (id) => {
  activeFileId.value = id
  emit('fileSelect', id, findNode(id, props.fileTree))
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

const handleContextAction = (action) => {
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
      const newFile = { id: genId(), type: 'file', name: '새 파일', content: '' }
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
      deleteNode(targetId, copy)
      emit('showToast', `"${node.name}" 삭제됨`)
      break
  }
  emit('update:fileTree', copy)
}

const handleNewFile = () => {
  const newFile = { id: genId(), type: 'file', name: '새 파일', content: '' }
  emit('update:fileTree', [...props.fileTree, newFile])
  emit('showToast', '새 파일이 추가되었습니다')
}

const handleNewFolder = () => {
  const color = FOLDER_COLORS[colorIdx++ % FOLDER_COLORS.length]
  const newFolder = { id: genId(), type: 'folder', name: '새 폴더', color, expanded: false, children: [] }
  emit('update:fileTree', [...props.fileTree, newFolder])
  emit('showToast', '새 폴더가 추가되었습니다')
}
</script>

<template>
  <div class="flex flex-col flex-1 overflow-hidden">
    <!-- 검색 창 -->
    <div class="sidebar-search-bg workspace-inset-shell rounded-[24px] px-3 py-3 flex items-center gap-3 mb-6">
      <span class="material-symbols-outlined text-[#8e8e93] text-[20px]">search</span>
      <input
        class="bg-transparent border-none focus:ring-0 p-0 text-[14px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
        placeholder="제목으로 검색"
        type="text"
        v-model="searchQuery"
      />
    </div>

    <div class="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
      <!-- 활성화한 파일 -->
      <section>
        <div class="flex items-center justify-between px-1 mb-2">
          <span class="text-[13px] font-bold text-[#3a3a3c]">활성화한 파일</span>
        </div>
        <div v-if="activeNode" class="tree-item active-file-row">
          <span class="material-symbols-outlined text-[#1d1d1f]" style="font-size: 18px;">description</span>
          <span class="text-[13px] font-semibold text-[#1d1d1f] flex-1 truncate">{{ activeNode.name }}</span>
        </div>
      </section>

      <!-- 구조 (Tree) -->
      <section>
        <div class="flex items-center justify-between px-1 mb-2">
          <span class="text-[13px] font-bold text-[#3a3a3c]">구조</span>
          <div class="flex gap-1">
            <button class="btn-ghost-icon p-0.5 rounded-lg" title="새 폴더" @click="handleNewFolder">
              <span class="material-symbols-outlined text-[18px]" style="color: #8e8e93;">create_new_folder</span>
            </button>
            <button class="btn-ghost-icon p-0.5 rounded-lg" title="새 파일" @click="handleNewFile">
              <span class="material-symbols-outlined text-[18px]" style="color: #8e8e93;">note_add</span>
            </button>
          </div>
        </div>
        
        <div id="file-tree">
          <!-- 검색 결과 -->
          <template v-if="searchResults !== null">
            <template v-if="searchResults.length > 0">
              <div 
                v-for="n in searchResults" 
                :key="n.id" 
                :class="['tree-item', { 'is-selected': n.id === activeFileId }]"
                @click="n.type === 'file' ? handleSelectFile(n.id) : null"
              >
                <span 
                  class="material-symbols-outlined" 
                  :style="{ color: n.color || '#8e8e93', fontSize: '18px', fontVariationSettings: n.type === 'folder' ? `'FILL' 1` : `'FILL' 0` }"
                >
                  {{ n.type === 'folder' ? 'folder' : 'description' }}
                </span>
                <span class="text-[13px] flex-1 truncate">{{ n.name }}</span>
              </div>
            </template>
            <div v-else class="text-[12px] text-[#aeaeb2] px-2 py-2">검색 결과 없음</div>
          </template>
          
          <!-- 전체 트리 렌더링 (재귀 컴포넌트 생략 및 평탄화 구조 기반 렌더링 예시) -->
          <template v-else>
            <TreeItemComponent 
              v-for="node in fileTree" 
              :key="node.id" 
              :node="node" 
              :depth="0" 
              :activeFileId="activeFileId"
              @selectFile="handleSelectFile"
              @toggleFolder="handleToggleFolder"
              @showContextMenu="handleShowContextMenu"
            />
          </template>
        </div>
      </section>

      <!-- 즐겨찾기 -->
      <section>
        <div class="flex items-center justify-between px-1 mb-2">
          <span class="text-[13px] font-bold text-[#3a3a3c]">즐겨찾기</span>
        </div>
        <div class="flex flex-col gap-0.5">
          <div 
            v-for="n in favoriteNodes" 
            :key="n.id" 
            class="tree-item"
            @click="n.type === 'file' ? handleSelectFile(n.id) : handleToggleFolder(n.id)"
          >
            <span 
              class="material-symbols-outlined" 
              :style="{ color: n.color || '#8e8e93', fontSize: '18px', fontVariationSettings: n.type === 'folder' ? `'FILL' 1` : `'FILL' 0` }"
            >
              {{ n.type === 'folder' ? 'folder' : 'description' }}
            </span>
            <span class="text-[13px] font-medium text-[#3a3a3c] flex-1 truncate">{{ n.name }}</span>
          </div>
        </div>
      </section>

      <!-- 리스트 (Flatten All) -->
      <section>
        <div class="flex items-center justify-between px-1 mb-2">
          <span class="text-[13px] font-bold text-[#3a3a3c]">리스트</span>
        </div>
        <div class="flex flex-col gap-0.5 pb-4">
          <div 
            v-for="n in allFlat" 
            :key="n.id" 
            class="tree-item"
            @click="n.type === 'file' ? handleSelectFile(n.id) : null"
          >
            <span 
              class="material-symbols-outlined" 
              :style="{ color: n.color || '#8e8e93', fontSize: '18px', fontVariationSettings: n.type === 'folder' ? `'FILL' 1` : `'FILL' 0` }"
            >
              {{ n.type === 'folder' ? 'folder' : 'description' }}
            </span>
            <span class="text-[13px] text-[#3a3a3c] flex-1 truncate">{{ n.name }}</span>
          </div>
        </div>
      </section>
    </div>
  </div>

  <Teleport to="body">
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
import { defineComponent, h, resolveComponent } from 'vue'

const TreeItemComponent = defineComponent({
  name: 'TreeItemComponent',
  props: {
    node: { type: Object, required: true },
    depth: { type: Number, required: true },
    activeFileId: { type: String, default: null }
  },
  emits: ['selectFile', 'toggleFolder', 'showContextMenu'],
  setup(props, { emit }) {
    const isFile = computed(() => props.node.type === 'file')
    const isSelected = computed(() => props.node.id === props.activeFileId)

    const handleClick = (e) => {
      e.stopPropagation()
      if (isFile.value) emit('selectFile', props.node.id)
      else emit('toggleFolder', props.node.id)
    }

    const handleDotsClick = (e) => {
      e.stopPropagation()
      const rect = e.currentTarget.getBoundingClientRect()
      emit('showContextMenu', props.node.id, rect.right + 4, rect.top)
    }

    return () => {
      const children = []
      
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
            color: isFile.value ? (isSelected.value ? '#1d1d1f' : '#8e8e93') : (props.node.color || '#3b82f6'),
            fontVariationSettings: isFile.value ? undefined : '"FILL" 1'
          }
        }, isFile.value ? 'description' : (props.node.expanded ? 'folder_open' : 'folder')),
        h('span', {
          class: `text-[13px] flex-1 truncate ${isSelected.value ? 'font-semibold text-[#1d1d1f]' : 'font-medium text-[#3a3a3c]'}`
        }, props.node.name),
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
          onShowContextMenu: (id, x, y) => emit('showContextMenu', id, x, y)
        }))))
      }

      return h('div', null, children)
    }
  }
})

export default {
  components: { TreeItemComponent }
}
</script>
