import { ref, onMounted, onUnmounted } from 'vue'

const FOLDER_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
const LECTURE_FILE_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
const MEETING_FILE_COLORS = ['#ec4899', '#f97316', '#14b8a6', '#6366f1', '#0ea5e9']

// 홈/작업 폴더 화면의 모달, 폴더 이동, 파일/폴더 생성 액션을 관리합니다.
export function useHome(props, emit) {
  const isSidebarCollapsed = ref(false)
  const isFolderModalOpen = ref(false)
  const isFileModalOpen = ref(false)
  const isMeetingFileModalOpen = ref(false)
  const isEditItemModalOpen = ref(false)
  const selectedColor = ref('#3b82f6')
  const navigationStack = ref([]) // [{id, name}]
  const editingItemId = ref(null)
  const editingItemType = ref('file')
  const editingFileKind = ref('lecture')

  const newFolderName = ref('')
  const newFileName = ref('')

  const closeAllModals = () => {
    isFolderModalOpen.value = false
    isFileModalOpen.value = false
    isMeetingFileModalOpen.value = false
    isEditItemModalOpen.value = false
  }

  const resetEditingState = () => {
    editingItemId.value = null
    editingItemType.value = 'file'
    editingFileKind.value = 'lecture'
    newFileName.value = ''
  }

  // 모달 바깥 배경을 클릭하면 생성/수정 모달을 닫습니다.
  const handleOutsideClick = (e) => {
    if (e.target.classList.contains('modal-overlay')) {
      closeAllModals()
    }
  }

  onMounted(() => {
    document.addEventListener('mousedown', handleOutsideClick)
  })

  onUnmounted(() => {
    document.removeEventListener('mousedown', handleOutsideClick)
  })

  // 파일/폴더 카드의 즐겨찾기 상태를 토글합니다.
  const toggleStar = (e, targetId) => {
    e.stopPropagation()
    const next = new Set(props.favorites)
    if (next.has(targetId)) next.delete(targetId)
    else next.add(targetId)
    emit('update:favorites', next)
  }

  // 현재 위치 또는 특정 폴더 안에 새 파일/폴더 노드를 추가합니다.
  const addItemToTree = (nodes, parentId, newItem) => {
    if (!parentId) return [newItem, ...nodes]
    return nodes.map((node) => {
      if (node.id === parentId) {
        return { ...node, children: [newItem, ...(node.children || [])] }
      }
      if (node.children) {
        return { ...node, children: addItemToTree(node.children, parentId, newItem) }
      }
      return node
    })
  }

  const findItemInTree = (nodes, targetId) => {
    for (const node of nodes) {
      if (node.id === targetId) return node
      if (node.children) {
        const found = findItemInTree(node.children, targetId)
        if (found) return found
      }
    }
    return null
  }

  const updateItemInTree = (nodes, targetId, updater) => {
    return nodes.map((node) => {
      if (node.id === targetId) {
        return updater(node)
      }
      if (node.children) {
        return { ...node, children: updateItemInTree(node.children, targetId, updater) }
      }
      return node
    })
  }

  const removeItemFromTree = (nodes, targetId) => {
    return nodes
      .filter((node) => node.id !== targetId)
      .map((node) => {
        if (node.children) {
          return { ...node, children: removeItemFromTree(node.children, targetId) }
        }
        return node
      })
  }

  const formatNow = () => {
    const now = new Date()
    return `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`
  }

  // 입력한 이름과 색상으로 새 폴더를 생성합니다.
  const handleCreateFolder = () => {
    if (!newFolderName.value.trim()) return

    const newFolder = {
      id: 'f' + Date.now(),
      type: 'folder',
      name: newFolderName.value,
      date: formatNow(),
      color: selectedColor.value,
      expanded: false,
      children: []
    }

    const currentFolderId = navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null

    emit('update:fileTree', addItemToTree(props.fileTree, currentFolderId, newFolder))
    isFolderModalOpen.value = false
    newFolderName.value = ''
  }

  // 입력한 이름과 색상으로 새 파일을 생성합니다.
  const createFileNode = (fileKind) => ({
    id: 'file-' + Date.now(),
    type: 'file',
    fileKind,
    name: newFileName.value,
    date: formatNow(),
    color: selectedColor.value,
    content: '',
    attachments: []
  })

  const handleCreateFile = (fileKind = 'lecture') => {
    if (!newFileName.value.trim()) return
    const newFile = createFileNode(fileKind)

    const currentFolderId = navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null
    emit('update:fileTree', addItemToTree(props.fileTree, currentFolderId, newFile))
    isFileModalOpen.value = false
    isMeetingFileModalOpen.value = false
    newFileName.value = ''
  }

  const openItemEditModal = (targetId) => {
    const targetNode = findItemInTree(props.fileTree, targetId)
    if (!targetNode) return

    editingItemId.value = targetId
    editingItemType.value = targetNode.type || 'file'
    editingFileKind.value = targetNode.fileKind || 'lecture'
    newFileName.value = targetNode.name || ''

    if (targetNode.type === 'folder') {
      selectedColor.value = targetNode.color || FOLDER_COLORS[0]
    } else {
      selectedColor.value = targetNode.color || (editingFileKind.value === 'meeting' ? MEETING_FILE_COLORS[0] : LECTURE_FILE_COLORS[0])
    }

    isEditItemModalOpen.value = true
  }

  const closeEditItemModal = () => {
    isEditItemModalOpen.value = false
    resetEditingState()
  }

  const handleUpdateItem = () => {
    if (!editingItemId.value || !newFileName.value.trim()) return

    emit('update:fileTree', updateItemInTree(props.fileTree, editingItemId.value, (node) => ({
      ...node,
      name: newFileName.value.trim(),
      color: selectedColor.value
    })))

    closeEditItemModal()
  }

  const handleDeleteEditingItem = () => {
    if (!editingItemId.value) return

    emit('update:fileTree', removeItemFromTree(props.fileTree, editingItemId.value))

    if (props.favorites.has(editingItemId.value)) {
      const nextFavorites = new Set(props.favorites)
      nextFavorites.delete(editingItemId.value)
      emit('update:favorites', nextFavorites)
    }

    closeEditItemModal()
  }

  // 폴더 카드 안으로 들어가도록 navigationStack에 현재 폴더를 쌓습니다.
  const handleEnterFolder = (e, item) => {
    e.stopPropagation()
    navigationStack.value.push({ id: item.id, name: item.name })
  }

  // 상위 폴더로 돌아갑니다.
  const handleGoBack = () => {
    navigationStack.value.pop()
  }

  return {
    isSidebarCollapsed,
    isFolderModalOpen,
    isFileModalOpen,
    isMeetingFileModalOpen,
    isEditItemModalOpen,
    selectedColor,
    navigationStack,
    editingItemType,
    editingFileKind,
    newFolderName,
    newFileName,
    FOLDER_COLORS,
    LECTURE_FILE_COLORS,
    MEETING_FILE_COLORS,
    toggleStar,
    handleCreateFolder,
    handleCreateFile,
    openItemEditModal,
    closeEditItemModal,
    handleUpdateItem,
    handleDeleteEditingItem,
    handleEnterFolder,
    handleGoBack
  }
}
