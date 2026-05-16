import { ref, onMounted, onUnmounted } from 'vue'
import {
  createWorkspaceFile,
  createWorkspaceFolder,
  deleteWorkspaceFile,
  deleteWorkspaceFolder,
  isWorkspaceUuid,
  updateWorkspaceFolder
} from '../api/workspaceApi.js'

const FOLDER_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
const LECTURE_FILE_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
const MEETING_FILE_COLORS = ['#ec4899', '#f97316', '#14b8a6', '#6366f1', '#0ea5e9']
const FILE_TAGS = ['수업', '회의', '프로젝트', '개인', '중요']
const FILE_ICONS = ['article', 'groups_2', 'workspaces', 'person', 'priority_high', 'star', 'task_alt', 'lightbulb', 'bookmark', 'school']

const getDefaultIconForTag = (tag = '') => {
  if (tag === '회의') return 'groups_2'
  if (tag === '프로젝트') return 'workspaces'
  if (tag === '개인') return 'person'
  if (tag === '중요') return 'priority_high'
  return 'article'
}

// 홈/작업 폴더 화면의 모달, 폴더 이동, 파일/폴더 생성 액션을 관리합니다.
export function useHome(props, emit) {
  const isSidebarCollapsed = ref(false)
  const isFolderModalOpen = ref(false)
  const isFileModalOpen = ref(false)
  const isEditItemModalOpen = ref(false)
  const selectedColor = ref('#3b82f6')
  const selectedTag = ref('수업')
  const selectedFileIcon = ref(getDefaultIconForTag('수업'))
  const navigationStack = ref([]) // [{id, name}]
  const editingItemId = ref(null)
  const editingItemType = ref('file')
  const editingFileKind = ref('lecture')

  const newFolderName = ref('')
  const newFileName = ref('')

  const closeAllModals = () => {
    isFolderModalOpen.value = false
    isFileModalOpen.value = false
    isEditItemModalOpen.value = false
  }

  const resetEditingState = () => {
    editingItemId.value = null
    editingItemType.value = 'file'
    editingFileKind.value = 'lecture'
    newFileName.value = ''
  }

  const handleFileTagChange = (tag = '수업') => {
    selectedTag.value = tag
    selectedFileIcon.value = getDefaultIconForTag(tag)
  }

  const openFileCreateModal = () => {
    newFileName.value = ''
    selectedColor.value = LECTURE_FILE_COLORS[0]
    handleFileTagChange('수업')
    isFileModalOpen.value = true
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

  const getCurrentFolderId = () => {
    return navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null
  }

  // 입력한 이름과 색상으로 새 폴더 생성
  const handleCreateFolder = async () => {
    if (!newFolderName.value.trim()) return

    const currentFolderId = getCurrentFolderId()
    const newFolder = await createWorkspaceFolder({
      title: newFolderName.value.trim(),
      parent_course_id: isWorkspaceUuid(currentFolderId) ? currentFolderId : null,
      color: selectedColor.value,
      icon: 'folder'
    })

    emit('update:fileTree', addItemToTree(props.fileTree, currentFolderId, newFolder))
    isFolderModalOpen.value = false
    newFolderName.value = ''
  }

  // 입력한 이름과 색상으로 새 파일 생성
  const handleCreateFile = async (fileKind = selectedTag.value === '회의' ? 'meeting' : 'lecture') => {
    if (!newFileName.value.trim()) return

    const currentFolderId = getCurrentFolderId()
    const newFile = await createWorkspaceFile({
      course_id: isWorkspaceUuid(currentFolderId) ? currentFolderId : null,
      title: newFileName.value.trim(),
      file_kind: fileKind,
      tag: selectedTag.value,
      icon: selectedFileIcon.value || getDefaultIconForTag(selectedTag.value),
      color: selectedColor.value
    })

    emit('update:fileTree', addItemToTree(props.fileTree, currentFolderId, newFile))
    isFileModalOpen.value = false
    newFileName.value = ''
  }

  const openItemEditModal = (targetId) => {
    const targetNode = findItemInTree(props.fileTree, targetId)
    if (!targetNode) return

    editingItemId.value = targetId
    editingItemType.value = targetNode.type || 'file'
    editingFileKind.value = targetNode.fileKind || 'lecture'
    newFileName.value = targetNode.name || ''
    selectedTag.value = targetNode.tag || (editingFileKind.value === 'meeting' ? '회의' : '수업')
    selectedFileIcon.value = targetNode.fileIcon || getDefaultIconForTag(selectedTag.value)

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

  const handleUpdateItem = async () => {
    if (!editingItemId.value || !newFileName.value.trim()) return

    const targetNode = findItemInTree(props.fileTree, editingItemId.value)
    if (targetNode?.type === 'folder' && isWorkspaceUuid(targetNode.id)) {
      await updateWorkspaceFolder(targetNode.id, {
        title: newFileName.value.trim(),
        color: selectedColor.value,
        icon: targetNode.icon || 'folder'
      })
    }

    emit('update:fileTree', updateItemInTree(props.fileTree, editingItemId.value, (node) => ({
      ...node,
      name: newFileName.value.trim(),
      color: selectedColor.value,
      ...(node.type === 'file'
        ? {
            tag: selectedTag.value,
            fileIcon: selectedFileIcon.value || getDefaultIconForTag(selectedTag.value)
          }
        : {})
    })))

    closeEditItemModal()
  }

  const deleteItemById = async (targetId) => {
    if (!targetId) return

    const targetNode = findItemInTree(props.fileTree, targetId)
    if (targetNode?.type === 'file' && isWorkspaceUuid(targetNode.id)) {
      await deleteWorkspaceFile(targetNode.id)
    } else if (targetNode?.type === 'folder' && isWorkspaceUuid(targetNode.id)) {
      await deleteWorkspaceFolder(targetNode.id)
    }

    emit('update:fileTree', removeItemFromTree(props.fileTree, targetId))

    if (props.favorites.has(targetId)) {
      const nextFavorites = new Set(props.favorites)
      nextFavorites.delete(targetId)
      emit('update:favorites', nextFavorites)
    }
  }

  const handleDeleteEditingItem = async () => {
    if (!editingItemId.value) return

    await deleteItemById(editingItemId.value)

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
    handleEnterFolder,
    handleGoBack
  }
}
