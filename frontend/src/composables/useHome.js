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
const DEFAULT_FILE_ICON = 'article'
const DEFAULT_FOLDER_NAME = '기본폴더'
const DEFAULT_FOLDER_LEGACY_NAME = '기본파일'

const getDefaultIconForTag = () => DEFAULT_FILE_ICON

const isDefaultFolderNode = (node) => (
  node?.type === 'folder'
  && (node.isDefaultFolder || node.name === DEFAULT_FOLDER_NAME || node.name === DEFAULT_FOLDER_LEGACY_NAME)
)

// 홈/작업 폴더 화면의 모달, 폴더 이동, 파일/폴더 생성 액션을 관리합니다.
export function useHome(props, emit) {
  const isSidebarCollapsed = ref(false)
  const isFolderModalOpen = ref(false)
  const isFileModalOpen = ref(false)
  const isEditItemModalOpen = ref(false)
  const selectedColor = ref('#3b82f6')
  const selectedTag = ref('수업')
  const selectedFileIcon = ref(DEFAULT_FILE_ICON)
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
    selectedFileIcon.value = DEFAULT_FILE_ICON
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

  const removeItemsFromTree = (nodes, targetIds) => {
    return nodes
      .filter((node) => !targetIds.has(node.id))
      .map((node) => {
        if (node.children) {
          return { ...node, children: removeItemsFromTree(node.children, targetIds) }
        }
        return node
      })
  }

  const getCurrentFolderId = () => {
    return navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null
  }

  const getDefaultFolderId = () => {
    const defaultFolder = props.fileTree.find(isDefaultFolderNode)
    return defaultFolder?.id || null
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
    if (!newFileName.value.trim()) return null

    const currentFolderId = getCurrentFolderId()
    const targetFolderId = isWorkspaceUuid(currentFolderId) ? currentFolderId : getDefaultFolderId()
    const newFile = await createWorkspaceFile({
      course_id: isWorkspaceUuid(targetFolderId) ? targetFolderId : null,
      title: newFileName.value.trim(),
      file_kind: fileKind,
      tag: selectedTag.value,
      icon: DEFAULT_FILE_ICON,
      color: selectedColor.value
    })

    emit('update:fileTree', addItemToTree(props.fileTree, targetFolderId, newFile))
    isFileModalOpen.value = false
    newFileName.value = ''
    return newFile
  }

  const openItemEditModal = (targetId) => {
    const targetNode = findItemInTree(props.fileTree, targetId)
    if (!targetNode) return

    editingItemId.value = targetId
    editingItemType.value = targetNode.type || 'file'
    editingFileKind.value = targetNode.fileKind || 'lecture'
    newFileName.value = targetNode.name || ''
    selectedTag.value = targetNode.tag || (editingFileKind.value === 'meeting' ? '회의' : '수업')
    selectedFileIcon.value = DEFAULT_FILE_ICON

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
            fileIcon: DEFAULT_FILE_ICON
          }
        : {})
    })))

    closeEditItemModal()
  }

  const deleteItemById = async (targetId) => {
    if (!targetId) return

    const targetNode = findItemInTree(props.fileTree, targetId)
    if (isDefaultFolderNode(targetNode)) return

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

  const deleteItemsByIds = async (targetIds = []) => {
    const ids = Array.from(new Set(targetIds)).filter(Boolean)
    if (!ids.length) return

    const targets = ids
      .map((id) => findItemInTree(props.fileTree, id))
      .filter((targetNode) => !isDefaultFolderNode(targetNode))
      .filter(Boolean)
    const deletableIds = new Set(targets.map((targetNode) => targetNode.id))
    if (!deletableIds.size) return

    await Promise.all(targets.map(async (targetNode) => {
      if (targetNode.type === 'file' && isWorkspaceUuid(targetNode.id)) {
        await deleteWorkspaceFile(targetNode.id)
      } else if (targetNode.type === 'folder' && isWorkspaceUuid(targetNode.id)) {
        await deleteWorkspaceFolder(targetNode.id)
      }
    }))

    emit('update:fileTree', removeItemsFromTree(props.fileTree, deletableIds))

    const nextFavorites = new Set(props.favorites)
    deletableIds.forEach((id) => nextFavorites.delete(id))
    emit('update:favorites', nextFavorites)

    navigationStack.value = navigationStack.value.filter((item) => !deletableIds.has(item.id))
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
    deleteItemsByIds,
    handleEnterFolder,
    handleGoBack
  }
}
