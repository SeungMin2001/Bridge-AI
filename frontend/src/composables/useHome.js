import { ref, onMounted, onUnmounted } from 'vue'

export function useHome({ fileTree, favorites }, emit) {
  const isSidebarCollapsed = ref(false)
  const isFolderModalOpen = ref(false)
  const isFileModalOpen = ref(false)
  const selectedColor = ref('#3b82f6')
  const navigationStack = ref([]) // [{id, name}]

  const newFolderName = ref('')
  const newFileName = ref('')

  const handleOutsideClick = (e) => {
    // Modals
    if (e.target.classList.contains('modal-overlay')) {
      isFolderModalOpen.value = false
      isFileModalOpen.value = false
    }
  }

  onMounted(() => {
    document.addEventListener('mousedown', handleOutsideClick)
  })

  onUnmounted(() => {
    document.removeEventListener('mousedown', handleOutsideClick)
  })

  const toggleStar = (e, targetId) => {
    e.stopPropagation()
    const next = new Set(favorites)
    if (next.has(targetId)) next.delete(targetId)
    else next.add(targetId)
    emit('update:favorites', next)
  }

  const addItemToTree = (nodes, parentId, newItem) => {
    if (!parentId) return [newItem, ...nodes]
    return nodes.map(node => {
      if (node.id === parentId) {
        return { ...node, children: [newItem, ...(node.children || [])] }
      }
      if (node.children) {
        return { ...node, children: addItemToTree(node.children, parentId, newItem) }
      }
      return node
    })
  }

  const handleCreateFolder = () => {
    if (!newFolderName.value.trim()) return
    const now = new Date()
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`

    const newFolder = {
      id: 'f' + Date.now(),
      type: 'folder',
      name: newFolderName.value,
      date: timeStr,
      color: selectedColor.value,
      expanded: false,
      children: []
    }

    const currentFolderId = navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null
    
    emit('update:fileTree', addItemToTree(fileTree, currentFolderId, newFolder))
    isFolderModalOpen.value = false
    newFolderName.value = ''
  }

  const handleCreateFile = () => {
    if (!newFileName.value.trim()) return
    const now = new Date()
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`

    const newFile = {
      id: 'file-' + Date.now(),
      type: 'file',
      name: newFileName.value,
      date: timeStr,
      color: '#1d1d1f'
    }

    const currentFolderId = navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null
    emit('update:fileTree', addItemToTree(fileTree, currentFolderId, newFile))
    isFileModalOpen.value = false
    newFileName.value = ''
  }

  const handleEnterFolder = (e, item) => {
    e.stopPropagation()
    navigationStack.value.push({ id: item.id, name: item.name })
  }

  const handleGoBack = () => {
    navigationStack.value.pop()
  }

  return {
    isSidebarCollapsed,
    isFolderModalOpen,
    isFileModalOpen,
    selectedColor,
    navigationStack,
    newFolderName,
    newFileName,
    toggleStar,
    handleCreateFolder,
    handleCreateFile,
    handleEnterFolder,
    handleGoBack
  }
}
