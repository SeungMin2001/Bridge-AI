import { ref, onMounted, onUnmounted } from 'vue'

// 홈/작업 폴더 화면의 모달, 폴더 이동, 파일/폴더 생성 액션을 관리합니다.
export function useHome({ fileTree, favorites }, emit) {
  const isSidebarCollapsed = ref(false)
  const isFolderModalOpen = ref(false)
  const isFileModalOpen = ref(false)
  const selectedColor = ref('#3b82f6')
  const navigationStack = ref([]) // [{id, name}]

  const newFolderName = ref('')
  const newFileName = ref('')

  // 모달 바깥 배경을 클릭하면 생성 모달을 닫습니다.
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

  // 파일/폴더 카드의 즐겨찾기 상태를 토글합니다.
  const toggleStar = (e, targetId) => {
    e.stopPropagation()
    const next = new Set(favorites)
    if (next.has(targetId)) next.delete(targetId)
    else next.add(targetId)
    emit('update:favorites', next)
  }

  // 현재 위치 또는 특정 폴더 안에 새 파일/폴더 노드를 추가합니다.
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

  // 입력한 이름과 색상으로 새 폴더를 생성합니다.
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

  // 입력한 이름과 색상으로 새 파일을 생성합니다.
  const handleCreateFile = () => {
    if (!newFileName.value.trim()) return
    const now = new Date()
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`

    const newFile = {
      id: 'file-' + Date.now(),
      type: 'file',
      name: newFileName.value,
      date: timeStr,
      color: selectedColor.value
    }

    const currentFolderId = navigationStack.value.length > 0 ? navigationStack.value[navigationStack.value.length - 1].id : null
    emit('update:fileTree', addItemToTree(fileTree, currentFolderId, newFile))
    isFileModalOpen.value = false
    newFileName.value = ''
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
