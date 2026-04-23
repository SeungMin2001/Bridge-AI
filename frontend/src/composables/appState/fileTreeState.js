import { computed, onMounted, watch, ref } from 'vue'

// 홈/워크스페이스에서 사용하는 파일 트리, 즐겨찾기, 현재 선택 파일을 관리합니다.
const STORAGE_KEYS = {
  fileTree: 'lecto_file_tree',
  favorites: 'lecto_favorites'
}

// 강의자료 첨부가 붙는 기본 파일 노드입니다.
const createLectureOneNode = () => ({
  id: 'lecture-1',
  type: 'file',
  name: '강의1',
  content: '',
  attachments: []
})

// 저장된 트리에 '강의1' 기본 노드가 없으면 생성하고, 첨부 배열 형태를 보정합니다.
export const ensureLectureOneFile = (nodes) => {
  const list = Array.isArray(nodes) ? [...nodes] : []
  const existingIndex = list.findIndex((node) => node?.id === 'lecture-1' || node?.name === '강의1')

  if (existingIndex === -1) {
    return [createLectureOneNode(), ...list]
  }

  const existingNode = list[existingIndex]
  list[existingIndex] = {
    ...existingNode,
    id: existingNode.id || 'lecture-1',
    content: existingNode.content || '',
    attachments: Array.isArray(existingNode.attachments) ? existingNode.attachments : []
  }

  return list
}

// 트리 안의 특정 노드를 찾아 updater 결과로 교체합니다.
export const updateNodeById = (nodes, targetId, updater) => {
  return nodes.map((node) => {
    if (node.id === targetId) {
      return updater(node)
    }

    if (node.children) {
      return {
        ...node,
        children: updateNodeById(node.children, targetId, updater)
      }
    }

    return node
  })
}

// 중첩된 폴더/파일 트리에서 id로 현재 파일 노드를 찾습니다.
const findNodeById = (nodes, targetId) => {
  for (const node of nodes) {
    if (node.id === targetId) return node
    if (node.children) {
      const found = findNodeById(node.children, targetId)
      if (found) return found
    }
  }
  return null
}

// 파일 트리 상태를 localStorage와 동기화해서 새로고침 후에도 목록을 유지합니다.
export function useFileTreeState() {
  const fileTree = ref([])
  const favorites = ref(new Set())
  const activeFileName = ref('강의1')
  const activeFileId = ref('lecture-1')

  const currentFileNode = computed(() => findNodeById(fileTree.value, activeFileId.value))
  const currentAttachments = computed(() => currentFileNode.value?.attachments || [])

  // 외부 컴포넌트에서 수정한 트리를 받아 기본 강의 노드를 보정합니다.
  const handleFileTreeUpdate = (nodes) => {
    fileTree.value = ensureLectureOneFile(nodes)
  }

  // 즐겨찾기 Set 상태를 갱신합니다.
  const handleFavoritesUpdate = (nextFavorites) => {
    favorites.value = nextFavorites
  }

  // 현재 열려 있는 파일 id와 이름을 바꿉니다.
  const handleFileSelect = (id, node) => {
    if (!node) return
    activeFileId.value = id
    activeFileName.value = node.name
  }

  onMounted(() => {
    const savedTree = localStorage.getItem(STORAGE_KEYS.fileTree)
    fileTree.value = savedTree ? ensureLectureOneFile(JSON.parse(savedTree)) : ensureLectureOneFile([])

    const savedFavs = localStorage.getItem(STORAGE_KEYS.favorites)
    if (savedFavs) {
      favorites.value = new Set(JSON.parse(savedFavs))
    }
  })

  watch(fileTree, (newVal) => {
    localStorage.setItem(STORAGE_KEYS.fileTree, JSON.stringify(newVal))
  }, { deep: true })

  watch(favorites, (newVal) => {
    localStorage.setItem(STORAGE_KEYS.favorites, JSON.stringify(Array.from(newVal)))
  }, { deep: true })

  return {
    fileTree,
    favorites,
    activeFileName,
    activeFileId,
    currentAttachments,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleFileSelect
  }
}
