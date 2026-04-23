import { computed, onMounted, watch, ref } from 'vue'

const STORAGE_KEYS = {
  fileTree: 'lecto_file_tree',
  favorites: 'lecto_favorites'
}

const createLectureOneNode = () => ({
  id: 'lecture-1',
  type: 'file',
  name: '강의1',
  content: '',
  attachments: []
})

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

export function useFileTreeState() {
  const fileTree = ref([])
  const favorites = ref(new Set())
  const activeFileName = ref('강의1')
  const activeFileId = ref('lecture-1')

  const currentFileNode = computed(() => findNodeById(fileTree.value, activeFileId.value))
  const currentAttachments = computed(() => currentFileNode.value?.attachments || [])

  const handleFileTreeUpdate = (nodes) => {
    fileTree.value = ensureLectureOneFile(nodes)
  }

  const handleFavoritesUpdate = (nextFavorites) => {
    favorites.value = nextFavorites
  }

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
