import { computed, onMounted, watch, ref } from 'vue'
import { getWorkspaceSession, getWorkspaceTree, isWorkspaceUuid } from '../../api/workspaceApi.js'

// 홈/워크스페이스에서 사용하는 파일 트리, 즐겨찾기, 현재 선택 파일을 관리합니다.
const STORAGE_KEYS = {
  fileTree: 'lecto_file_tree',
  favorites: 'lecto_favorites',
  activeFileId: 'lecto_active_file_id',
  activeFileName: 'lecto_active_file_name',
  recentFileIds: 'lecto_recent_file_ids'
}

const KOREAN_WEEKDAYS = ['일요일', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일']

const getWeekStartDate = (date = new Date()) => {
  const nextDate = new Date(date)
  const day = nextDate.getDay()
  const diff = day === 0 ? -6 : 1 - day
  nextDate.setDate(nextDate.getDate() + diff)
  nextDate.setHours(0, 0, 0, 0)
  return nextDate
}

const formatDateKey = (date = new Date()) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

const formatWeekDateLabel = (date = new Date()) => {
  return `${date.getFullYear()}. ${date.getMonth() + 1}. ${date.getDate()}. ${KOREAN_WEEKDAYS[date.getDay()]}`
}

const getCurrentWeekKey = (date = new Date()) => formatDateKey(getWeekStartDate(date))

const createWeekGroup = (index = 0, date = new Date()) => {
  const weekStart = getWeekStartDate(date)

  return {
    id: `week-${formatDateKey(weekStart)}`,
    type: 'week',
    label: `${index + 1}주차`,
    weekKey: formatDateKey(weekStart),
    dateLabel: formatWeekDateLabel(date),
    createdAt: date.toISOString(),
    expanded: true,
    materialFolderExpanded: true,
    recordingFolderExpanded: true,
    materials: [],
    recordings: []
  }
}

const normalizeWeek = (week, index = 0) => {
  const materials = Array.isArray(week?.materials)
    ? week.materials
    : (Array.isArray(week?.attachments) ? week.attachments : [])
  const recordings = Array.isArray(week?.recordings) ? week.recordings : []
  const fallback = createWeekGroup(index)

  return {
    ...fallback,
    ...week,
    id: week?.id || `${fallback.id}-${index + 1}`,
    type: 'week',
    label: week?.label || `${index + 1}주차`,
    weekKey: week?.weekKey || fallback.weekKey,
    dateLabel: week?.dateLabel || fallback.dateLabel,
    expanded: week?.expanded ?? true,
    materialFolderExpanded: week?.materialFolderExpanded ?? true,
    recordingFolderExpanded: week?.recordingFolderExpanded ?? true,
    materials,
    recordings
  }
}

const containsItem = (items, target) => {
  if (!target) return false
  return items.some((item) => item?.id && target?.id ? item.id === target.id : item === target)
}

const mergeMissingItems = (baseItems, legacyItems) => {
  const nextItems = [...baseItems]
  legacyItems.forEach((item) => {
    if (!containsItem(nextItems, item)) nextItems.push(item)
  })
  return nextItems
}

const normalizeWeeksForFile = (node) => {
  const legacyAttachments = Array.isArray(node.attachments) ? node.attachments : []
  const legacyRecordings = Array.isArray(node.recordings) ? node.recordings : []
  const sourceWeeks = Array.isArray(node.weeks) ? node.weeks : []
  let weeks = sourceWeeks.map(normalizeWeek)

  if (!weeks.length) {
    weeks = [createWeekGroup(0)]
  }

  if (legacyAttachments.length || legacyRecordings.length) {
    const allWeekMaterials = weeks.flatMap((week) => week.materials || [])
    const allWeekRecordings = weeks.flatMap((week) => week.recordings || [])
    const firstWeek = weeks[0]

    weeks = [
      {
        ...firstWeek,
        materials: mergeMissingItems(firstWeek.materials || [], legacyAttachments.filter((item) => !containsItem(allWeekMaterials, item))),
        recordings: mergeMissingItems(firstWeek.recordings || [], legacyRecordings.filter((item) => !containsItem(allWeekRecordings, item)))
      },
      ...weeks.slice(1)
    ]
  }

  return weeks
}

export const collectFileMaterials = (node) => {
  if (!node || node.type !== 'file') return []
  const weekMaterials = Array.isArray(node.weeks)
    ? node.weeks.flatMap((week) => week.materials || [])
    : []

  return weekMaterials.length ? weekMaterials : (node.attachments || [])
}

export const collectFileRecordings = (node) => {
  if (!node || node.type !== 'file') return []
  const weekRecordings = Array.isArray(node.weeks)
    ? node.weeks.flatMap((week) => week.recordings || [])
    : []

  return weekRecordings.length ? weekRecordings : (node.recordings || [])
}

const ensureCurrentWeekSlot = (weeks, date = new Date()) => {
  const currentWeekKey = getCurrentWeekKey(date)
  const existingIndex = weeks.findIndex((week) => week.weekKey === currentWeekKey)

  if (existingIndex !== -1) {
    return {
      weeks,
      targetWeekId: weeks[existingIndex].id
    }
  }

  const nextWeek = createWeekGroup(weeks.length, date)

  return {
    weeks: [nextWeek, ...weeks],
    targetWeekId: nextWeek.id
  }
}

export const addMaterialToCurrentWeek = (node, material) => {
  const normalizedNode = normalizeNode(node)
  const { weeks, targetWeekId } = ensureCurrentWeekSlot(normalizedNode.weeks || [])

  return {
    ...normalizedNode,
    attachments: [material, ...(normalizedNode.attachments || []).filter((item) => item.id !== material.id)],
    weeks: weeks.map((week) => (
      week.id === targetWeekId
        ? {
            ...week,
            expanded: true,
            materialFolderExpanded: true,
            materials: [material, ...(week.materials || []).filter((item) => item.id !== material.id)]
          }
        : week
    ))
  }
}

export const addRecordingToCurrentWeek = (node, recording) => {
  const normalizedNode = normalizeNode(node)
  const { weeks, targetWeekId } = ensureCurrentWeekSlot(normalizedNode.weeks || [])

  return {
    ...normalizedNode,
    recordings: [recording, ...(normalizedNode.recordings || []).filter((item) => item.id !== recording.id)],
    weeks: weeks.map((week) => (
      week.id === targetWeekId
        ? {
            ...week,
            expanded: true,
            recordingFolderExpanded: true,
            recordings: [recording, ...(week.recordings || []).filter((item) => item.id !== recording.id)]
          }
        : week
    ))
  }
}

const normalizeNode = (node) => {
  if (!node) return node

  const normalized = {
    ...node,
    ...(node.type === 'file'
      ? {
          content: node.content || '',
          attachments: Array.isArray(node.attachments) ? node.attachments : [],
          recordings: Array.isArray(node.recordings) ? node.recordings : [],
          weeks: normalizeWeeksForFile(node)
        }
      : {})
  }

  if (Array.isArray(node.children)) {
    normalized.children = node.children.map(normalizeNode)
  }

  return normalized
}

const stripRecordingForStorage = (recording = {}) => {
  if (!recording || typeof recording !== 'object') return recording
  const { transcriptions, ...rest } = recording
  return {
    ...rest,
    transcriptionCount: Array.isArray(transcriptions)
      ? transcriptions.length
      : (rest.transcriptionCount || 0),
    resourcesLoaded: false
  }
}

const stripNodeForStorage = (node) => {
  if (!node || typeof node !== 'object') return node

  const nextNode = { ...node }
  if (nextNode.type === 'file') {
    nextNode.resourcesLoaded = false
    nextNode.recordings = Array.isArray(nextNode.recordings)
      ? nextNode.recordings.map(stripRecordingForStorage)
      : []
    nextNode.weeks = Array.isArray(nextNode.weeks)
      ? nextNode.weeks.map((week) => ({
          ...week,
          recordings: Array.isArray(week?.recordings)
            ? week.recordings.map(stripRecordingForStorage)
            : []
        }))
      : []
  }

  if (Array.isArray(nextNode.children)) {
    nextNode.children = nextNode.children.map(stripNodeForStorage)
  }
  return nextNode
}

// 저장된 트리의 파일 노드 자료/녹음 배열 형태 보정
export const normalizeFileTree = (nodes) => {
  const list = Array.isArray(nodes) ? nodes.map(normalizeNode) : []
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

// 중첩된 폴더/파일 트리에서 id로 현재 파일 노드 찾기
export const findNodeById = (nodes, targetId) => {
  for (const node of nodes) {
    if (node.id === targetId) return node
    if (node.children) {
      const found = findNodeById(node.children, targetId)
      if (found) return found
    }
  }
  return null
}

const findFirstFileNode = (nodes) => {
  for (const node of nodes) {
    if (node?.type === 'file') return node
    if (node?.children) {
      const found = findFirstFileNode(node.children)
      if (found) return found
    }
  }
  return null
}

const collectFileNodes = (nodes = []) => {
  const files = []
  nodes.forEach((node) => {
    if (node?.type === 'file') files.push(node)
    if (Array.isArray(node?.children)) {
      files.push(...collectFileNodes(node.children))
    }
  })
  return files
}

// 파일 트리 상태를 localStorage와 동기화해서 새로고침 후에도 목록을 유지합니다.
export function useFileTreeState() {
  const fileTree = ref([])
  const favorites = ref(new Set())
  const activeFileName = ref(localStorage.getItem(STORAGE_KEYS.activeFileName) || '')
  const activeFileId = ref(localStorage.getItem(STORAGE_KEYS.activeFileId) || '')
  const recentFileIds = ref([])
  const hydratingFileIds = new Set()

  const currentFileNode = computed(() => findNodeById(fileTree.value, activeFileId.value))
  const activeFileType = computed(() => currentFileNode.value?.fileKind || 'lecture')
  const currentAttachments = computed(() => collectFileMaterials(currentFileNode.value))
  const currentRecordings = computed(() => collectFileRecordings(currentFileNode.value))
  const recentFiles = computed(() => {
    const allFiles = collectFileNodes(fileTree.value)
    const byId = new Map(allFiles.map((file) => [file.id, file]))
    const recentNodes = recentFileIds.value
      .map((id) => byId.get(id))
      .filter(Boolean)
    const fallbackNodes = allFiles.filter((file) => !recentFileIds.value.includes(file.id))

    return [...recentNodes, ...fallbackNodes].slice(0, 3).map((file) => ({
      id: file.id,
      name: file.name || '이름 없는 파일',
      type: file.fileKind || file.type || 'file',
      date: file.date || '최근',
      node: file
    }))
  })

  // 외부 컴포넌트에서 수정한 트리 보정
  const handleFileTreeUpdate = (nodes) => {
    fileTree.value = normalizeFileTree(nodes)
  }

  // 즐겨찾기 Set 상태를 갱신합니다.
  const handleFavoritesUpdate = (nextFavorites) => {
    favorites.value = nextFavorites
  }

  // 현재 열려 있는 파일 id와 이름을 바꿉니다.
  const hydrateFileResources = async (id, node) => {
    if (!isWorkspaceUuid(id) || node?.type !== 'file' || node?.resourcesLoaded !== false) return
    if (hydratingFileIds.has(id)) return

    hydratingFileIds.add(id)
    try {
      const fullNode = await getWorkspaceSession(id)
      fileTree.value = updateNodeById(fileTree.value, id, () => normalizeNode(fullNode))
      if (activeFileId.value === id) {
        activeFileName.value = fullNode.name || activeFileName.value
      }
    } catch (error) {
      console.warn('[workspace] file resource hydration failed:', error)
    } finally {
      hydratingFileIds.delete(id)
    }
  }

  const handleFileSelect = (id, node) => {
    if (!node) return
    activeFileId.value = id
    activeFileName.value = node.name
    if (node.type === 'file' && id) {
      recentFileIds.value = [
        id,
        ...recentFileIds.value.filter((fileId) => fileId !== id)
      ].slice(0, 8)
      hydrateFileResources(id, node)
    }
  }

  const syncActiveFileWithTree = () => {
    const activeNode = findNodeById(fileTree.value, activeFileId.value)
    if (activeNode) {
      activeFileName.value = activeNode.name
      return
    }

    const firstFile = findFirstFileNode(fileTree.value)
    if (firstFile) {
      activeFileId.value = firstFile.id
      activeFileName.value = firstFile.name
      return
    }

    activeFileId.value = ''
    activeFileName.value = ''
  }

  const loadLocalFileTree = () => {
    const savedTree = localStorage.getItem(STORAGE_KEYS.fileTree)
    fileTree.value = savedTree ? normalizeFileTree(JSON.parse(savedTree)) : normalizeFileTree([])
  }

  const loadWorkspaceTree = async () => {
    try {
      fileTree.value = normalizeFileTree(await getWorkspaceTree())
    } catch {
      loadLocalFileTree()
    }
    syncActiveFileWithTree()
    const activeNode = findNodeById(fileTree.value, activeFileId.value)
    if (activeNode) hydrateFileResources(activeFileId.value, activeNode)
  }

  onMounted(() => {
    loadWorkspaceTree()

    const savedFavs = localStorage.getItem(STORAGE_KEYS.favorites)
    if (savedFavs) {
      favorites.value = new Set(JSON.parse(savedFavs))
    }

    const savedRecentFileIds = localStorage.getItem(STORAGE_KEYS.recentFileIds)
    if (savedRecentFileIds) {
      recentFileIds.value = JSON.parse(savedRecentFileIds)
    }
  })

  watch(fileTree, (newVal) => {
    localStorage.setItem(STORAGE_KEYS.fileTree, JSON.stringify(newVal.map(stripNodeForStorage)))
  }, { deep: true })

  watch(favorites, (newVal) => {
    localStorage.setItem(STORAGE_KEYS.favorites, JSON.stringify(Array.from(newVal)))
  }, { deep: true })

  watch(activeFileId, (newVal) => {
    if (newVal) localStorage.setItem(STORAGE_KEYS.activeFileId, newVal)
    else localStorage.removeItem(STORAGE_KEYS.activeFileId)
  })

  watch(activeFileName, (newVal) => {
    if (newVal) localStorage.setItem(STORAGE_KEYS.activeFileName, newVal)
    else localStorage.removeItem(STORAGE_KEYS.activeFileName)
  })

  watch(recentFileIds, (newVal) => {
    localStorage.setItem(STORAGE_KEYS.recentFileIds, JSON.stringify(newVal))
  }, { deep: true })

  return {
    fileTree,
    favorites,
    recentFiles,
    activeFileName,
    activeFileId,
    activeFileType,
    currentAttachments,
    currentRecordings,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleFileSelect
  }
}
