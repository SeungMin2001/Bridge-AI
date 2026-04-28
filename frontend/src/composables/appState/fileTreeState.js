import { computed, onMounted, watch, ref } from 'vue'

// 홈/워크스페이스에서 사용하는 파일 트리, 즐겨찾기, 현재 선택 파일을 관리합니다.
const STORAGE_KEYS = {
  fileTree: 'lecto_file_tree',
  favorites: 'lecto_favorites'
}

const formatFileDate = (date = new Date()) => {
  return `${date.getFullYear()}. ${date.getMonth() + 1}. ${date.getDate()}. ${date.getHours() >= 12 ? '오후' : '오전'} ${date.getHours() % 12 || 12}:${date.getMinutes().toString().padStart(2, '0')}`
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

// 강의자료 첨부가 붙는 기본 파일 노드입니다.
const createLectureOneNode = () => ({
  id: 'lecture-1',
  type: 'file',
  fileKind: 'lecture',
  name: '강의1',
  date: formatFileDate(),
  content: '',
  attachments: [],
  recordings: [],
  weeks: [createWeekGroup(0)]
})

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

// 저장된 트리의 파일 노드 자료/녹음 배열 형태를 보정합니다.
export const ensureLectureOneFile = (nodes) => {
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
  const activeFileType = computed(() => currentFileNode.value?.fileKind || 'lecture')
  const currentAttachments = computed(() => collectFileMaterials(currentFileNode.value))
  const currentRecordings = computed(() => collectFileRecordings(currentFileNode.value))

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

  const loadLocalFileTree = () => {
    const savedTree = localStorage.getItem(STORAGE_KEYS.fileTree)
    fileTree.value = savedTree ? ensureLectureOneFile(JSON.parse(savedTree)) : ensureLectureOneFile([])
  }

  const loadWorkspaceTree = async () => {
    try {
      const response = await fetch('/workspace/tree')
      const result = await response.json()
      if (!response.ok || !result.ok || !Array.isArray(result.tree)) {
        throw new Error(result.error || '워크스페이스 목록을 불러오지 못했습니다.')
      }
      fileTree.value = ensureLectureOneFile(result.tree)
    } catch {
      loadLocalFileTree()
    }
  }

  onMounted(() => {
    loadWorkspaceTree()

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
    activeFileType,
    currentAttachments,
    currentRecordings,
    handleFileTreeUpdate,
    handleFavoritesUpdate,
    handleFileSelect
  }
}
